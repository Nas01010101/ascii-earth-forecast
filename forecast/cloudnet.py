"""Forecasts a day's mean cloud cover, temperature and pressure with one U-Net each, run with numpy in the daily job.

The weights are trained offline on ERA5 (research/weather) and frozen here. A forecast reads the 30 days before
its target, 6-hourly, so replaying a past day gives what would have been issued on it. The formulas match
research/weather, and research/weather/test_production.py checks them against the training code.
"""
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.special import erf

WEIGHTS = Path(__file__).with_name("cloud_weights.npz")
OTHER_WEIGHTS = {name: Path(__file__).with_name(f"{name}_weights.npz") for name in ("temp", "pressure")}
VARIABLES = ("cloud", "temp", "precip", "pressure", "u", "v", "water")
STEPS_PER_DAY = 4
HOURS_PER_STEP = 6
RAIN_HOURS = 6
LAGS = 3
DAYS_NEEDED = 30
GROUPNORM_EPS = 1e-5
DOWN_BLOCKS = 4
UP_BLOCKS = 3


def load(path=WEIGHTS):
    with np.load(path) as saved:
        return {name: saved[name].astype(np.float32) for name in saved.files}


def load_others(saved, files=None):
    """The networks of the other targets, each its own layers over the scales of `saved`. No file means no network."""
    found = {}
    for name, path in (OTHER_WEIGHTS if files is None else files).items():
        if Path(path).exists():
            found[name] = {**saved, **load(path)}
    return found


def day_steps(hours, fields, day):
    """One UTC day as (variables, steps, lat, lon) in the training data's form, or None if its hours are not all held.

    ERA5 rain is the 6 hours ending at each step and an Open-Meteo hour holds the hour ending at its stamp, so the
    day's first step reads the day before. Wind direction names where the wind comes from, and u and v where it goes.
    """
    first = np.searchsorted(hours, np.datetime64(day, "h"))
    if first < RAIN_HOURS - 1 or first + 18 >= len(hours) or hours[first] != np.datetime64(day, "h"):
        return None
    at = first + np.arange(STEPS_PER_DAY) * HOURS_PER_STEP
    rain = np.stack([fields["precip"][i + 1 - RAIN_HOURS : i + 1].sum(0, dtype=np.float64) for i in at])
    speed, toward = fields["speed"][at], np.radians(fields["direction"][at])
    found = {"precip": rain, "u": -speed * np.sin(toward), "v": -speed * np.cos(toward)}
    return np.stack([found[n] if n in found else fields[n][at] for n in VARIABLES]).astype(np.float32)


def view(steps, when):
    """The 30 days of steps before a target day, oldest first, or None if any is missing."""
    days = [(when - timedelta(days=back)).isoformat() for back in range(DAYS_NEEDED, 0, -1)]
    if any(d not in steps for d in days):
        return None
    return np.concatenate([steps[d] for d in days], axis=1)


def _doy(when):
    return when.timetuple().tm_yday - 1


def inputs(held, when, saved, target="cloud"):
    """(channels, lat, lon) and the base forecast in the target's unit, as research/weather/features.py builds them."""
    start = when - timedelta(days=DAYS_NEEDED)
    doys = np.array([_doy(start + timedelta(days=d)) for d in range(DAYS_NEEDED)])
    days = {}
    parts = []
    for k, name in enumerate(VARIABLES):
        scaled = (held[k] - saved[f"clim_{name}"][np.repeat(doys, STEPS_PER_DAY)]) / saved[f"std_{name}"]
        days[name] = scaled.reshape(DAYS_NEEDED, STEPS_PER_DAY, *scaled.shape[1:]).mean(1)
        parts += [scaled[::-1][:STEPS_PER_DAY], days[name][::-1][:LAGS]]
    trail = {n: days[n].mean(0, dtype=np.float64).astype(np.float32) for n in {"cloud", "temp", target}}
    day = _doy(when)
    mean = saved[f"clim_{target}"][day]
    std = saved[f"std_{target}"]
    centre = 0.0 if target == "cloud" else float(saved[f"clim_{target}"].mean())
    angle = 2 * np.pi * day / 366
    shape = mean.shape
    extra = [trail["cloud"], trail["temp"], *saved["fixed"], np.full(shape, math.sin(angle)), np.full(shape, math.cos(angle)),
             (mean - centre) / std]
    return np.concatenate([*parts, np.stack(extra)]).astype(np.float32), mean + trail[target] * std


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _conv(x, weight, bias):
    """Slides (out, in, k, k) over (in, lat, lon), keeping only full windows."""
    size = weight.shape[-1]
    windows = np.lib.stride_tricks.sliding_window_view(x, (size, size), axis=(1, 2))
    return np.tensordot(weight, windows, axes=([1, 2, 3], [0, 3, 4])) + bias[:, None, None]


def _wrap(x):
    """Longitude wraps and latitude repeats its edge row."""
    return np.pad(np.pad(x, ((0, 0), (0, 0), (1, 1)), mode="wrap"), ((0, 0), (1, 1), (0, 0)), mode="edge")


def _norm(net, name, x):
    y = (x - x.mean()) / math.sqrt(x.var() + GROUPNORM_EPS)
    return y * net[f"{name}.weight"][:, None, None] + net[f"{name}.bias"][:, None, None]


def _block(net, name, x):
    y = _gelu(_norm(net, f"{name}.norm1", _conv(_wrap(x), net[f"{name}.first.weight"], net[f"{name}.first.bias"])))
    skip = _conv(x, net[f"{name}.skip.weight"], net[f"{name}.skip.bias"]) if f"{name}.skip.weight" in net else x
    return _gelu(_norm(net, f"{name}.norm2", _conv(_wrap(y), net[f"{name}.second.weight"], net[f"{name}.second.bias"])) + skip)


def _halve(x):
    lat, lon = x.shape[1] // 2, x.shape[2] // 2
    return x.reshape(x.shape[0], lat, 2, lon, 2).mean((2, 4))


def network(x, net):
    """The change from the base forecast, in units of the target's scale."""
    x = np.concatenate([x, net["place"][0]])
    kept = []
    for k in range(DOWN_BLOCKS):
        x = _block(net, f"down.{k}", x if k == 0 else _halve(kept[-1]))
        kept.append(x)
    for k in range(UP_BLOCKS):
        twice = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = _block(net, f"up.{k}", np.concatenate([twice, kept[-2 - k]]))
    return _conv(x, net["out.weight"], net["out.bias"])[0]


def predict(held, when, saved, target="cloud"):
    """The target variable's daily mean in its own unit (cloud in percent, not yet clipped)."""
    x, base = inputs(held, when, saved, target)
    return base + network(x, saved) * saved[f"std_{target}"]

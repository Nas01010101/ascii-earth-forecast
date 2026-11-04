"""Corrects each satellite's SGP4 error with a residual network, run with numpy in the daily job.

The weights are trained offline on the public catalogue (research/sat) and frozen here. A forecast
starts from one element set and reads only sets at or before it, so replaying a past day gives what
would have been issued on it. The formulas match research/sat, and research/sat/test_production.py
checks them against the training code.
"""
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.special import erf

from . import orbits

WEIGHTS = Path(__file__).with_name("drift_weights.npz")
JD_UNIX_EPOCH = 2440587.5
SGP4_EPOCH_JD = 2433281.5
HORIZONS = (1, 3, 7)
LOOKBACKS = (1.0, 3.0, 7.0, 15.0)
LOOKBACK_TOLERANCE = 1.0
TRUTH_TOLERANCE = 1.0
UPDATE_WINDOW = 30.0
EARTH_RADIUS_KM = 6378.135
MU = 398600.8  # WGS72, km^3/s^2, the constant SGP4 uses here
MAX_DRIFT_KM = 40000.0  # a diverged hindcast implies drifts up to 1e13 km
NETWORK_ROWS = 65536

ELEMENTS = ["no_rev", "ecco", "inclo", "sin_argp", "cos_argp", "bstar", "ndot", "nddot", "perigee_km", "apogee_km", "updates_30d"]
HISTORY = [f"{name}_{int(a)}" for a in LOOKBACKS for name in ("age", "height", "along", "cross", "dno", "dbstar")]
WEATHER = ["f107", "f107_81", "ap"]  # The chosen network was trained with these zeroed, so they stay zero here.
FEATURES = ELEMENTS + HISTORY + WEATHER
INDEX = {name: i for i, name in enumerate(FEATURES)}
LOGGED = [f for f in HISTORY if f.split("_")[0] in ("height", "along", "cross")] + ["perigee_km", "apogee_km", "updates_30d"]
RATES = ["bstar", "ndot", "nddot"] + [f for f in HISTORY if f.split("_")[0] in ("dno", "dbstar")]
AGES = [f for f in HISTORY if f.startswith("age_")]


def epoch_day(row):
    """Days since the SGP4 epoch date, from a row's ISO time."""
    moment = datetime.fromisoformat(row["EPOCH"]).replace(tzinfo=timezone.utc)
    return moment.timestamp() / 86400 + JD_UNIX_EPOCH - SGP4_EPOCH_JD


def sets_by_object(snapshots):
    """Gathers snapshots into {norad: (epochs, rows)}, oldest first, each element set once."""
    found = {}
    for snapshot in snapshots:
        for norad, row in snapshot.items():
            found.setdefault(norad, {})[row["EPOCH"]] = row
    out = {}
    for norad, sets in found.items():
        rows = sorted(sets.values(), key=epoch_day)
        out[norad] = np.array([epoch_day(r) for r in rows]), rows
    return out


def _curvilinear(pred_r, pred_v, true_r):
    """True minus predicted as height, along-track arc and cross-track arc, km, plus the predicted radius.

    Only the predicted state defines the frame, so a correction can be applied without the truth.
    """
    rn = math.sqrt(pred_r[0] ** 2 + pred_r[1] ** 2 + pred_r[2] ** 2)
    rx, ry, rz = pred_r[0] / rn, pred_r[1] / rn, pred_r[2] / rn
    wx = pred_r[1] * pred_v[2] - pred_r[2] * pred_v[1]
    wy = pred_r[2] * pred_v[0] - pred_r[0] * pred_v[2]
    wz = pred_r[0] * pred_v[1] - pred_r[1] * pred_v[0]
    wn = math.sqrt(wx * wx + wy * wy + wz * wz)
    wx /= wn; wy /= wn; wz /= wn
    sx, sy, sz = wy * rz - wz * ry, wz * rx - wx * rz, wx * ry - wy * rx
    a = true_r[0] * rx + true_r[1] * ry + true_r[2] * rz
    b = true_r[0] * sx + true_r[1] * sy + true_r[2] * sz
    c = true_r[0] * wx + true_r[1] * wy + true_r[2] * wz
    tn = math.sqrt(a * a + b * b + c * c)
    return tn - rn, math.atan2(b, a) * rn, math.asin(max(-1.0, min(1.0, c / tn))) * rn, rn


def _error(older, newer, days):
    e0, r0, v0 = older.sgp4_tsince(days * 1440.0)
    e1, r1, _ = newer.sgp4_tsince(0.0)
    return None if e0 or e1 else _curvilinear(r0, v0, r1)


def features(epochs, rows, i, held_days):
    """The row of FEATURES for a forecast starting at set i. Sets newer than i are never read."""
    start = rows[i]
    no_rev, e = float(start["MEAN_MOTION"]), float(start["ECCENTRICITY"])
    bstar = float(start["BSTAR"])
    n = no_rev * 2.0 * np.pi / 86400.0
    a = (MU / max(n * n, 1e-16)) ** (1.0 / 3.0)
    argp = np.radians(float(start["ARG_OF_PERICENTER"]))
    t = epochs[i]
    # Snapshots may not reach back a full window yet, so the count is scaled up to one.
    updates = (i - np.searchsorted(epochs, t - UPDATE_WINDOW)) * UPDATE_WINDOW / min(UPDATE_WINDOW, max(held_days, 1))
    out = [no_rev, e, float(start["INCLINATION"]), np.sin(argp), np.cos(argp), bstar, float(start["MEAN_MOTION_DOT"]),
           float(start["MEAN_MOTION_DDOT"]), a * (1 - e) - EARTH_RADIUS_KM, a * (1 + e) - EARTH_RADIUS_KM, updates]
    newest = orbits.satrec(start)
    for back in LOOKBACKS:
        j = int(np.searchsorted(epochs, t - back, side="right")) - 1
        found = None
        if j >= 0 and t - back - epochs[j] <= LOOKBACK_TOLERANCE:
            found = _error(orbits.satrec(rows[j]), newest, float(t - epochs[j]))
        if found is None:
            out += [np.nan] * 6
        else:
            out += [float(t - epochs[j]), *found[:3], no_rev - float(rows[j]["MEAN_MOTION"]), bstar - float(rows[j]["BSTAR"])]
    return out + [0.0] * len(WEATHER)


def own_drift(x, dt):
    """Along-track error the object's own hindcasts imply at dt: along = c * age^2 through the origin."""
    top = np.zeros(len(x))
    bottom = np.zeros(len(x))
    for back in LOOKBACKS:
        age = x[:, INDEX[f"age_{int(back)}"]].astype(np.float64)
        along = x[:, INDEX[f"along_{int(back)}"]].astype(np.float64)
        ok = np.isfinite(age) & np.isfinite(along)
        top[ok] += along[ok] * age[ok] ** 2
        bottom[ok] += age[ok] ** 4
    c = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
    return (c * dt.astype(np.float64) ** 2).astype(np.float32)


def _slog(v):
    return np.sign(v) * np.log1p(np.abs(v))


def inputs(x, dt):
    """Network inputs: compressed features, the horizon, the drift estimate, and one missing flag per lookback."""
    missing = ~np.isfinite(x)
    out = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    for name in LOGGED:
        out[:, INDEX[name]] = _slog(out[:, INDEX[name]])
    for name in RATES:
        out[:, INDEX[name]] = _slog(out[:, INDEX[name]] * 1e4)
    out[:, INDEX["no_rev"]] /= 16.0
    out[:, INDEX["inclo"]] /= 180.0
    for name in AGES:
        out[:, INDEX[name]] /= 30.0
    days = dt.astype(np.float64)
    extra = np.stack([np.log(days + 0.1), days / 30.0, _slog(own_drift(x, dt).astype(np.float64))], axis=1)
    flags = missing[:, [INDEX[name] for name in AGES]].astype(np.float64)
    return np.concatenate([out, extra, flags], axis=1).astype(np.float32)


def load_nets(path=WEIGHTS):
    """One dict of arrays per seed, keyed by the training code's parameter names."""
    nets = {}
    with np.load(path) as saved:
        for key in saved.files:
            seed, name = key.split("/", 1)
            nets.setdefault(seed, {})[name] = saved[key]
    return list(nets.values())


def _linear(net, name, x):
    return x @ net[f"{name}.weight"].T + net[f"{name}.bias"]


def _layer_norm(net, name, x):
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + 1e-5) * net[f"{name}.weight"] + net[f"{name}.bias"]


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def forward(net, x, drift, scale):
    """Height, along and cross corrections in km. All-zero outputs would leave SGP4 unchanged."""
    depth = 1 + max(int(k.split(".")[1]) for k in net if k.startswith("blocks."))
    h = _linear(net, "first", (x - net["mean"]) / net["std"])
    for b in range(depth):
        y = _gelu(_linear(net, f"blocks.{b}.net.1", _layer_norm(net, f"blocks.{b}.net.0", h)))
        h = h + _linear(net, f"blocks.{b}.net.4", y)
    o = _linear(net, "last.1", _layer_norm(net, "last.0", h))
    return np.stack([o[:, 2] * scale * 0.1, o[:, 0] * drift + o[:, 1] * scale, o[:, 3] * scale * 0.1], axis=1)


def correct(nets, x, dt):
    """The mean correction of all nets for raw feature rows at horizons dt."""
    net_in = inputs(x, dt)
    drift = np.clip(own_drift(x, dt), -MAX_DRIFT_KM, MAX_DRIFT_KM)
    scale = np.maximum(np.abs(drift), 1.0)
    parts = [np.concatenate([forward(net, net_in[i:i + NETWORK_ROWS], drift[i:i + NETWORK_ROWS], scale[i:i + NETWORK_ROWS])
                             for i in range(0, len(x), NETWORK_ROWS)]) for net in nets]
    return np.mean(parts, axis=0)


def miss_km(radius, truth, correction):
    """3D distance between the true point and the corrected point, both as (height, along, cross) in km."""
    def point(d):
        rho = radius + d[:, 0]
        u, w = d[:, 1] / radius, d[:, 2] / radius
        return np.stack([rho * np.cos(w) * np.cos(u), rho * np.cos(w) * np.sin(u), rho * np.sin(w)], axis=1)
    return np.linalg.norm(point(truth) - point(correction), axis=1)


def score_start(known, start, horizon, held_days, nets):
    """Scores forecasts that start from the sets in one snapshot, against the first later set past the horizon.

    Returns {n, sgp4_km, model_km}, medians, or None if no satellite has a usable later set.
    """
    x, truth, radius = [], [], []
    for norad, row in start.items():
        if norad not in known:
            continue
        epochs, rows = known[norad]
        i = int(np.searchsorted(epochs, epoch_day(row)))
        j = int(np.searchsorted(epochs, epochs[i] + horizon))
        if j >= len(epochs) or epochs[j] - (epochs[i] + horizon) > TRUTH_TOLERANCE:
            continue
        found = _error(orbits.satrec(rows[i]), orbits.satrec(rows[j]), float(epochs[j] - epochs[i]))
        if found is None:
            continue
        x.append(features(epochs, rows, i, held_days))
        truth.append(found[:3])
        radius.append(found[3])
    if not x:
        return None
    x = np.array(x, dtype=np.float32)
    truth, radius = np.array(truth), np.array(radius)
    corrected = correct(nets, x, np.full(len(x), float(horizon), np.float32)).astype(np.float64)
    raw = miss_km(radius, truth, np.zeros_like(truth))
    return {"n": len(x), "sgp4_km": round(float(np.median(raw)), 2),
            "model_km": round(float(np.median(miss_km(radius, truth, corrected))), 2)}


def score_day(snapshots, day, nets):
    """Scores the forecasts checked on one day, by horizon, from {date: parsed snapshot}. Later snapshots are not read."""
    known = sets_by_object([snapshots[d] for d in sorted(snapshots) if d <= day])
    first = min(snapshots)
    out = {}
    for horizon in HORIZONS:
        begun = day - timedelta(days=horizon)
        found = score_start(known, snapshots[begun], horizon, (begun - first).days, nets) if begun in snapshots else None
        if found:
            out[str(horizon)] = found
    return out

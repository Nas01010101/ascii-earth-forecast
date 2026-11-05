"""Rewrites a network's input scales from Open-Meteo steps, leaving its weights alone.

usage: rescale_stats.py NET.pt FIELDS.npz FIRST_DAY LAST_DAY OUT.pt
For each variable the steps of FIRST_DAY..LAST_DAY give a mean and a spread of the network's own scaled inputs. The
climatology moves by that mean and the scale grows by that spread, so the inputs reach the network as ERA5's did.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import torch

from common import STEPS_PER_DAY
from features import doy_of


def step_days(first: date, days: int) -> np.ndarray:
    """Climatology index of each step, the one features.Frames uses."""
    return np.repeat([doy_of(first + timedelta(days=d)) for d in range(days)], STEPS_PER_DAY)


def rescaled(stats: dict, fields: dict, first: date, start: int, end: int, names: tuple) -> dict:
    """New stats from steps of days `start` (inclusive) to `end` (exclusive) of `fields`."""
    out = dict(stats)
    doys = step_days(first, end)[start * STEPS_PER_DAY:]
    for name in names:
        held = fields[name][start * STEPS_PER_DAY:end * STEPS_PER_DAY]
        clim, std = np.asarray(stats[f"clim_{name}"]), float(stats[f"std_{name}"])
        scaled = (held - clim[doys]) / std
        out[f"clim_{name}"] = clim + float(scaled.mean()) * std
        out[f"std_{name}"] = np.asarray(std * float(scaled.std()), np.float32)
    return out


def main(path: str, fields_path: str, first_day: str, last_day: str, out: str) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    held = np.load(fields_path)
    first = date.fromisoformat(str(held["first_date"]))
    start = (date.fromisoformat(first_day) - first).days
    end = (date.fromisoformat(last_day) - first).days + 1
    names = tuple(saved["spec"]["variables"])
    saved["stats"] = rescaled(saved["stats"], {name: held[name] for name in names}, first, start, end, names)
    torch.save(saved, out)


if __name__ == "__main__":
    main(*sys.argv[1:])

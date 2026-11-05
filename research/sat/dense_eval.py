"""Scores nets told the real gap against nets read at fixed steps and blended, as the viewer does, per gap bucket.

usage: dense_eval.py DENSE_FOLDER OUT_CSV ANYTIME.pt[,ANYTIME.pt] STEPS.pt[,STEPS.pt]
Rows are 2024 on the validation shards. The blend answers nothing before the first step or past the last, so it
equals SGP4 there, as the viewer shows no network point.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from daily_check import load_net
from data import load_folder
from dt_ab import per_object_gain
from sat_common import miss_km
from train_net import END, VAL_SHARDS, VAL_START, predict, tensors

STEPS = (1.0, 3.0, 7.0, 15.0)  # the horizons in corrections.bin
HEADER = ["bucket", "rows", "objects", "sgp4_km", "anytime_km", "steps_km", "anytime_gain", "steps_gain", "anytime_worse", "steps_worse"]


def in_steps(dt: np.ndarray, at: dict) -> np.ndarray:
    """correct.js correctionAt: linear in dt squared between steps, zero outside them."""
    steps = sorted(at)
    out = np.zeros_like(at[steps[0]])
    for low, high in zip(steps, steps[1:]):
        m = (dt >= low) & (dt <= high)
        k = ((dt[m] ** 2 - low ** 2) / (high ** 2 - low ** 2))[:, None]
        out[m] = at[low][m] * (1 - k) + at[high][m] * k
    return out


def mean_of(nets: list, d: dict, device: str) -> np.ndarray:
    return np.mean([predict(net, tensors(d, args), device) for net, args in nets], axis=0)


def worse_share(norad: np.ndarray, base: np.ndarray, model: np.ndarray) -> float:
    """Share of objects whose mean miss got larger."""
    objects, index = np.unique(norad, return_inverse=True)
    return float(np.mean(np.bincount(index, model) > np.bincount(index, base)))


def main(folder: str, out: str, anytime: str, steps: str) -> None:
    device = "cuda"
    d = load_folder(folder, VAL_START, END, VAL_SHARDS)
    any_nets = [load_net(path, device) for path in anytime.split(",")]
    step_nets = [load_net(path, device) for path in steps.split(",")]
    y, radius = d["y"].astype(np.float64), d["radius"].astype(np.float64)
    dt = d["dt"].astype(np.float64)
    at = {}
    for h in STEPS:  # fixed_dt nets read the horizon, the others dt: both are set to the step
        fill = np.full(len(dt), h, d["dt"].dtype)
        at[h] = mean_of(step_nets, {**d, "dt": fill, "horizon": fill}, device).astype(np.float64)
    base = miss_km(radius, y, np.zeros_like(y))
    any_miss = miss_km(radius, y, mean_of(any_nets, d, device).astype(np.float64))
    step_miss = miss_km(radius, y, in_steps(dt, at))
    rows = []
    for bucket in np.unique(d["horizon"]):
        m = d["horizon"] == bucket
        n = d["norad"][m]
        rows.append([float(bucket), int(m.sum()), len(np.unique(n)), round(float(np.median(base[m])), 3),
                     round(float(np.median(any_miss[m])), 3), round(float(np.median(step_miss[m])), 3),
                     round(per_object_gain(n, base[m], any_miss[m]), 2), round(per_object_gain(n, base[m], step_miss[m]), 2),
                     round(worse_share(n, base[m], any_miss[m]), 3), round(worse_share(n, base[m], step_miss[m]), 3)])
        print(*rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:5])

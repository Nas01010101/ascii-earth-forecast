"""Validation corrections broken down by horizon, orbit regime and whether training saw the object.

usage: report.py FIT_PARQUET OUT_CSV NAME=OOF_NPY [NAME=OOF_NPY ...]
Each OOF file holds corrections from train_net.py, in the row order of its validation load.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from build_samples import HORIZONS
from data import col, load
from diagnose import REGIMES
from sat_common import miss_km
from score import per_object
from train_net import END, TRAIN_SHARDS, VAL_SHARDS, VAL_START

HEADER = ["name", "horizon", "regime", "objects", "mean_gain", "median_gain", "share_improved", "raw_km", "model_km", "seen_mean", "unseen_mean"]


def row(name: str, h: float, regime: str, d: dict, y: np.ndarray, radius: np.ndarray, c: np.ndarray, m: np.ndarray, trained_on: set = TRAIN_SHARDS) -> tuple:
    ids, gain = per_object(d["norad"][m], radius[m], y[m], c[m])
    seen = np.isin(ids % 64, list(trained_on))
    raw_km, model_km = miss_km(radius[m], y[m], np.zeros_like(y[m])), miss_km(radius[m], y[m], c[m])
    finite = raw_km < 1e5  # diverged SGP4 rows would otherwise set the pooled mean alone
    return (name, h, regime, len(gain), round(float(gain.mean()), 2), round(float(np.median(gain)), 2), round(float((gain > 0).mean()), 3),
            round(float(raw_km[finite].mean()), 2), round(float(model_km[finite].mean()), 2), round(float(gain[seen].mean()), 2), round(float(gain[~seen].mean()), 2))


def main(fit: str, out: str, specs: list) -> None:
    d = load(fit, VAL_START, END, shards=VAL_SHARDS)
    y, radius, no_rev = d["y"].astype(np.float64), d["radius"].astype(np.float64), col(d, "no_rev")
    rows = []
    for spec in specs:
        name, path = spec.split("=", 1)
        c = np.load(path).astype(np.float64)
        assert len(c) == len(y), f"{path} has {len(c)} rows, validation has {len(y)}"
        for h in HORIZONS:
            for regime, low, high in REGIMES:
                m = (d["horizon"] == h) & (no_rev >= low) & (no_rev < high)
                if m.any():
                    rows.append(row(name, h, regime, d, y, radius, c, m))
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(*HEADER)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

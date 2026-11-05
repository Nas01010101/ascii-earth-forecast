"""Compares saved validation corrections by how late the truth set arrived.

usage: dt_ab.py FIT_PARQUET OUT_CSV NAME=OOF.npy [NAME=OOF.npy ...]
Rows and order match train_net.main's validation set. Each file gets one row per horizon and lateness bucket: the
median miss and the percent gain in mean miss over raw SGP4, per object as in score.py, plus the along-track slope.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from data import load
from dt_check import slope
from sat_common import miss_km
from train_net import END, VAL_SHARDS, VAL_START

HORIZONS = (1, 3, 7, 15, 30)
LATE = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01))
HEADER = ["name", "horizon", "late_from", "late_to", "rows", "sgp4_km", "model_km", "gain_mean_pct", "slope"]


def per_object_gain(norad: np.ndarray, base: np.ndarray, model: np.ndarray) -> float:
    """Mean over objects of the percent gain in mean miss, so one busy object counts once."""
    objects, index = np.unique(norad, return_inverse=True)
    raw = np.bincount(index, base) / np.bincount(index)
    got = np.bincount(index, model) / np.bincount(index)
    return float(np.mean(100 * (raw - got) / np.maximum(raw, 1e-9)))


def main(fit: str, out: str, pairs: list) -> None:
    d = load(fit, VAL_START, END, shards=VAL_SHARDS)
    y, radius = d["y"].astype(np.float64), d["radius"].astype(np.float64)
    base = miss_km(radius, y, np.zeros_like(y))
    rows = []
    for pair in pairs:
        name, path = pair.split("=")
        c = np.load(path).astype(np.float64)
        model = miss_km(radius, y, c)
        for h in HORIZONS:
            for low, high in LATE:
                late = d["dt"] - h
                m = (d["horizon"] == h) & (late >= low) & (late < high)
                if not m.any():
                    continue
                rows.append([name, h, low, high, int(m.sum()), round(float(np.median(base[m])), 3),
                             round(float(np.median(model[m])), 3), round(per_object_gain(d["norad"][m], base[m], model[m]), 2),
                             round(slope(y[m, 1], c[m, 1]), 3)])
                print(*rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

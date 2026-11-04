"""Where the non-neural corrections help and hurt on the 2024 validation period.

usage: diagnose.py FIT_PARQUET OUT_CSV
Prints, per correction, horizon and orbit regime: objects, mean and median percent gain per object, the share
of objects improved, and pooled mean km. Then how well the object's own drift estimate predicts the truth.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from baselines import END, VAL_SHARDS, VAL_START, linear_walk
from build_samples import HORIZONS
from data import col, load, own_drift
from sat_common import miss_km
from score import per_object

REGIMES = (("all", 0.0, 99.0), ("LEO", 11.25, 99.0), ("MEO", 1.5, 11.25), ("GEO", 0.0, 1.5))
DRIFT_BINS = (0.0, 1.0, 10.0, 100.0, 1000.0, 1e12)


def main(fit: str, out: str) -> None:
    d = load(fit, VAL_START, END, shards=VAL_SHARDS)
    y = d["y"].astype(np.float64)
    radius = d["radius"].astype(np.float64)
    no_rev = col(d, "no_rev")
    drift = own_drift(d).astype(np.float64)
    own = np.zeros_like(y)
    own[:, 1] = drift
    corrections = {"own": own, "linear": linear_walk(d).astype(np.float64)}

    rows = []
    for name, c in corrections.items():
        for h in HORIZONS:
            for regime, low, high in REGIMES:
                m = (d["horizon"] == h) & (no_rev >= low) & (no_rev < high)
                if not m.any():
                    continue
                _, gain = per_object(d["norad"][m], radius[m], y[m], c[m])
                raw_km, model_km = miss_km(radius[m], y[m], np.zeros_like(y[m])), miss_km(radius[m], y[m], c[m])
                finite = raw_km < 1e5  # diverged SGP4 rows would otherwise set the pooled mean alone
                rows.append((name, h, regime, len(gain), round(float(gain.mean()), 2), round(float(np.median(gain)), 2), round(float((gain > 0).mean()), 3),
                             round(float(raw_km[finite].mean()), 2), round(float(model_km[finite].mean()), 2)))
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["correction", "horizon", "regime", "objects", "mean_gain", "median_gain", "share_improved", "raw_km", "model_km"])
        writer.writerows(rows)
    for row in rows:
        print(*row)

    m = (d["horizon"] == 15.0) & (np.abs(y[:, 1]) < 5000.0) & (no_rev >= 11.25)
    truth, guess = y[m, 1], drift[m]
    print("LEO 15d, |along| < 5000 km: rows", int(m.sum()), "own-drift slope", round(float(truth @ guess / max(guess @ guess, 1e-9)), 3))
    for low, high in zip(DRIFT_BINS[:-1], DRIFT_BINS[1:]):
        b = (np.abs(guess) >= low) & (np.abs(guess) < high)
        if b.sum() > 100:
            print(f"|drift| {low:g}-{high:g} km: rows {int(b.sum())} slope {truth[b] @ guess[b] / max(guess[b] @ guess[b], 1e-9):.3f} median |along| {np.median(np.abs(truth[b])):.1f} median |drift| {np.median(np.abs(guess[b])):.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

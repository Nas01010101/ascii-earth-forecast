"""Scores nets trained on the nominal horizon by how late the truth set arrived. Run only after the reveal.

usage: dt_check.py DAILY_SAMPLES_DIR OUT_CSV NET.pt [NET.pt ...]
A row at horizon h has its truth at dt between h and h + 1 day, and the nets are told dt = h. The slope is the
least-squares factor that would best scale the along-track correction: under 1 means the net overshoots at that lateness.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from daily_check import load_net
from daily_samples import END, START
from data import load_folder
from sat_common import miss_km
from train_net import predict, tensors

HORIZONS = (1, 3, 7)
LATE = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01))
SHARDS = set(range(0, 64, 2))  # half the objects, as in history_check.py
HEADER = ["horizon", "late_from", "late_to", "rows", "median_dt", "sgp4_km", "told_nominal_km", "told_actual_km", "slope_nominal", "slope_actual"]
TRIM_KM = 500.0  # a manoeuvre sets a least-squares slope on its own, so those rows stay out of it


def slope(truth: np.ndarray, correction: np.ndarray) -> float:
    keep = np.abs(truth) < TRIM_KM
    return float((truth[keep] * correction[keep]).sum() / max((correction[keep] ** 2).sum(), 1e-9))


def main(folder: str, out: str, paths: list) -> None:
    device = "cuda"
    d = load_folder(folder, START, END, SHARDS)
    keep = np.isin(d["horizon"], HORIZONS)
    d = {k: v[keep] for k, v in d.items()}
    nets = [load_net(path, device) for path in paths]
    actual = []
    for net, args in nets:  # the recipe's fixed_dt would overwrite the actual dt
        actual.append(predict(net, tensors(d, type(args)(**{**vars(args), "fixed_dt": False})), device))
    actual = np.mean(actual, axis=0).astype(np.float64)
    nominal = np.mean([predict(net, tensors(d, args), device) for net, args in nets], axis=0).astype(np.float64)
    y, radius = d["y"].astype(np.float64), d["radius"].astype(np.float64)
    rows = []
    for h in HORIZONS:
        for low, high in LATE:
            late = d["dt"] - h
            m = (d["horizon"] == h) & (late >= low) & (late < high)
            if not m.any():
                continue
            rows.append([h, low, high, int(m.sum()), round(float(np.median(d["dt"][m])), 3),
                         round(float(np.median(miss_km(radius[m], y[m], np.zeros_like(y[m])))), 3),
                         round(float(np.median(miss_km(radius[m], y[m], nominal[m]))), 3),
                         round(float(np.median(miss_km(radius[m], y[m], actual[m]))), 3),
                         round(slope(y[m, 1], nominal[m, 1]), 3), round(slope(y[m, 1], actual[m, 1]), 3)])
            print(*rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

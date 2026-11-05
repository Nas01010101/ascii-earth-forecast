"""Scores the networks when the job holds only a few days of element sets.

usage: history_check.py SOURCE OUT_CSV NET.pt [NET.pt ...]
SOURCE is a folder of 2025 daily samples, for nets refit through 2024 (2025 is burned: a diagnostic only), or
FIT_PARQUET, for nets fit through 2023 scored on the 2024 validation rows. A lookback needs a set that old, so
a job with d days of snapshots sees every older lookback as missing. d=0 is the first day, with no updates counted.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from daily_check import load_net
from daily_samples import END, START
from data import load, load_folder, with_days_held
from sat_common import miss_km
from score import per_object
from train_net import END as VAL_END
from train_net import VAL_SHARDS, VAL_START, predict, tensors

DAYS_HELD = (0, 2, 4, 8, 16, 999)
HORIZONS = (1, 3, 7)
SHARDS = set(range(0, 64, 2))  # half the objects: enough for medians, and the box is shared
HEADER = ["days_held", "horizon", "rows", "objects", "sgp4_median_km", "model_median_km", "gain_mean", "gain_median", "worse_share"]


def rows_of(source: str) -> dict:
    if source.endswith(".parquet"):
        return load(source, VAL_START, VAL_END, shards=VAL_SHARDS & SHARDS)
    return load_folder(source, START, END, SHARDS)


def main(source: str, out: str, paths: list) -> None:
    device = "cuda"
    d = rows_of(source)
    keep = np.isin(d["horizon"], HORIZONS)
    d = {k: v[keep] for k, v in d.items()}
    y, radius = d["y"].astype(np.float64), d["radius"].astype(np.float64)
    nets = [load_net(path, device) for path in paths]
    rows = []
    for days in DAYS_HELD:
        seen = with_days_held(d, days)
        c = np.mean([predict(net, tensors(seen, args), device) for net, args in nets], axis=0).astype(np.float64)
        for h in HORIZONS:
            m = d["horizon"] == h
            raw, model = miss_km(radius[m], y[m], np.zeros_like(y[m])), miss_km(radius[m], y[m], c[m])
            ids, gain = per_object(d["norad"][m], radius[m], y[m], c[m])
            rows.append([days, h, int(m.sum()), len(ids), round(float(np.median(raw)), 2), round(float(np.median(model)), 2),
                         round(float(gain.mean()), 2), round(float(np.median(gain)), 2), round(float((gain < 0).mean()), 3)])
            print(*rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

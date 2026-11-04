"""Finds the rows that dominate the training loss: huge drift estimates multiplied by the network output.

usage: tail_check.py FIT_PARQUET OUT_CSV
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from data import day, load, metric_weights, own_drift
from train_net import VAL_START


def main(fit: str, out: str) -> None:
    rows = []
    for name, shards in (("0-15", set(range(16))), ("32-47", set(range(32, 48))), ("48-63", set(range(48, 64)))):
        d = load(fit, day(2020), VAL_START, shards=shards)
        d = {**d, "dt": d["horizon"].astype(d["dt"].dtype)}
        drift = np.abs(own_drift(d).astype(np.float64))
        w = metric_weights(d).astype(np.float64)
        weighted = w * drift  # the loss a gain of 1 on the drift term would add to this row
        top = np.argsort(weighted)[-5:][::-1]
        rows.append((name, len(w), round(float(np.quantile(drift, 0.5)), 1), round(float(np.quantile(drift, 0.999)), 1), float(drift.max()),
                     int((drift > 4e4).sum()), int((drift > 1e6).sum()), round(float(weighted.mean()), 2), round(float(weighted[drift <= 4e4].mean()), 2)))
        print(name, "top rows", [(int(d["norad"][i]), float(d["horizon"][i]), round(float(drift[i])), round(float(w[i]), 6), round(float(np.linalg.norm(d["y"][i])))) for i in top])
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shards", "rows", "drift_median", "drift_p999", "drift_max", "over_40000km", "over_1e6km", "loss_gain1", "loss_gain1_capped"])
        writer.writerows(rows)
    for row in rows:
        print(*row)


if __name__ == "__main__":
    main(*sys.argv[1:3])

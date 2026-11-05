"""Scores the networks when the job holds only a few days of element sets. Run only after the reveal.

usage: history_check.py DAILY_SAMPLES_DIR OUT_CSV NET.pt [NET.pt ...]
Each .pt was refit through 2024, so 2025 stays out of time. A lookback needs a set that old, so a job with
d days of snapshots sees every older lookback as missing. d=0 is the first day, with no updates counted.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from build_samples import FEATURES, LOOKBACKS
from daily_check import load_net
from daily_samples import END, START
from data import load_folder
from sat_common import miss_km
from score import per_object
from train_net import predict, tensors

DAYS_HELD = (0, 2, 4, 8, 16, 999)
HORIZONS = (1, 3, 7)
SHARDS = set(range(0, 64, 2))  # half the objects: enough for medians, and the box is shared
HEADER = ["days_held", "horizon", "rows", "objects", "sgp4_median_km", "model_median_km", "gain_mean", "gain_median", "worse_share"]


def with_days_held(d: dict, days: int) -> dict:
    """A copy of the samples as a job with `days` of snapshots would see them."""
    x = d["x"].copy()
    for back in LOOKBACKS:
        age = x[:, FEATURES.index(f"age_{int(back)}")]
        hidden = ~(age <= days)
        for name in ("age", "height", "along", "cross", "dno", "dbstar"):
            x[hidden, FEATURES.index(f"{name}_{int(back)}")] = np.nan
    if days == 0:
        x[:, FEATURES.index("updates_30d")] = 0.0
    return {**d, "x": x}


def main(folder: str, out: str, paths: list) -> None:
    device = "cuda"
    d = load_folder(folder, START, END, SHARDS)
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

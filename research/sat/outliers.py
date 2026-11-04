"""How far single rows dominate raw error at 15 days, per object, in the 2024 validation period.

usage: outliers.py FIT_PARQUET OUT_CSV
Reads the fit set only.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import pyarrow.parquet as pq

from data import day
from sat_common import miss_km
from score import MIN_SAMPLES

COLUMNS = ["norad", "epoch", "horizon", "radius", "t_height", "t_along", "t_cross", "no_rev", "perigee_km", "apogee_km"]
THRESHOLDS = (1e3, 1e4, 5e4, 1e5, 1e6)


def main(fit: str, out: str) -> None:
    parts = {c: [] for c in COLUMNS}
    for batch in pq.ParquetFile(fit).iter_batches(batch_size=500000, columns=COLUMNS):
        keep = (batch["horizon"].to_numpy() == 15.0) & (batch["epoch"].to_numpy() >= day(2024))
        for c in COLUMNS:
            parts[c].append(batch[c].to_numpy()[keep])
    d = {c: np.concatenate(parts[c]) for c in COLUMNS}
    truth = np.stack([d["t_height"], d["t_along"], d["t_cross"]], axis=1).astype(np.float64)
    raw = miss_km(d["radius"].astype(np.float64), truth, np.zeros_like(truth))
    arc = np.linalg.norm(truth, axis=1)

    rows = [("rows", len(raw)), ("objects", len(np.unique(d["norad"])))]
    for t in THRESHOLDS:
        rows.append((f"rows_chord_over_{int(t)}", int((raw > t).sum())))
        rows.append((f"rows_arc_over_{int(t)}", int((arc > t).sum())))
    rows += [("chord_mean_km", float(raw.mean())), ("arc_mean_km", float(arc.mean())), ("height_abs_max_km", float(np.abs(d["t_height"]).max())),
             ("radius_max_km", float(d["radius"].max()))]

    ids, inverse, counts = np.unique(d["norad"], return_inverse=True, return_counts=True)
    total = np.bincount(inverse, raw)
    top = np.zeros(len(ids))
    np.maximum.at(top, inverse, raw)
    keep = counts >= MIN_SAMPLES
    rows += [("objects_scored", int(keep.sum())), ("objects_top_row_over_half", int((top[keep] / np.maximum(total[keep], 1e-9) > 0.5).sum())),
             ("objects_mean_under_1km", int((total[keep] / counts[keep] < 1.0).sum())), ("objects_mean_over_100km", int((total[keep] / counts[keep] > 100.0).sum()))]
    worst = np.argsort(raw)[-8:][::-1]
    for i in worst:
        print("worst", int(d["norad"][i]), round(float(d["epoch"][i]), 2), "miss", round(float(raw[i]), 1), "arc", round(float(arc[i]), 1), "height", round(float(d["t_height"][i]), 1),
              "radius", round(float(d["radius"][i]), 1), "rev", round(float(d["no_rev"][i]), 2), "perigee", round(float(d["perigee_km"][i]), 1))
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows([("name", "value"), *rows])
    for row in rows:
        print(*row)


if __name__ == "__main__":
    main(*sys.argv[1:3])

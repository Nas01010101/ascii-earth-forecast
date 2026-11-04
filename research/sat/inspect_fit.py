"""Row counts and target sizes of the fit set, by year, horizon and orbit regime.

usage: inspect_fit.py FIT_PARQUET OUT_CSV
Reads the fit set only: holdout labels stay sealed.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import pyarrow.parquet as pq

from build_samples import FEATURES, HISTORY_FEATURES
from data import day

YEARS = range(2017, 2025)
REGIMES = (("LEO", 11.25, 99.0), ("MEO", 1.5, 11.25), ("GEO", 0.0, 1.5))


def main(fit: str, out: str) -> None:
    columns = ["norad", "epoch", "horizon", "dt", "t_along", "t_height", "t_cross", "no_rev", "age_1", "age_15"]
    parts = {c: [] for c in columns}
    file = pq.ParquetFile(fit)
    rng = np.random.default_rng(0)
    for batch in file.iter_batches(batch_size=500000, columns=columns):
        keep = rng.random(batch.num_rows) < 0.2  # a fifth of the rows settles these sizes and keeps memory low
        for c in columns:
            parts[c].append(batch[c].to_numpy()[keep])
    d = {c: np.concatenate(parts[c]) for c in columns}
    print(f"rows {file.metadata.num_rows}, sampled {len(d['dt'])}, objects in sample {len(np.unique(d['norad']))}")
    for name in FEATURES:
        if name in columns:
            print(name, "missing", round(float(np.isnan(d[name]).mean()), 3))
    print("history columns", len(HISTORY_FEATURES))

    miss = np.sqrt(d["t_along"] ** 2 + d["t_height"] ** 2 + d["t_cross"] ** 2)
    rows = []
    for h in np.unique(d["horizon"]):
        m = d["horizon"] == h
        rows.append(("horizon", float(h), int(m.sum()), *np.round(np.percentile(miss[m], [50, 90, 99]), 2), round(float(miss[m].mean()), 2)))
    for year in YEARS:
        m = (d["epoch"] >= day(year)) & (d["epoch"] < day(year + 1))
        if m.any():
            rows.append(("year", year, int(m.sum()), *np.round(np.percentile(miss[m], [50, 90, 99]), 2), round(float(miss[m].mean()), 2)))
    for name, low, high in REGIMES:
        m = (d["no_rev"] >= low) & (d["no_rev"] < high) & (d["horizon"] == 15.0)
        if m.any():
            rows.append(("regime_15d", name, int(m.sum()), *np.round(np.percentile(miss[m], [50, 90, 99]), 2), round(float(miss[m].mean()), 2)))
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "value", "rows", "p50_km", "p90_km", "p99_km", "mean_km"])
        writer.writerows(rows)
    for row in rows:
        print(*row)


if __name__ == "__main__":
    main(*sys.argv[1:3])

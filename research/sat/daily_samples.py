"""Rebuilds 2025 samples from what a once-a-day download would have held, which is all production ever sees.

usage: daily_samples.py SHARD_DIR SPACE_WEATHER_CSV OUT_PARQUET
Features and truth both come from the daily view, so updates_30d counts at most one set per day and a
lookback can only land on a set some snapshot held.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_samples import FEATURES, MAX_NORAD, TARGETS, object_samples, read_space_weather
from data import day
from sat_common import COLUMNS, dedupe_sorted

SNAPSHOT_HOUR = 1 + 37 / 60  # .github/workflows/daily.yml runs at 01:37 UTC
START, END = day(2025), day(2026)
HISTORY_DAYS = 31  # updates_30d looks back 30 days from t0


def daily_view(epoch: np.ndarray) -> np.ndarray:
    """Indices of the sets a daily snapshot would have held: the newest one at or before each download."""
    taken = np.ceil(epoch - SNAPSHOT_HOUR / 24)
    return np.flatnonzero(np.r_[taken[1:] != taken[:-1], True])


def main(shard: str, sw_path: str, out_path: str) -> None:
    weather = read_space_weather(sw_path)
    table = pa.concat_tables([pq.read_table(p, columns=list(COLUMNS)) for p in sorted(Path(shard).glob("*.parquet"))])
    norad = table["norad"].to_numpy()
    arrays = {c: table[c].to_numpy() for c in COLUMNS[1:]}
    names = ["norad", "epoch", "horizon", "dt", "radius"] + TARGETS + FEATURES
    schema = pa.schema([("norad", pa.int32()), ("epoch", pa.float64())] + [(n, pa.float32()) for n in names[2:]])
    order = np.argsort(norad, kind="stable")
    ids, starts = np.unique(norad[order], return_index=True)
    bounds = np.append(starts, len(order))
    rows = []
    for k, obj in enumerate(ids):
        if obj >= MAX_NORAD:
            continue
        idx = order[bounds[k]:bounds[k + 1]]
        idx = idx[dedupe_sorted(arrays["epoch"][idx])]
        idx = idx[arrays["epoch"][idx] >= START - HISTORY_DAYS]
        if not len(idx):
            continue
        idx = idx[daily_view(arrays["epoch"][idx])]
        rows += [r for r in object_samples(int(obj), {c: arrays[c][idx] for c in COLUMNS[1:]}, weather) if r[1] >= START]
    cols = list(zip(*rows)) if rows else [[] for _ in names]
    pq.write_table(pa.table({n: pa.array(c, type=schema.field(n).type) for n, c in zip(names, cols)}), out_path, compression="zstd")
    print(f"{shard}: {len(rows)} samples", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])

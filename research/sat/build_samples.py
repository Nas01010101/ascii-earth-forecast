"""Turns one catalogue shard into forecast samples: what is known at t0, and the error at each horizon.

usage: build_samples.py SHARD_DIR SPACE_WEATHER_CSV OUT_PARQUET

A sample is an element set at t0, a horizon, and the SGP4 error against the first element set at or
after t0 + horizon. Every feature uses element sets with epochs at or before t0 and space weather
observed on the day before t0, so a sample can be rebuilt on the day it is needed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sat_common import COLUMNS, curvilinear_error, dedupe_sorted, make_satrec

HORIZONS = (1.0, 3.0, 7.0, 15.0, 30.0)
LOOKBACKS = (1.0, 3.0, 7.0, 15.0)
STEP_DAYS = 5.0  # one t0 per object per step, so a few fast-updating objects don't dominate
TOLERANCE_DAYS = 1.0  # how far past the horizon the truth element set may be
LOOKBACK_TOLERANCE = 1.0
MAX_NORAD = 80000  # higher numbers are analyst objects with no public history
EARTH_RADIUS_KM = 6378.135
MU = 398600.8  # WGS72, km^3/s^2, the constant SGP4 uses here

ELEMENT_FEATURES = ["no_rev", "ecco", "inclo", "sin_argp", "cos_argp", "bstar", "ndot", "nddot", "perigee_km", "apogee_km", "updates_30d"]
HISTORY_FEATURES = [f"{name}_{int(a)}" for a in LOOKBACKS for name in ("age", "height", "along", "cross", "dno", "dbstar")]
WEATHER_FEATURES = ["f107", "f107_81", "ap"]
FEATURES = ELEMENT_FEATURES + HISTORY_FEATURES + WEATHER_FEATURES
TARGETS = ["t_height", "t_along", "t_cross"]


def read_space_weather(path: str) -> dict:
    """Observed rows only: the file also holds predicted values out to 2041, unknown at any real t0."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["F10.7_DATA_TYPE"] != "OBS":
                continue
            y, m, d = (int(x) for x in row["DATE"].split("-"))
            day = int(np.datetime64(f"{y:04d}-{m:02d}-{d:02d}").astype("datetime64[D]").astype(int))
            out[day] = (float(row["F10.7_OBS"]), float(row["F10.7_OBS_LAST81"]), float(row["AP_AVG"]))
    return out


# sgp4init epochs count from 1949-12-31 00:00 UT, which is day -7306 of the unix day count.
EPOCH_TO_UNIX_DAY = -7306


def element_features(row: dict, i: int, updates: int) -> list:
    n = row["no_rev"][i] * 2.0 * np.pi / 86400.0
    a = (MU / max(n * n, 1e-16)) ** (1.0 / 3.0)
    e = row["ecco"][i]
    return [row["no_rev"][i], e, row["inclo"][i], np.sin(np.radians(row["argpo"][i])), np.cos(np.radians(row["argpo"][i])),
            row["bstar"][i], row["ndot"][i], row["nddot"][i], a * (1 - e) - EARTH_RADIUS_KM, a * (1 + e) - EARTH_RADIUS_KM, updates]


def error_of(sat_old, sat_new, dt_days: float):
    """(height, along, cross, radius) of the true position relative to the propagated one, or None."""
    e0, r0, v0 = sat_old.sgp4_tsince(dt_days * 1440.0)
    e1, r1, _ = sat_new.sgp4_tsince(0.0)
    if e0 or e1:
        return None
    return curvilinear_error(r0, v0, r1)


def fixed_targets(norad: int, t: float) -> list:
    """(label, gap after t0 the truth set must reach, how much later it may arrive), one per row wanted at t0."""
    return [(h, h, TOLERANCE_DAYS) for h in HORIZONS]


def object_samples(norad: int, row: dict, weather: dict, targets=fixed_targets) -> list:
    epoch = row["epoch"]
    sats, out = {}, []

    def sat(i):
        if i not in sats:
            sats[i] = make_satrec(norad, *(row[c][i] for c in COLUMNS[1:]))
        return sats[i]

    last = -np.inf
    for i, t in enumerate(epoch):
        if t - last < STEP_DAYS:
            continue
        last = t
        sw = weather.get(int(np.floor(t)) + EPOCH_TO_UNIX_DAY - 1)
        if sw is None:
            continue
        updates = int(i - np.searchsorted(epoch, t - 30.0))
        base = element_features(row, i, updates)
        history = []
        for back in LOOKBACKS:
            # The newest element set at least `back` days older than t0.
            j = int(np.searchsorted(epoch, t - back, side="right")) - 1
            found = None
            if j >= 0 and t - back - epoch[j] <= LOOKBACK_TOLERANCE:
                found = error_of(sat(j), sat(i), float(t - epoch[j]))
            if found is None:
                history += [np.nan] * 6
            else:
                history += [float(t - epoch[j]), found[0], found[1], found[2], row["no_rev"][i] - row["no_rev"][j], row["bstar"][i] - row["bstar"][j]]
        for label, gap, tolerance in targets(norad, float(t)):
            j = int(np.searchsorted(epoch, t + gap))
            if j >= len(epoch) or epoch[j] - (t + gap) > tolerance:
                continue
            dt = float(epoch[j] - t)
            found = error_of(sat(i), sat(j), dt)
            if found is None:
                continue
            out.append((norad, float(t), label, dt, found[3], *found[:3], *base, *history, *sw))
    return out


def main(shard: str, sw_path: str, out_path: str, targets=fixed_targets) -> None:
    weather = read_space_weather(sw_path)
    tables =[pq.read_table(p, columns=list(COLUMNS)) for p in sorted(Path(shard).glob("*.parquet"))]
    table = pa.concat_tables(tables)
    norad = table["norad"].to_numpy()
    arrays = {c: table[c].to_numpy() for c in COLUMNS[1:]}
    del tables, table

    names = ["norad", "epoch", "horizon", "dt", "radius"] + TARGETS + FEATURES
    schema = pa.schema([("norad", pa.int32()), ("epoch", pa.float64())] + [(n, pa.float32()) for n in names[2:]])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    rows, objects, total = [], 0, 0

    def flush():
        nonlocal rows, total
        if rows:
            cols = list(zip(*rows))
            writer.write_table(pa.table({n: pa.array(c, type=schema.field(n).type) for n, c in zip(names, cols)}))
            total += len(rows)
            rows = []

    order = np.argsort(norad, kind="stable")
    ids, starts = np.unique(norad[order], return_index=True)
    bounds = np.append(starts, len(order))
    for k, obj in enumerate(ids):
        if obj >= MAX_NORAD:
            continue
        idx = order[bounds[k]:bounds[k + 1]]
        keep = idx[dedupe_sorted(arrays["epoch"][idx])]
        row = {c: arrays[c][keep] for c in COLUMNS[1:]}
        rows += object_samples(int(obj), row, weather, targets)
        objects += 1
        if len(rows) >= 200000:
            flush()
    flush()
    writer.close()
    print(f"{shard}: {objects} objects, {total} samples")


if __name__ == "__main__":
    main(*sys.argv[1:4])

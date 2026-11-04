"""Raw SGP4 error on the objects of Liu et al. 2021 (SSC21-IX-02), compared with the numbers that paper publishes.

usage: positive_control.py TLE_TEXT OUT_CSV
Truth is SGP4 at the epoch of the first later element set within TOLERANCE_DAYS of the horizon.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from sgp4.api import jday

from sat_common import dedupe_sorted, error_between, make_satrec, parse_pair

OBJECTS = (27845, 20774, 23975)
HORIZONS = (15.0, 30.0)
# Raw SGP4 3D error in km read from doi.org/10.26077/aspb-yk57, by object and horizon.
PUBLISHED = {(27845, 15.0): 2, (20774, 15.0): 12, (23975, 15.0): 50, (27845, 30.0): 5, (20774, 30.0): 70, (23975, 30.0): 200}
SGP4_EPOCH_JD = 2433281.5
START = jday(2017, 6, 1, 0, 0, 0)[0] - SGP4_EPOCH_JD
END = jday(2020, 6, 30, 0, 0, 0)[0] - SGP4_EPOCH_JD
TOLERANCE_DAYS = 1.0


def read_rows(path: str) -> dict:
    rows, first = defaultdict(list), None
    with open(path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line.startswith("1 "):
                first = line
            elif line.startswith("2 ") and first is not None:
                try:
                    row = parse_pair(first, line)
                except (ValueError, IndexError):
                    first = None
                    continue
                first = None
                if row[0] in OBJECTS and START <= row[1] <= END:
                    rows[row[0]].append(row)
    return rows


def errors_at(rows: list, horizon: float) -> np.ndarray:
    epoch = np.array([r[1] for r in rows])
    order = dedupe_sorted(epoch)
    epoch = epoch[order]
    sats = [make_satrec(*rows[i]) for i in order]
    out = []
    for i, t in enumerate(epoch):
        j = int(np.searchsorted(epoch, t + horizon))
        if j >= len(epoch) or epoch[j] - (t + horizon) > TOLERANCE_DAYS:
            continue
        e = error_between(sats[i], sats[j], float(epoch[j] - t))
        if e is not None:
            out.append(e)
    return np.array(out).reshape(-1, 3)


def main(path: str, out: str) -> None:
    rows = read_rows(path)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["norad", "horizon_days", "tles", "samples", "mean_km", "median_km", "p90_km", "radial_km", "cross_km", "along_km", "published_km"])
        for norad in OBJECTS:
            for h in HORIZONS:
                e = errors_at(rows[norad], h)
                norm = np.linalg.norm(e, axis=1)
                w.writerow([norad, h, len(rows[norad]), len(e), f"{norm.mean():.2f}", f"{np.median(norm):.2f}", f"{np.percentile(norm, 90):.2f}",
                            *[f"{np.abs(e[:, c]).mean():.2f}" for c in range(3)], PUBLISHED[norad, h]])
    print(open(out).read())


if __name__ == "__main__":
    main(*sys.argv[1:3])

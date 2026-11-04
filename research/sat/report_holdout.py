"""Holdout corrections by horizon, orbit regime and whether training saw the object. Run only after the reveal.

usage: report_holdout.py HOLDOUT_X HOLDOUT_Y OUT_CSV NAME=HOLD_NPY [NAME=HOLD_NPY ...]
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from build_samples import HORIZONS, TARGETS
from data import col, day, load
from diagnose import REGIMES
from report import HEADER, row
from train_net import END, train_shards


def aligned_labels(d: dict, labels) -> tuple:
    """The scorer reads labels by position, so a holdout row order that differs here would be silent without this check."""
    assert np.array_equal(d["norad"], labels["norad"]) and np.array_equal(d["horizon"], labels["horizon"]), "labels and features are not aligned"
    return np.stack([labels[t] for t in TARGETS], axis=1).astype(np.float64), labels["radius"].astype(np.float64)


def main(holdout: str, labels: str, out: str, specs: list) -> None:
    d = load(holdout, END, day(2026), labelled=False)
    y, radius = aligned_labels(d, np.load(labels))
    no_rev, seen = col(d, "no_rev"), train_shards(32)
    rows = []
    for spec in specs:
        name, path = spec.split("=", 1)
        c = np.load(path).astype(np.float64)
        assert len(c) == len(y), f"{path} has {len(c)} rows, holdout has {len(y)}"
        for h in HORIZONS:
            for regime, low, high in REGIMES:
                m = (d["horizon"] == h) & (no_rev >= low) & (no_rev < high)
                if m.any():
                    rows.append(row(name, h, regime, d, y, radius, c, m, seen))
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(*HEADER)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])

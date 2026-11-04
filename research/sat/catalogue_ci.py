"""Per-object holdout gain with calendar-block intervals and with the largest rows removed. Run only after the reveal.

usage: catalogue_ci.py FIT_PARQUET HOLDOUT_X HOLDOUT_Y OUT_CSV NAME=HOLD_NPY [NAME=HOLD_NPY ...]
Objects share calendar months, so resampling months keeps errors from one solar storm together.
new_mean covers objects with no row in the fit table: launches the training never saw, siblings included.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.stats import trim_mean

from build_samples import HORIZONS
from data import day, load
from head_to_head import BOOTSTRAP_DRAWS, month_blocks
from report_holdout import aligned_labels
from sat_common import miss_km
from score import MIN_SAMPLES, per_object
from train_net import END

HEADER = ["name", "horizon", "rows", "objects", "mean", "lo", "hi", "median", "trim5", "drop_top", "new_objects", "new_mean"]


def block_interval(norad, months, raw, model, draws=BOOTSTRAP_DRAWS, seed=0) -> tuple:
    """95% interval of the mean per-object gain, resampling whole blocks across every object."""
    ids, obj = np.unique(norad, return_inverse=True)
    labels, block = np.unique(months, return_inverse=True)

    def by_block(v):
        return coo_matrix((v, (obj, block)), shape=(len(ids), len(labels))).tocsr()

    raw_sums, model_sums, row_counts = by_block(raw), by_block(model), by_block(np.ones_like(raw))
    rng, gains = np.random.default_rng(seed), np.zeros(draws)
    for i in range(draws):
        weight = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels)).astype(np.float64)
        r, m, n = raw_sums @ weight, model_sums @ weight, row_counts @ weight
        keep = (n >= MIN_SAMPLES) & (r > 0)
        gains[i] = (100.0 * (1.0 - m[keep] / r[keep])).mean()
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def drop_top_row(norad, raw) -> np.ndarray:
    """Mask without each object's largest raw-error row, the row a manoeuvre or decay would dominate."""
    order = np.lexsort((raw, norad))
    last = np.r_[norad[order][1:] != norad[order][:-1], True]
    keep = np.ones(len(raw), dtype=bool)
    keep[order[last]] = False
    return keep


def summary(norad, months, radius, y, c, fit_ids) -> list:
    raw, model = miss_km(radius, y, np.zeros_like(y)), miss_km(radius, y, c)
    ids, gain = per_object(norad, radius, y, c)
    lo, hi = block_interval(norad, months, raw, model)
    keep = drop_top_row(norad, raw)
    _, dropped = per_object(norad[keep], radius[keep], y[keep], c[keep])
    new = ~np.isin(ids, fit_ids)
    values = [gain.mean(), lo, hi, np.median(gain), trim_mean(gain, 0.05), dropped.mean()]
    return [len(norad), len(ids), *[round(float(v), 2) for v in values], int(new.sum()), round(float(gain[new].mean()), 2) if new.any() else ""]


def main(fit: str, holdout: str, labels: str, out: str, specs: list) -> None:
    fit_ids = np.unique(pq.read_table(fit, columns=["norad"])["norad"].to_numpy())
    d = load(holdout, END, day(2026), labelled=False)
    y, radius = aligned_labels(d, np.load(labels))
    months, rows = month_blocks(d["epoch"]), []
    for spec in specs:
        name, path = spec.split("=", 1)
        c = np.load(path).astype(np.float64)
        assert len(c) == len(y), f"{path} has {len(c)} rows, holdout has {len(y)}"
        for h in HORIZONS:
            m = d["horizon"] == h
            rows.append([name, h, *summary(d["norad"][m], months[m], radius[m], y[m], c[m], fit_ids)])
            print(*rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])

"""Percent less position error than raw SGP4 at 15 days, averaged over objects, for 5 disjoint object groups.

usage: score.py CORRECTIONS.npy LABELS.npz
CORRECTIONS holds (height, along, cross) in km per labelled row. Prints 5 comma-separated scores.
An object counts when it has at least MIN_SAMPLES samples at the horizon.
"""
from __future__ import annotations

import sys

import numpy as np

from sat_common import miss_km

HORIZON = 15.0
MIN_SAMPLES = 5
GROUPS = 5


def per_object(norad, radius, truth, correction) -> tuple[np.ndarray, np.ndarray]:
    """Returns the object ids and each one's percent reduction of mean 3D error against no correction."""
    raw = miss_km(radius, truth, np.zeros_like(truth))
    model = miss_km(radius, truth, correction)
    ids, inverse, counts = np.unique(norad, return_inverse=True, return_counts=True)
    raw_sum, model_sum = np.bincount(inverse, raw), np.bincount(inverse, model)
    keep = counts >= MIN_SAMPLES
    # A raw error of zero has nothing to reduce and says nothing about the correction.
    keep &= raw_sum > 0
    return ids[keep], 100.0 * (1.0 - model_sum[keep] / raw_sum[keep])


def group_scores(norad, horizon, radius, truth, correction, at=HORIZON) -> np.ndarray:
    mask = horizon == at
    ids, gain = per_object(norad[mask], radius[mask], truth[mask], correction[mask])
    # Multiplicative hashing keeps neighbouring catalogue numbers, often one launch, in different groups.
    group = (ids.astype(np.int64) * 2654435761 % 2**32) % GROUPS
    return np.array([gain[group == g].mean() for g in range(GROUPS)])


def main(cand: str, labels: str) -> None:
    d = np.load(labels)
    correction = np.load(cand)
    if correction.shape != (len(d["norad"]), 3):
        raise SystemExit(f"expected shape {(len(d['norad']), 3)}, got {correction.shape}")
    if not np.isfinite(correction).all():
        raise SystemExit("corrections hold non-finite values")
    truth = np.stack([d["t_height"], d["t_along"], d["t_cross"]], axis=1)
    print(",".join(f"{s:.4f}" for s in group_scores(d["norad"], d["horizon"], d["radius"], truth, correction)))


if __name__ == "__main__":
    main(*sys.argv[1:3])

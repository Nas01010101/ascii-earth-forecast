"""Percent below the ridge on 4-day parts of a labels file, one number per part.

usage: parts_score.py PREDICTIONS.npz LABELS.npz
Both files hold `days` and `cloud`, and the labels also hold the ridge's forecast of each day. Predictions are clipped
to 0..100 as production does. The parts are disjoint, so their spread is a fair estimate of the error of the mean.
"""
from __future__ import annotations

import sys

import numpy as np

from forecast import weather
from score import CLOUD_RANGE

DAYS_PER_PART = 4


def part_scores(cloud: np.ndarray, truth: np.ndarray, ridge: np.ndarray) -> list:
    rows = []
    for start in range(0, len(truth) - DAYS_PER_PART + 1, DAYS_PER_PART):
        days = range(start, start + DAYS_PER_PART)
        model = np.mean([weather.mse(np.clip(cloud[d], *CLOUD_RANGE), truth[d]) for d in days])
        base = np.mean([weather.mse(np.clip(ridge[d], *CLOUD_RANGE), truth[d]) for d in days])
        rows.append(100 * (1 - model / base))
    return rows


def main(path: str, labels: str) -> None:
    got, want = np.load(path), np.load(labels)
    if list(got["days"]) != list(want["days"]) or not np.isfinite(got["cloud"]).all():
        raise ValueError("predictions do not cover the labelled days")
    print(",".join(f"{s:.4f}" for s in part_scores(got["cloud"], want["cloud"], want["ridge"])))


if __name__ == "__main__":
    main(*sys.argv[1:])

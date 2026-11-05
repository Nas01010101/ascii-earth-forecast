"""Percent below the ridge on 4-day parts of a labels file, one number per part, averaged over temperature and pressure.

usage: fields_score.py PREDICTIONS.npz LABELS.npz [--each]
--each prints each field's parts on a line of its own instead. The parts are disjoint, so their spread is a fair
estimate of the error of the mean.
"""
from __future__ import annotations

import sys

import numpy as np

from field_labels import TARGETS
from forecast import weather
from parts_score import DAYS_PER_PART


def field_parts(forecast: np.ndarray, truth: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    rows = []
    for start in range(0, len(truth) - DAYS_PER_PART + 1, DAYS_PER_PART):
        days = range(start, start + DAYS_PER_PART)
        model = np.mean([weather.mse(forecast[d], truth[d]) for d in days])
        base = np.mean([weather.mse(ridge[d], truth[d]) for d in days])
        rows.append(100 * (1 - model / base))
    return np.array(rows)


def scores(got, want) -> dict:
    if list(got["days"]) != list(want["days"]):
        raise ValueError("predictions do not cover the labelled days")
    for name in TARGETS:
        if not np.isfinite(got[name]).all():
            raise ValueError(f"{name}: the forecast is not finite")
    return {name: field_parts(got[name], want[name], want[f"ridge_{name}"]) for name in TARGETS}


def main(path: str, labels: str, *flags: str) -> None:
    found = scores(np.load(path), np.load(labels))
    if "--each" in flags:
        for name, parts in found.items():
            print(f"{name} mean {parts.mean():.2f}:", ",".join(f"{s:.4f}" for s in parts))
    else:
        print(",".join(f"{s:.4f}" for s in np.mean(list(found.values()), axis=0)))


if __name__ == "__main__":
    main(*sys.argv[1:])

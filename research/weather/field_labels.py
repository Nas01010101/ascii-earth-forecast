"""Writes the temperature and pressure labels of the Open-Meteo days, on the dates of the cloud round.

usage: field_labels.py DEV.npz LATE.npz OUT_DIR
Writes select_y.npz and test_y.npz, each with `days`, the truth of `temp` and `pressure` and the walk-forward ridge
forecast of both. Temperature truth is the 24-hour mean. Pressure truth is the mean of the 4 samples, because the source
gives no 24-hour mean and pressure hardly moves within a day.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

from common import daily
from openmeteo_split import FIRST_SELECT, FIRST_TEST, days_from, walk_forward_ridge

TARGETS = ("temp", "pressure")


def _days(held, name: str) -> np.ndarray:
    return held[f"{name}_day"] if f"{name}_day" in held.files else daily(held[name])


def joined_days(dev, late, name: str) -> np.ndarray:
    """Daily truth from the first day of dev to the last of late. The files overlap, and must agree where they do."""
    start, join = date.fromisoformat(str(dev["first_date"])), date.fromisoformat(str(late["first_date"]))
    early, later = _days(dev, name), _days(late, name)
    overlap = len(early) - (join - start).days
    if overlap <= 0 or not np.allclose(early[-overlap:], later[:overlap], atol=1e-3):
        raise ValueError(f"{name}: the field files do not agree where they overlap")
    return np.concatenate([early, later[overlap:]])


def labels(dev, late) -> dict:
    start = date.fromisoformat(str(dev["first_date"]))
    series = {name: joined_days(dev, late, name) for name in TARGETS}
    total = len(series["temp"])
    select_from, test_from = (FIRST_SELECT - start).days, (FIRST_TEST - start).days
    out = {}
    for label, first, begin, end in (("select_y", FIRST_SELECT, select_from, test_from), ("test_y", FIRST_TEST, test_from, total)):
        arrays = {"days": np.array(days_from(first, end - begin))}
        for name in TARGETS:
            arrays[name] = series[name][begin:end]
            arrays[f"ridge_{name}"] = walk_forward_ridge(series[name][:end], end - begin)
        out[label] = arrays
    return out


def main(dev: str, late: str, out: str) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for name, arrays in labels(np.load(dev), np.load(late)).items():
        np.savez(folder / f"{name}.npz", **arrays)


if __name__ == "__main__":
    main(*sys.argv[1:])

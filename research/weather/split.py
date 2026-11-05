"""Builds the validation and holdout files, then removes the holdout years' raw files.

usage: split.py ERA5_DIR VAL.npz HOLDOUT.npz
Training reads 1979-2018 raw. Validation targets are 2019-2020 and holdout targets are 2021-2022. Inputs of a later
day are the labels of an earlier one, so the holdout seals the fields themselves, with the days before its first
target that the ridge reads.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from candidate import RIDGE_DAYS
from common import FIELDS, STEPS_PER_DAY, load_year

VAL = (2019, 2020)
HOLDOUT = (2021, 2022)


def window(folder: str, name: str, first: date, last_year: int) -> np.ndarray:
    field = np.concatenate([load_year(folder, name, y) for y in range(first.year, last_year + 1)])
    return field[(first - date(first.year, 1, 1)).days * STEPS_PER_DAY:]


def write(path: str, folder: str, years: tuple) -> None:
    target = date(years[0], 1, 1)
    first = target - timedelta(days=RIDGE_DAYS)
    np.savez(path, first_date=first.isoformat(), first_target=target.isoformat(),
             **{name: window(folder, name, first, years[1]) for name in FIELDS})


def main(folder: str, val: str, holdout: str) -> None:
    write(val, folder, VAL)
    write(holdout, folder, HOLDOUT)
    for year in HOLDOUT:
        for store, _, _ in FIELDS.values():
            (Path(folder) / f"{store}_{year}.npy").unlink()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

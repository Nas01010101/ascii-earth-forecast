"""Fits day-of-year means and anomaly scales on the training years only.

usage: climatology.py ERA5_DIR OUT.npz [LAST_YEAR]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

from common import FIELDS, daily, load_year

FIRST_YEAR = 1979
LAST_TRAIN = 2018
SMOOTH_DAYS = 15


def doy_of(when: date) -> int:
    return when.timetuple().tm_yday - 1


def smooth(by_day: np.ndarray) -> np.ndarray:
    """Circular running mean over the 366 days, since 40 years of one calendar day is still noisy."""
    kernel = np.ones(SMOOTH_DAYS) / SMOOTH_DAYS
    half = SMOOTH_DAYS // 2
    wrapped = np.concatenate([by_day[-half:], by_day, by_day[:half]])
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, "valid"), 0, wrapped).astype(np.float32)


def fit(folder: str, last_year: int = LAST_TRAIN) -> dict:
    out = {}
    for name in FIELDS:
        total, count = np.zeros((366, 24, 48)), np.zeros(366)
        years = range(FIRST_YEAR, last_year + 1)
        for year in years:
            days = daily(load_year(folder, name, year))
            index = np.array([doy_of(date.fromordinal(date(year, 1, 1).toordinal() + d)) for d in range(len(days))])
            np.add.at(total, index, days)
            np.add.at(count, index, 1)
        clim = smooth(total / count[:, None, None])
        squares, n = 0.0, 0
        for year in years:
            field = load_year(folder, name, year)
            step_doy = np.repeat([doy_of(date.fromordinal(date(year, 1, 1).toordinal() + d)) for d in range(len(field) // 4)], 4)
            squares += float(((field - clim[step_doy]) ** 2).sum())
            n += field.size
        out[f"clim_{name}"], out[f"std_{name}"] = clim, np.float32((squares / n) ** 0.5)
    for name, scale in (("land_sea_mask", 1.0), ("geopotential_at_surface", 1 / 9.80665 / 1000)):
        out[name] = np.load(Path(folder) / f"{name}.npy") * np.float32(scale)
    return out


if __name__ == "__main__":
    np.savez(sys.argv[2], **fit(sys.argv[1], *map(int, sys.argv[3:])))

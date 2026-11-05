"""Splits Open-Meteo's day-earlier and two-day-earlier forecasts into forecast files for stack.py.

usage: lagged_features.py LAGGED.npz ONE_DAY.npz TWO_DAYS.npz
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np

from predict_net import days_from


def split(lagged: dict) -> tuple:
    """Each forecast as {days, cloud}: LAGGED holds no truth, so days past the select window can stay in the files."""
    days = np.array(days_from(date.fromisoformat(str(lagged["first_date"])), len(lagged["cloud_day"])))
    return {"days": days, "cloud": lagged["cloud_day"]}, {"days": days, "cloud": lagged["cloud2_day"]}


def main(path: str, one_day: str, two_days: str) -> None:
    first, second = split(np.load(path))
    np.savez(one_day, **first)
    np.savez(two_days, **second)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Random hourly fields on the forecast grid, shared by the daily-job tests."""
from datetime import date, timedelta

import numpy as np

from forecast import fetch

TODAY = date(2026, 11, 3)
SHAPE = (len(fetch.LATS), len(fetch.LONS))


def hourly(days_back, seed=0):
    rng = np.random.default_rng(seed)
    start = np.datetime64(TODAY - timedelta(days=days_back), "h")
    hours = start + np.arange((days_back + fetch.FORECAST_DAYS) * 24)
    fields = {"cloud": rng.uniform(0, 100, (hours.size, *SHAPE)).astype(np.float32),
              "temp": rng.uniform(-20, 30, (hours.size, *SHAPE)).astype(np.float32),
              "precip": rng.uniform(0, 3, (hours.size, *SHAPE)).astype(np.float32),
              "pressure": rng.uniform(980, 1030, (hours.size, *SHAPE)).astype(np.float32),
              "speed": rng.uniform(0, 20, (hours.size, *SHAPE)).astype(np.float32),
              "direction": rng.uniform(0, 360, (hours.size, *SHAPE)).astype(np.float32),
              "water": rng.uniform(0, 60, (hours.size, *SHAPE)).astype(np.float32)}
    return hours, fields


def field(value):
    return np.full(SHAPE, value, dtype=np.float32)


def past_days(n):
    """The n days before TODAY, oldest first."""
    return [(TODAY - timedelta(days=d)).isoformat() for d in range(n, 0, -1)]

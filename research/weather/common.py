"""ERA5 fields in the daily job's units, and the daily mean the job stores."""
from __future__ import annotations

from pathlib import Path

import numpy as np

STEPS_PER_DAY = 4
# name in the code: (store name, scale, offset) to reach percent, Celsius, mm per 6 h, hPa, m/s and kg/m2.
FIELDS = {
    "cloud": ("total_cloud_cover", 100.0, 0.0),
    "temp": ("2m_temperature", 1.0, -273.15),
    "precip": ("total_precipitation_6hr", 1000.0, 0.0),
    "pressure": ("mean_sea_level_pressure", 0.01, 0.0),
    "u": ("10m_u_component_of_wind", 1.0, 0.0),
    "v": ("10m_v_component_of_wind", 1.0, 0.0),
    "water": ("total_column_water_vapour", 1.0, 0.0),
}


def load_year(folder: str, name: str, year: int) -> np.ndarray:
    store, scale, offset = FIELDS[name]
    return np.load(Path(folder) / f"{store}_{year}.npy").astype(np.float32) * np.float32(scale) + np.float32(offset)


def daily(field: np.ndarray) -> np.ndarray:
    """(steps, ...) to (days, ...). A step count that is not a whole number of days is a bug upstream."""
    days, rest = divmod(len(field), STEPS_PER_DAY)
    assert rest == 0, len(field)
    return field.reshape(days, STEPS_PER_DAY, *field.shape[1:]).mean(axis=1)

"""Scores Open-Meteo's forecast made a day earlier on the daily job's stored days. Comparison only.

usage: nwp_loss.py HISTORY.npz NWP.npz [DAYS]
Prints mean cloud and temperature loss over the last DAYS stored days. The forecast file starts on the first of them.
"""
from __future__ import annotations

import sys

import numpy as np

from forecast import weather
from score import CLOUD_RANGE


def losses(cloud: np.ndarray, temp: np.ndarray, nwp_cloud: np.ndarray, nwp_temp: np.ndarray) -> dict:
    return {
        "cloud": float(np.mean([weather.mse(np.clip(a, *CLOUD_RANGE), b) for a, b in zip(nwp_cloud, cloud, strict=True)])),
        "temp": float(np.mean([weather.mse(a, b) for a, b in zip(nwp_temp, temp, strict=True)])),
    }


def main(history: str, nwp: str, count: str = "28") -> None:
    stored, made = np.load(history), np.load(nwp)
    days = int(count)
    if str(stored["days"][-days]) != str(made["first_date"]):
        raise SystemExit(f"stored days start {stored['days'][-days]}, forecast starts {made['first_date']}")
    print(losses(stored["cloud"][-days:], stored["temp"][-days:], made["cloud_day"], made["temp_day"]))


if __name__ == "__main__":
    main(*sys.argv[1:])

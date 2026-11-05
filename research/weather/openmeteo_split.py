"""Splits the Open-Meteo days into fit, select and test for candidates trained on the job's own source.

usage: openmeteo_split.py FIT_FIELDS.npz TEST_FIELDS.npz HISTORY.npz OUT_DIR
Writes dev.npz (6-hourly fields through the last select day), select_y.npz, test_x.npz and test_y.npz. The first 30
days of fields are lead-in. Test truth is the stored days the job scored, and the test ridge is the same walk-forward
ridge as replay_days.py. Fields of a later test day are labels of an earlier one, so test_x.npz must only be read
through inputs_of, which stops at 18 UTC the day before.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import candidate
from common import STEPS_PER_DAY

FIRST_FIT = date(2026, 7, 18)
FIRST_SELECT = date(2026, 9, 1)
FIRST_TEST = date(2026, 10, 7)
TEST_DAYS = 28
FIELD_NAMES = ("cloud", "temp", "pressure", "water", "precip", "u", "v")


def days_from(first: date, count: int) -> list:
    return [(first + timedelta(days=i)).isoformat() for i in range(count)]


def inputs_of(fields: dict, first: date, when: date) -> dict:
    """The steps a forecast of `when` may read: every step before 00 UTC of that day."""
    end = (when - first).days * STEPS_PER_DAY
    if end <= 0:
        raise ValueError(f"no fields before {when}")
    return {name: fields[name][:end] for name in FIELD_NAMES}


def walk_forward_ridge(cloud: np.ndarray, count: int) -> np.ndarray:
    """The ridge forecast of each of the last `count` days, reading only the days before it."""
    steps = np.repeat(cloud, STEPS_PER_DAY, axis=0)
    ridge = candidate.Ridge()
    return np.stack([ridge.cloud({"cloud": steps[:day * STEPS_PER_DAY]}, None) for day in range(len(cloud) - count, len(cloud))])


def split(early: dict, late: dict, history: dict) -> dict:
    """Joins the two field files, which must touch, and returns the four files as dicts."""
    start, joined = date.fromisoformat(str(early["first_date"])), date.fromisoformat(str(late["first_date"]))
    if start + timedelta(days=len(early["cloud_day"])) != joined:
        raise ValueError("field files do not touch")
    keep = (FIRST_TEST - joined).days
    dev = {name: np.concatenate([early[name], late[name][:keep * STEPS_PER_DAY]]) for name in FIELD_NAMES}
    for name in ("cloud_day", "temp_day"):
        dev[name] = np.concatenate([early[name], late[name][:keep]])
    dev.update(first_date=start.isoformat(), first_target=FIRST_FIT.isoformat())

    select_from = (FIRST_SELECT - start).days
    daily = dev["cloud_day"]
    select_y = {"days": np.array(days_from(FIRST_SELECT, len(daily) - select_from)), "cloud": daily[select_from:],
                "ridge": walk_forward_ridge(daily, len(daily) - select_from)}

    if str(history["days"][-TEST_DAYS]) != FIRST_TEST.isoformat():
        raise ValueError(f"stored test days start at {history['days'][-TEST_DAYS]}")
    test_y = {"days": history["days"][-TEST_DAYS:], "cloud": history["cloud"][-TEST_DAYS:],
              "ridge": walk_forward_ridge(history["cloud"], TEST_DAYS)}
    test_x = {name: late[name] for name in FIELD_NAMES}
    test_x.update(first_date=joined.isoformat())
    return {"dev": dev, "select_y": select_y, "test_x": test_x, "test_y": test_y}


def main(early: str, late: str, history: str, out: str) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for name, arrays in split(np.load(early), np.load(late), np.load(history)).items():
        np.savez(folder / f"{name}.npz", **arrays)


if __name__ == "__main__":
    main(*sys.argv[1:])

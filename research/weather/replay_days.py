"""Replays a candidate on the daily job's own stored days, which hold daily means.

usage: replay_days.py CANDIDATE HISTORY.npz [DAYS] [OPENMETEO.npz]
With no OPENMETEO file each day is held as four equal steps, which suits a candidate that reads daily means. With one,
the candidate reads its 6-hourly fields, built by fetch_openmeteo.py. The baselines and the truth stay on the stored
days either way. Nothing here feeds a choice of model. Prints mean cloud loss of the candidate, the ridge and
repeat-yesterday over the last DAYS days.
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np

import candidate
from common import STEPS_PER_DAY
from forecast import weather
from score import CLOUD_RANGE


def replay(model, cloud: np.ndarray, temp: np.ndarray, days: list, count: int, fields=None) -> dict:
    steps = np.repeat(cloud, STEPS_PER_DAY, axis=0)
    warm = np.repeat(temp, STEPS_PER_DAY, axis=0)
    baselines = {"ridge": candidate.Ridge(), "repeat": candidate.Repeat()}
    found = {name: [] for name in ("model", *baselines)}
    start = None if fields is None else date.fromisoformat(str(fields["first_date"]))
    for day in range(len(cloud) - count, len(cloud)):
        when = date.fromisoformat(str(days[day]))
        held = {"cloud": steps[:day * STEPS_PER_DAY], "temp": warm[:day * STEPS_PER_DAY]}
        view = held
        if fields is not None:
            # Inputs end at 18 UTC the day before the target, as in the judge.
            end = (when - start).days * STEPS_PER_DAY
            view = {name: fields[name][:end] for name in fields.files if name not in ("first_date", "first_target", "cloud_day", "temp_day")}
        found["model"].append(weather.mse(np.clip(model.cloud(view, when), *CLOUD_RANGE), cloud[day]))
        for name, each in baselines.items():
            found[name].append(weather.mse(np.clip(each.cloud(held, when), *CLOUD_RANGE), cloud[day]))
    return {name: float(np.mean(rows)) for name, rows in found.items()}


def main(path: str, history: str, count: str = "28", openmeteo: str | None = None) -> None:
    data = np.load(history)
    print(replay(candidate.load(path), data["cloud"], data["temp"], list(data["days"]), int(count), None if openmeteo is None else np.load(openmeteo)))


if __name__ == "__main__":
    main(*sys.argv[1:])

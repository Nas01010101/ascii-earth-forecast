"""Writes a forecast of temperature and pressure for each day in a range, reading only the steps before that day.

usage: predict_fields.py TEMP_MODEL PRESSURE_MODEL FIELDS.npz FIRST_DAY LAST_DAY OUT.npz
A model is repeat, month, ridge, or a network file. Several network files joined by commas forecast their mean.
The output holds `days`, `temp` and `pressure`, which fields_score.py reads.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np

import candidate
from common import STEPS_PER_DAY, daily
from forecast import weather
from openmeteo_split import FIELD_NAMES, days_from, inputs_of

MONTH_DAYS = 30


def forecaster(name: str, target: str):
    if name == "repeat":
        return lambda fields, when: daily(fields[target][-STEPS_PER_DAY:])[0]
    if name == "month":
        return lambda fields, when: daily(fields[target][-MONTH_DAYS * STEPS_PER_DAY:]).mean(axis=0)
    if name == "ridge":
        def ridge(fields, when):
            series = daily(fields[target][-candidate.RIDGE_DAYS * STEPS_PER_DAY:])
            return weather.predict(series, len(series))[0]
        return ridge
    models = [candidate.load(path) for path in name.split(",")]
    for model, path in zip(models, name.split(",")):
        if model.spec.target != target:
            raise ValueError(f"{path} forecasts {model.spec.target}, not {target}")
    return lambda fields, when: np.mean([model.forecast(fields, when) for model in models], axis=0)


def predict(model, fields: dict, first: date, days: list) -> np.ndarray:
    return np.stack([model(inputs_of(fields, first, date.fromisoformat(day)), date.fromisoformat(day)) for day in days])


def main(temp: str, pressure: str, path: str, first_day: str, last_day: str, out: str) -> None:
    held = np.load(path)
    fields = {name: held[name] for name in FIELD_NAMES}
    first, start = date.fromisoformat(str(held["first_date"])), date.fromisoformat(first_day)
    days = days_from(start, (date.fromisoformat(last_day) - start + timedelta(days=1)).days)
    np.savez(out, days=np.array(days), temp=predict(forecaster(temp, "temp"), fields, first, days),
             pressure=predict(forecaster(pressure, "pressure"), fields, first, days))


if __name__ == "__main__":
    main(*sys.argv[1:])

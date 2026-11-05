"""Writes a model's forecast of each day in a range, reading only the steps before that day.

usage: predict_net.py MODEL FIELDS.npz FIRST_DAY LAST_DAY OUT.npz
MODEL is what candidate.load reads, or month for the mean of the last 30 days. A day: prefix gives the model each day's
24-hour mean cloud in place of its 4 samples. The output holds `days` and `cloud`, which parts_score.py and stack.py read.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np

import candidate
from common import STEPS_PER_DAY, daily
from openmeteo_split import FIELD_NAMES, days_from, inputs_of
from score import CLOUD_RANGE

MONTH_DAYS = 30
DAILY = "day:"


class Month:
    def cloud(self, fields: dict, when: date) -> np.ndarray:
        return daily(fields["cloud"][-MONTH_DAYS * STEPS_PER_DAY:]).mean(axis=0)


def load_model(name: str):
    return Month() if name == "month" else candidate.load(name)


def predict(model, fields: dict, first: date, days: list) -> np.ndarray:
    rows = [model.cloud(inputs_of(fields, first, date.fromisoformat(day)), date.fromisoformat(day)) for day in days]
    return np.clip(np.stack(rows), *CLOUD_RANGE)


def main(name: str, path: str, first_day: str, last_day: str, out: str) -> None:
    held = np.load(path)
    fields = {field: held[field] for field in FIELD_NAMES}
    if name.startswith(DAILY):
        # Truth is a 24-hour mean, and 4 samples of a day miss it by about 73 squared points.
        name = name[len(DAILY):]
        fields["cloud"] = np.repeat(held["cloud_day"], STEPS_PER_DAY, axis=0)
    first, start = date.fromisoformat(str(held["first_date"])), date.fromisoformat(first_day)
    days = days_from(start, (date.fromisoformat(last_day) - start + timedelta(days=1)).days)
    np.savez(out, days=np.array(days), cloud=predict(load_model(name), fields, first, days))


if __name__ == "__main__":
    main(*sys.argv[1:])

"""The judge: percent lower area-weighted cloud loss than the production ridge, one number per calendar quarter.

usage: score.py CANDIDATE DATA.npz
DATA holds 6-hourly fields from first_date, and targets run from first_target to its last day. A candidate reads
a view that ends at 18 UTC the day before its target, which is the last complete step the daily job has at 01:37 UTC.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np

import candidate
from common import FIELDS, STEPS_PER_DAY, daily
from forecast import weather

CLOUD_RANGE = (0.0, 100.0)  # run.CLOUD_RANGE


def quarter(when: date) -> str:
    return f"{when.year}Q{(when.month - 1) // 3 + 1}"


def losses(model, data) -> dict:
    """{quarter: [(model loss, ridge loss), one per target day]}."""
    fields = {name: data[name] for name in FIELDS}
    start = date.fromisoformat(str(data["first_date"]))
    first = (date.fromisoformat(str(data["first_target"])) - start).days
    truth = daily(fields["cloud"])
    ridge = candidate.Ridge()
    found = {}
    for day in range(first, len(truth)):
        when = start + timedelta(days=day)
        view = {name: field[:day * STEPS_PER_DAY] for name, field in fields.items()}
        ours = np.clip(model.cloud(view, when), *CLOUD_RANGE)
        base = np.clip(ridge.cloud(view, when), *CLOUD_RANGE)
        found.setdefault(quarter(when), []).append((weather.mse(ours, truth[day]), weather.mse(base, truth[day])))
    return found


def percent_lower(rows: list) -> float:
    ours, base = np.array(rows).sum(axis=0)
    return float(100 * (1 - ours / base))


def main(path: str, data_path: str) -> None:
    found = losses(candidate.load(path), np.load(data_path))
    print(",".join(f"{percent_lower(found[q]):.3f}" for q in sorted(found)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

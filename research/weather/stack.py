"""A linear correction of forecasts: one set of weights per latitude row, pulled toward one set for the globe.

usage:
  stack.py fit DEV.npz FIRST LAST OUT.npz STRENGTHS SELECT_FIRST SELECT_LAST FEATURE.npz ...
  stack.py apply COEF.npz FIRST LAST OUT.npz FEATURE.npz ...
  stack.py mean OUT.npz FORECAST.npz ...
A feature is a forecast file holding `days` and `cloud`, and NAME.npz:smooth takes that forecast's 3x3 mean. `fit` fits
on FIRST..LAST of DEV's daily truth, tries each strength in STRENGTHS (comma separated) and keeps the lowest squared
error on SELECT_FIRST..SELECT_LAST. Squared error rewards a hedged forecast, which is why shrinking and smoothing a
sharp forecast can lower it.
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np

from forecast import weather
from predict_net import days_from
from score import CLOUD_RANGE

# Keeps the intercept on the percent scale of the forecasts, so the same strength applies to every weight.
INTERCEPT = 50.0
GLOBAL_STRENGTH = 1.0


def _days(first: str, last: str) -> list:
    start = date.fromisoformat(first)
    return days_from(start, (date.fromisoformat(last) - start).days + 1)


def smooth(cloud: np.ndarray) -> np.ndarray:
    """3x3 mean of (days, rows, cols); longitude wraps and the poles repeat their own row."""
    padded = np.concatenate([cloud[:, :1], cloud, cloud[:, -1:]], axis=1)
    stacked = sum(np.roll(padded, shift, axis=2) for shift in (-1, 0, 1))
    return (stacked[:, :-2] + stacked[:, 1:-1] + stacked[:, 2:]) / 9


def feature(spec: str, days: list) -> np.ndarray:
    path, _, kind = spec.partition(":")
    held = np.load(path)
    known = {str(day): i for i, day in enumerate(held["days"])}
    cloud = held["cloud"][[known[day] for day in days]]
    return smooth(cloud) if kind == "smooth" else cloud


def design(specs: list, days: list) -> np.ndarray:
    """(days, rows, cols, weights), the intercept last."""
    rows = [feature(spec, days) for spec in specs]
    return np.stack([*rows, np.full_like(rows[0], INTERCEPT)], axis=-1).astype(np.float64)


def fit_rows(x: np.ndarray, y: np.ndarray, strength: float) -> np.ndarray:
    """(rows, weights). A row's weights minimise its squared error plus strength times the distance to the global fit."""
    x_rows = x.reshape(x.shape[0], x.shape[1], -1, x.shape[3]).transpose(1, 0, 2, 3).reshape(x.shape[1], -1, x.shape[3])
    y_rows = y.transpose(1, 0, 2).reshape(y.shape[1], -1)
    size = x.shape[3]
    area = weather.AREA[:, 0]
    grams = [(row.T @ row, row.T @ want) for row, want in zip(x_rows, y_rows)]
    total = sum(w * g for w, (g, _) in zip(area, grams))
    moment = sum(w * m for w, (_, m) in zip(area, grams))
    pull = GLOBAL_STRENGTH * np.eye(size)
    shared = np.linalg.solve(total + pull, moment)
    return np.stack([shared + np.linalg.solve(g + strength * np.eye(size), m - g @ shared) for g, m in grams])


def apply_rows(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.clip(np.einsum("drcw,rw->drc", x, coef), *CLOUD_RANGE)


def loss(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([weather.mse(a, b) for a, b in zip(pred, truth)]))


def truth_of(dev: dict, days: list) -> np.ndarray:
    first = date.fromisoformat(str(dev["first_date"]))
    return dev["cloud_day"][[(date.fromisoformat(day) - first).days for day in days]].astype(np.float64)


def fit(dev_path: str, first: str, last: str, out: str, strengths: str, select_first: str, select_last: str, *specs: str) -> None:
    dev = np.load(dev_path)
    train, select = _days(first, last), _days(select_first, select_last)
    x, y = design(list(specs), train), truth_of(dev, train)
    x_select, y_select = design(list(specs), select), truth_of(dev, select)
    found = []
    for strength in (float(s) for s in strengths.split(",")):
        coef = fit_rows(x, y, strength)
        found.append((loss(apply_rows(coef, x_select), y_select), strength, coef))
        print(f"strength {strength:g} select loss {found[-1][0]:.2f}")
    best, strength, coef = min(found, key=lambda row: row[0])
    print(f"kept strength {strength:g} loss {best:.2f}")
    np.savez(out, coef=coef, strength=strength, specs=np.array(specs), days=np.array(select),
             cloud=apply_rows(coef, x_select))


def apply(coef_path: str, first: str, last: str, out: str, *specs: str) -> None:
    held = np.load(coef_path)
    days = _days(first, last)
    np.savez(out, days=np.array(days), cloud=apply_rows(held["coef"], design(list(specs), days)))


def mean(out: str, *paths: str) -> None:
    """Writes the plain mean of forecast files that hold the same days."""
    held = [np.load(path) for path in paths]
    days = held[0]["days"]
    if any(list(one["days"]) != list(days) for one in held):
        raise ValueError("forecasts hold different days")
    np.savez(out, days=days, cloud=np.mean([one["cloud"] for one in held], axis=0))


if __name__ == "__main__":
    {"fit": fit, "apply": apply, "mean": mean}[sys.argv[1]](*sys.argv[2:])

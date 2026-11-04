"""One small model per object and horizon, refit monthly on that object's rows whose truth had arrived.

usage: per_object.py FIT_PARQUET OUT_CSV {--select | --model NAME | --diagnose NAME}
--select scores every model on t0 in 2018, the part of the published objects' history before the test window.
--model scores one model once on t0 from 2019-01-01 to 2020-06-30 and writes the head-to-head table.
--diagnose scores one model with and without time separation; the shuffled rows are not a forecast result.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from data import day, inputs, load, own_drift
from head_to_head import MONTH_DAYS, OBJECTS, TEST_END, TEST_START, TRAIN_START, table

SELECT_START = day(2018)
LIU_START = day(2019, 6, 1)  # the paper plots its benchmark from June 2019 to June 2020
MIN_ROWS = 60


class Ridge:
    """Ridge on standardised inputs, with an intercept."""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mean, self.std = x.mean(0), np.maximum(x.std(0), 1e-6)
        z = (x - self.mean) / self.std
        self.intercept = y.mean()
        self.w = np.linalg.solve(z.T @ z + self.penalty * np.eye(z.shape[1]), z.T @ (y - self.intercept))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + ((x - self.mean) / self.std) @ self.w


def lightgbm(**params):
    def make():
        from lightgbm import LGBMRegressor

        return LGBMRegressor(verbose=-1, n_jobs=8, random_state=0, **params)

    return make


def nominal(d: dict) -> dict:
    """Inputs read the nominal horizon: the true dt depends on when the truth elements arrive."""
    return {**d, "dt": d["horizon"].astype(np.float32)}


def drift_only(d: dict) -> np.ndarray:
    return own_drift(nominal(d)).astype(np.float64)[:, None]


def all_inputs(d: dict) -> np.ndarray:
    return inputs(nominal(d)).astype(np.float64)


MODELS = {
    "ridge_drift": (drift_only, lambda: Ridge(1.0)),
    "ridge_all": (all_inputs, lambda: Ridge(100.0)),
    "forest": (all_inputs, lightgbm(boosting_type="rf", n_estimators=200, bagging_freq=1, bagging_fraction=0.7, feature_fraction=0.5, min_child_samples=5)),
    "boost": (all_inputs, lightgbm(n_estimators=200, learning_rate=0.03, max_depth=3, min_child_samples=5, objective="l1")),
}


def walk_forward(d: dict, x: np.ndarray, make, start: float, end: float) -> np.ndarray:
    """Corrections for rows with t0 in [start, end); a row with too little arrived history stays uncorrected."""
    c = np.zeros((len(x), d["y"].shape[1]), dtype=np.float64)
    arrival = d["epoch"] + d["dt"].astype(np.float64)
    for norad in np.unique(d["norad"]):
        for h in np.unique(d["horizon"]):
            rows = np.flatnonzero((d["norad"] == norad) & (d["horizon"] == h))
            for month in np.arange(start, end, MONTH_DAYS):
                test = rows[(d["epoch"][rows] >= month) & (d["epoch"][rows] < min(month + MONTH_DAYS, end))]
                train = rows[arrival[rows] < month]
                if len(test) == 0 or len(train) < MIN_ROWS:
                    continue
                for t in range(c.shape[1]):
                    c[test, t] = make().fit(x[train], d["y"][train, t].astype(np.float64)).predict(x[test])
    return c


def shuffled_folds(d: dict, x: np.ndarray, make, start: float, end: float, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Time separation removed: a row in [start, end) is predicted from the object's rows on both sides of it."""
    c = np.zeros((len(x), d["y"].shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for norad in np.unique(d["norad"]):
        for h in np.unique(d["horizon"]):
            rows = np.flatnonzero((d["norad"] == norad) & (d["horizon"] == h))
            window = rows[(d["epoch"][rows] >= start) & (d["epoch"][rows] < end)]
            part = rng.integers(folds, size=len(window))
            for f in range(folds):
                test, train = window[part == f], np.setdiff1d(rows, window[part == f])
                if len(test) == 0 or len(train) < MIN_ROWS:
                    continue
                for t in range(c.shape[1]):
                    c[test, t] = make().fit(x[train], d["y"][train, t].astype(np.float64)).predict(x[test])
    return c


def subset(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def in_window(d: dict, c: np.ndarray, start: float, end: float) -> list:
    m = (d["epoch"] >= start) & (d["epoch"] < end)
    return table(subset(d, m), c[m])


def score(d: dict, name: str, start: float, end: float) -> list:
    features, make = MODELS[name]
    return in_window(d, walk_forward(d, features(d), make, start, end), start, end)


def diagnose(d: dict, name: str) -> list:
    """One model with and without time separation, on the full test window and on the window Liu et al. report."""
    features, make = MODELS[name]
    x, rows = features(d), []
    for fit, label in ((walk_forward, "walk"), (shuffled_folds, "shuffled")):
        c = fit(d, x, make, TEST_START, TEST_END)
        rows += [(f"{name}_{label}_full", *r) for r in in_window(d, c, TEST_START, TEST_END)]
        rows += [(f"{name}_{label}_liu", *r) for r in in_window(d, c, LIU_START, TEST_END)]
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fit"), p.add_argument("out")
    p.add_argument("--select", action="store_true")
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--diagnose", choices=sorted(MODELS))
    a = p.parse_args()
    d = load(a.fit, TRAIN_START, TEST_END, shards={n % 64 for n in OBJECTS})
    d = subset(d, np.isin(d["norad"], OBJECTS))
    if a.diagnose:
        rows = diagnose(d, a.diagnose)
    else:
        if a.select:
            names, start, end = sorted(MODELS), SELECT_START, TEST_START
        else:
            names, start, end = [a.model], TEST_START, TEST_END
        rows = [(name, *r) for name in names for r in score(d, name, start, end)]
    with open(a.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "norad", "horizon", "samples", "raw_km", "model_km", "gain", "gain_lo", "gain_hi"])
        writer.writerows(rows)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()

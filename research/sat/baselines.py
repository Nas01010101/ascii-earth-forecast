"""Non-neural corrections, scored on 2024 after fitting on earlier years.

usage: baselines.py FIT_PARQUET HOLDOUT_X OUT_DIR NAME
NAME is one of raw, linear, own, lgbm. Writes val_NAME.csv (5 group scores) and hold_NAME.npy.
lgbm fits before 2024 for validation and is refit on all of fit.parquet for the holdout. own, linear and lgbm
read the true dt, which depends on when the truth elements arrive, so they know slightly more than t0 does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from build_samples import LOOKBACKS
from data import col, day, load, metric_weights, own_drift
from score import group_scores

TRAIN_SHARDS = set(range(16))
VAL_SHARDS = set(range(32))
START = day(2020)
VAL_START = day(2024)
END = day(2025)
LEO_REV_PER_DAY = 11.25
MAX_ALONG_KM = 5000.0
WINDOW_DAYS = 30.0
REFIT_DAYS = 7.0


def drift_terms(age: np.ndarray, ndot: np.ndarray, bstar: np.ndarray) -> np.ndarray:
    # The five terms of forecast/orbits.py:_drift_inputs.
    return np.stack([age, age ** 2, ndot * age ** 2, bstar * age ** 2, np.ones_like(age)], axis=1)


def drift_inputs(d: dict) -> np.ndarray:
    return drift_terms(d["dt"].astype(np.float64), col(d, "ndot").astype(np.float64), col(d, "bstar").astype(np.float64))


def ridge(a: np.ndarray, b: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    return np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ b)


def linear_walk(target: dict) -> np.ndarray:
    """Refits weekly on the last 30 days of hindcasts, the way the daily job does, for LEO only.

    A hindcast is known at its t0, so this needs no labels and runs the same on the sealed year. The
    drag terms come from the element set at t0 rather than the older one, a small difference from the job.
    """
    _, first = np.unique(target["norad"].astype(np.int64).astype(np.float64) * 1e6 + target["epoch"], return_index=True)
    no_rev, ndot, bstar, epoch = col(target, "no_rev")[first], col(target, "ndot")[first], col(target, "bstar")[first], target["epoch"][first]
    rows = []
    for a in LOOKBACKS:
        age = col(target, f"age_{int(a)}")[first].astype(np.float64)
        along = col(target, f"along_{int(a)}")[first].astype(np.float64)
        ok = np.isfinite(age) & np.isfinite(along) & (no_rev >= LEO_REV_PER_DAY) & (np.abs(along) < MAX_ALONG_KM)
        rows.append((drift_terms(age[ok], ndot[ok], bstar[ok]), along[ok], epoch[ok]))
    a_all = np.concatenate([r[0] for r in rows])
    along_all = np.concatenate([r[1] for r in rows])
    known = np.concatenate([r[2] for r in rows])

    out = np.zeros((len(target["dt"]), 3), dtype=np.float32)
    t_all = target["epoch"]
    t_a = drift_inputs(target)
    target_leo = col(target, "no_rev") >= LEO_REV_PER_DAY
    for start in np.arange(t_all.min(), t_all.max() + REFIT_DAYS, REFIT_DAYS):
        now = (t_all >= start) & (t_all < start + REFIT_DAYS) & target_leo
        use = (known < start) & (known >= start - WINDOW_DAYS)
        if now.any() and use.sum() >= 100:
            out[now, 1] = t_a[now] @ ridge(a_all[use], along_all[use])
    return out


def fit_lgbm(train: dict):
    import lightgbm as lgb
    x = np.column_stack([train["x"], train["dt"], own_drift(train)])
    w = metric_weights(train)
    models = []
    for c in range(3):
        models.append(lgb.train(
            {"objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
             "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "num_threads": 4, "seed": 0},
            lgb.Dataset(x, label=train["y"][:, c], weight=w), num_boost_round=300))
    return models


def predict_lgbm(models, d: dict) -> np.ndarray:
    x = np.column_stack([d["x"], d["dt"], own_drift(d)])
    return np.stack([m.predict(x) for m in models], axis=1).astype(np.float32)


def correction(name: str, train: dict, target: dict) -> np.ndarray:
    n = len(target["dt"])
    if name == "raw":
        return np.zeros((n, 3), dtype=np.float32)
    if name == "own":
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 1] = own_drift(target)
        return out
    if name == "linear":
        return linear_walk(target)
    if name == "lgbm":
        return predict_lgbm(fit_lgbm(train), target)
    raise SystemExit(f"unknown baseline {name}")


def main(fit: str, holdout: str, out: str, name: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = name == "lgbm"  # the others learn nothing from the fit set
    train = load(fit, START, VAL_START, shards=TRAIN_SHARDS, rate=0.5) if fitted else None
    val = load(fit, VAL_START, END, shards=VAL_SHARDS)
    c = correction(name, train, val)
    scores = group_scores(val["norad"], val["horizon"], val["radius"], val["y"].astype(np.float64), c.astype(np.float64))
    (out_dir / f"val_{name}.csv").write_text(",".join(f"{s:.4f}" for s in scores) + "\n")
    print(name, "val", np.round(scores, 2), "mean", round(float(scores.mean()), 2))

    hold = load(holdout, END, day(2026), labelled=False)
    refit = load(fit, START, END, shards=TRAIN_SHARDS, rate=0.5) if fitted else None
    np.save(out_dir / f"hold_{name}.npy", correction(name, refit, hold))


if __name__ == "__main__":
    main(*sys.argv[1:5])

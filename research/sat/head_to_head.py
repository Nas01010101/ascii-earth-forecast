"""Correction on the three objects of Liu et al. 2021 (SSC21-IX-02), trained only on rows that end before the test window.

usage: head_to_head.py FIT_PARQUET OUT_CSV [--seeds 3]
Trains on t0 and truth in 2017-2018 with the shipped recipe, tests on t0 from 2019-01-01 to 2020-06-30,
the part of the published period (June 2017 to June 2020) that has earlier data to train on.
Writes one row per object and horizon, plus the mean of the three objects.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from build_samples import HORIZONS
from data import day, load
from sat_common import miss_km
from train_net import fit, predict, tensors, train_shards

OBJECTS = (27845, 20774, 23975)
TRAIN_START, TEST_START, TEST_END = day(2017), day(2019), day(2020, 7, 1)
MONTH_DAYS = 30.44
BOOTSTRAP_DRAWS = 2000


def percent_gain(raw: np.ndarray, model: np.ndarray) -> float:
    return float(100.0 * (1.0 - model.sum() / raw.sum()))


def month_blocks(epoch: np.ndarray) -> np.ndarray:
    """Fixed 30.44-day blocks on one grid for every object."""
    return np.floor(epoch / MONTH_DAYS).astype(int)


def month_bootstrap(raw: np.ndarray, model: np.ndarray, months: np.ndarray, labels: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Gain over resampled blocks: rows 5 days apart share most of their error, so rows are not independent.

    picks index labels and are shared across objects, so the mean of three objects keeps their dependence.
    """
    raw_by_month = np.array([raw[months == m].sum() for m in labels])
    model_by_month = np.array([model[months == m].sum() for m in labels])
    return 100.0 * (1.0 - model_by_month[picks].sum(1) / raw_by_month[picks].sum(1))


def table(d: dict, c: np.ndarray) -> list:
    y, radius = d["y"].astype(np.float64), d["radius"].astype(np.float64)
    raw_km, model_km = miss_km(radius, y, np.zeros_like(y)), miss_km(radius, y, c)
    months, rows = month_blocks(d["epoch"]), []
    for h in HORIZONS:
        draws = []
        labels = np.unique(months[(d["horizon"] == h) & np.isin(d["norad"], OBJECTS)])
        picks = np.random.default_rng(0).integers(0, len(labels), size=(BOOTSTRAP_DRAWS, len(labels)))
        for norad in OBJECTS:
            m = (d["norad"] == norad) & (d["horizon"] == h)
            if not m.any():
                continue
            boot = month_bootstrap(raw_km[m], model_km[m], months[m], labels, picks)
            draws.append(boot)
            rows.append((norad, h, int(m.sum()), round(float(raw_km[m].mean()), 2), round(float(model_km[m].mean()), 2),
                         round(percent_gain(raw_km[m], model_km[m]), 2), round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)))
        if len(draws) == len(OBJECTS):
            joint = np.mean(draws, axis=0)
            gains = [r[5] for r in rows[-len(OBJECTS):]]
            rows.append(("mean", h, "", "", "", round(float(np.mean(gains)), 2), round(float(np.percentile(joint, 2.5)), 2), round(float(np.percentile(joint, 97.5)), 2)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fit"), p.add_argument("out")
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = load(a.fit, TRAIN_START, TEST_START, shards=train_shards(32))
    test = load(a.fit, TEST_START, TEST_END, shards={n % 64 for n in OBJECTS})
    keep = np.isin(test["norad"], OBJECTS)
    test = {k: v[keep] for k, v in test.items()}
    print(f"train {len(train['dt'])} test {len(test['dt'])} rows on {device}", flush=True)
    members = []
    for seed in range(a.seeds):
        args = argparse.Namespace(depth=6, width=256, epochs=12, lr=1e-3, dropout=0.0, seed=seed, max_drift=40000.0,
                                  no_history=False, no_weather=True, solar_ratio=False, fixed_dt=True)
        members.append(predict(fit(train, args, device), tensors(test, args), device).astype(np.float64))
    for seed, member in enumerate(members):
        print("seed", seed, *[r[1::4] for r in table(test, member) if r[0] == "mean"], flush=True)
    rows = table(test, np.mean(members, axis=0))
    with open(a.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["norad", "horizon", "samples", "raw_km", "model_km", "gain", "gain_lo", "gain_hi"])
        writer.writerows(rows)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()

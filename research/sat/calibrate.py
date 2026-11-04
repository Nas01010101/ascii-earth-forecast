"""Per-object bias from the network's own past misses, using only rows whose truth had arrived by t0.

usage: calibrate.py FIT_PARQUET OUT_CSV OOF.npy
Tries each half-life and prior on 2024 validation and writes one line per setting, with the 5 group scores.
Applied to the along-track correction only, which holds nearly all of the error at long horizons.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from data import load
from score import group_scores
from train_net import END, VAL_SHARDS, VAL_START

SETTINGS = ((30.0, 2.0), (30.0, 8.0), (90.0, 2.0), (90.0, 8.0))
WARM_ROWS = 3  # rows before the object's own spread is known well enough to limit a manoeuvre
LIMIT_SPREADS = 3.0


def trailing_bias(epoch: np.ndarray, dt: np.ndarray, residual: np.ndarray, half_life: float, prior: float) -> np.ndarray:
    """Decayed mean of the residuals that had arrived before each row's t0, shrunk towards zero.

    Rows must be one object at one horizon, sorted by epoch. A row's residual becomes usable at epoch + dt.
    """
    arrival = epoch + dt
    order = np.argsort(arrival, kind="stable")
    rate = np.log(2.0) / half_life
    out = np.zeros(len(epoch), dtype=np.float64)
    total = weight = square = 0.0
    now, n, j = 0.0, 0, 0
    for i in range(len(epoch)):
        while j < len(order) and arrival[order[j]] < epoch[i]:
            t = arrival[order[j]]
            decay = np.exp(-rate * (t - now)) if n else 1.0
            total, weight, square = total * decay, weight * decay, square * decay
            r = float(residual[order[j]])
            if n >= WARM_ROWS:
                limit = LIMIT_SPREADS * np.sqrt(square / weight)
                r = min(max(r, -limit), limit)
            total, weight, square = total + r, weight + 1.0, square + r * r
            now, n, j = t, n + 1, j + 1
        if n:
            decay = np.exp(-rate * (epoch[i] - now))
            out[i] = total * decay / (weight * decay + prior)
    return out


def calibrated(d: dict, c: np.ndarray, half_life: float, prior: float) -> np.ndarray:
    """Returns c with the bias added to the along-track column, in the row order of c."""
    y_along, out = d["y"][:, 1].astype(np.float64), c.astype(np.float64).copy()
    order = np.lexsort((d["epoch"], d["horizon"], d["norad"]))
    key = np.stack([d["norad"][order], d["horizon"][order]], axis=1)
    edges = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
    for rows in np.split(order, edges):
        residual = np.nan_to_num(y_along[rows] - out[rows, 1], posinf=0.0, neginf=0.0)
        out[rows, 1] += trailing_bias(d["epoch"][rows], d["dt"][rows].astype(np.float64), residual, half_life, prior)
    return out


def main(fit: str, out: str, oof: str) -> None:
    d = load(fit, VAL_START, END, shards=VAL_SHARDS)
    c = np.load(oof).astype(np.float64)
    y, rows = d["y"].astype(np.float64), []
    base = group_scores(d["norad"], d["horizon"], d["radius"], y, c)
    rows.append((0.0, 0.0, *np.round(base, 4)))
    for half_life, prior in SETTINGS:
        rows.append((half_life, prior, *np.round(group_scores(d["norad"], d["horizon"], d["radius"], y, calibrated(d, c, half_life, prior)), 4)))
        print(half_life, prior, "mean", round(float(np.mean(rows[-1][2:])), 3), flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["half_life", "prior", "g0", "g1", "g2", "g3", "g4"])
        writer.writerows(rows)
    print("base", round(float(base.mean()), 3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

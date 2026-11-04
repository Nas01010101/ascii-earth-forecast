"""Score the mean of several validation predictions, to size an ensemble before training one.

usage: ensemble.py FIT_PARQUET OUT_CSV A.npy B.npy ...
       ensemble.py --save-mean OUT.npy A.npy B.npy ...
Writes the 5 group scores of the mean correction, in the same order as val_NAME.csv,
or with --save-mean the mean itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from data import load
from score import group_scores
from train_net import END, VAL_SHARDS, VAL_START


def mean_correction(arrays: list) -> np.ndarray:
    return np.mean(arrays, axis=0, dtype=np.float64)


def save_mean(out: str, *paths: str) -> None:
    """Holdout members have no labels here; the mean is written for the sealed scorer."""
    np.save(out, mean_correction([np.load(p) for p in paths]).astype(np.float32))


def main(fit: str, out: str, *paths: str) -> None:
    if fit == "--save-mean":
        return save_mean(out, *paths)
    d = load(fit, VAL_START, END, shards=VAL_SHARDS)
    c = mean_correction([np.load(p) for p in paths])
    scores = group_scores(d["norad"], d["horizon"], d["radius"], d["y"].astype(np.float64), c)
    Path(out).write_text(",".join(f"{s:.4f}" for s in scores) + "\n")
    print([Path(p).stem for p in paths], np.round(scores, 2), "mean", round(float(scores.mean()), 2))


if __name__ == "__main__":
    main(*sys.argv[1:])

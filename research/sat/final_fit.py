"""Refits a recipe for one seed on all labelled rows through a last year.

usage: final_fit.py SAMPLES_DIR OUT_DIR NAME --seed N [--last-year 2025] [--dense]
Through 2025 the weights can ship. Through 2024 they leave 2025 to compare recipes on, a burned year, so a diagnostic.
--dense trains on any-time rows with real dt and shows 15% of them with little history; SAMPLES_DIR is then the dense folder.
Keeps the pick's 48 shards: all 64 would need about 20 GB of host memory on a box shared with other jobs.
The csv holds in-sample scores, a check on the fit and never evidence of skill; nothing labelled is left out.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data import day
from score import HORIZON, group_scores
from train_net import fit, predict, tensors, train_rows, train_shards

LAST_DENSE_BUCKET = 16.0

RECIPE = dict(
    depth=6, width=256, epochs=12, lr=1e-3, dropout=0.0, rate=1.0, max_drift=40000.0,
    no_history=False, no_weather=True, solar_ratio=False, fixed_dt=True, start_year=2020, extra_shards=32,
    history_dropout=0.0,
)
DENSE = {**RECIPE, "fixed_dt": False, "history_dropout": 0.15}


def training_rows(samples: str, seed: int, last_year: int, dense: bool) -> tuple:
    args = SimpleNamespace(**(DENSE if dense else RECIPE), seed=seed, train_samples=samples, last_year=last_year)
    return args, train_rows(args, day(args.start_year), day(last_year + 1), train_shards(args.extra_shards))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("samples"), p.add_argument("out"), p.add_argument("name")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--last-year", type=int, default=2025)
    p.add_argument("--dense", action="store_true")
    cli = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cli.out)

    args, train = training_rows(cli.samples, cli.seed, cli.last_year, cli.dense)
    print(f"train {len(train['dt'])} rows on {device}", flush=True)
    net = fit(train, args, device)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out / f"{cli.name}.pt")
    c = predict(net, tensors(train, args), device).astype(np.float64)
    at = LAST_DENSE_BUCKET if cli.dense else HORIZON  # dense rows sit in buckets, with no row at exactly 15 days
    scores = group_scores(train["norad"], train["horizon"], train["radius"], train["y"].astype(np.float64), c, at=at)
    (out / f"{cli.name}.csv").write_text(",".join(f"{s:.4f}" for s in scores) + "\n")
    print(cli.name, "in-sample", np.round(scores, 2), flush=True)


if __name__ == "__main__":
    main()

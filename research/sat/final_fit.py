"""Refits the picked recipe through 2025 for one seed, after the holdout reveal. These are the weights that ship.

usage: final_fit.py SAMPLES_DIR OUT_DIR NAME --seed N
Keeps the pick's 48 shards: all 64 would need about 20 GB of host memory on a box shared with other jobs.
The csv holds in-sample scores, a check on the fit and never evidence of skill; nothing labelled is left out.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data import day, load_folder
from score import group_scores
from train_net import fit, predict, tensors, train_shards

RECIPE = dict(
    depth=6, width=256, epochs=12, lr=1e-3, dropout=0.0, rate=1.0, max_drift=40000.0,
    no_history=False, no_weather=True, solar_ratio=False, fixed_dt=True, start_year=2020, extra_shards=32,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("samples"), p.add_argument("out"), p.add_argument("name")
    p.add_argument("--seed", type=int, required=True)
    cli = p.parse_args()
    args = SimpleNamespace(**RECIPE, seed=cli.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cli.out)

    train = load_folder(cli.samples, day(args.start_year), day(2026), shards=train_shards(args.extra_shards))
    print(f"train {len(train['dt'])} rows on {device}", flush=True)
    net = fit(train, args, device)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out / f"{cli.name}.pt")
    c = predict(net, tensors(train, args), device).astype(np.float64)
    scores = group_scores(train["norad"], train["horizon"], train["radius"], train["y"].astype(np.float64), c)
    (out / f"{cli.name}.csv").write_text(",".join(f"{s:.4f}" for s in scores) + "\n")
    print(cli.name, "in-sample", np.round(scores, 2), flush=True)


if __name__ == "__main__":
    main()

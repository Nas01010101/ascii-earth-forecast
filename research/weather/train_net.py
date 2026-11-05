"""Trains a U-Net on the training years, stopping at the best validation epoch.

usage: train_net.py ERA5_DIR STATS.npz VAL.npz OUT.pt OUT.csv --seed N --vars cloud,temp --steps 0 --lags 3 --target cloud
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from datetime import date

import numpy as np
import torch

from common import FIELDS, load_year
from features import Frames, LATS, Spec
from nets import UNet

FIRST_YEAR, LAST_TRAIN = 1979, 2018
AREA = torch.from_numpy(np.cos(np.radians(LATS)).astype(np.float32))[:, None]


def weighted(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-pixel squared error with the judge's area weights, in the target's unit squared."""
    area = AREA.to(pred.device)
    return (((pred - truth) ** 2) * area).sum((1, 2)) / (area.sum() * pred.shape[-1])


@torch.no_grad()
def loss_of(net, frames: Frames, days: np.ndarray, size: int = 256) -> float:
    net.eval()
    total = 0.0
    for part in np.array_split(days, max(1, len(days) // size)):
        index = torch.from_numpy(part)
        x, base = frames.batch(index)
        total += float(weighted(frames.limited(base + net(x) * frames.target_std), frames.truth(index)).sum())
    return total / len(days)


def train(frames: Frames, val: Frames, val_days: np.ndarray, spec: Spec, args, device: str):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = UNet(spec.channels, args.width, args.place).to(device)
    days = np.arange(frames.first_day(), len(frames.daily["cloud"]))
    steps = args.epochs * (len(days) // args.batch)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr, total_steps=steps, pct_start=0.05)
    best, best_state, rows = float("inf"), None, []
    for epoch in range(args.epochs):
        net.train()
        started, seen, order = time.time(), 0.0, rng.permutation(days)
        for k in range(len(days) // args.batch):
            index = torch.from_numpy(order[k * args.batch: (k + 1) * args.batch])
            x, base = frames.batch(index)
            loss = weighted(base + net(x) * frames.target_std, frames.truth(index)).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10 * frames.target_std)
            optimiser.step()
            schedule.step()
            seen += loss.item()
        checked = loss_of(net, val, val_days)
        rows.append([epoch, round(seen / (len(days) // args.batch), 2), round(checked, 2), round(time.time() - started, 1)])
        print(*rows[-1], flush=True)
        if checked < best:
            best, best_state = checked, {k: v.cpu().clone() for k, v in net.state_dict().items()}
    return best_state, rows, best


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("era5", "stats", "val", "out", "log"):
        parser.add_argument(name)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vars", default="cloud,temp")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--target", default="cloud", choices=FIELDS)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--place", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_TRAIN)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = Spec(tuple(args.vars.split(",")), args.steps, args.lags, args.target)
    stats = dict(np.load(args.stats))
    names = set(spec.variables) | {"cloud", "temp", spec.target}
    years = range(args.first_year, args.last_year + 1)
    fields = {n: np.concatenate([load_year(args.era5, n, y) for y in years]) for n in names}
    frames = Frames(fields, date(args.first_year, 1, 1), spec, stats, device)
    del fields
    held = np.load(args.val)
    start = date.fromisoformat(str(held["first_date"]))
    val = Frames({n: held[n] for n in names}, start, spec, stats, device)
    val_days = np.arange((date.fromisoformat(str(held["first_target"])) - start).days, len(val.daily["cloud"]))
    state, rows, best = train(frames, val, val_days, spec, args, device)
    keep = {k: v for k, v in stats.items() if k in ("land_sea_mask", "geopotential_at_surface") or k.split("_", 1)[-1] in names}
    torch.save({"kind": "net", "state": state, "spec": asdict(spec), "width": args.width, "place": args.place, "stats": keep, "args": vars(args)}, args.out)
    with open(args.log, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])
        writer.writerows(rows)
    print("best validation loss", round(best, 2), "parameters", sum(v.numel() for v in state.values()))


if __name__ == "__main__":
    assert set(FIELDS) >= {"cloud", "temp"}
    main()

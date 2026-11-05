"""Fine-tunes a network trained on ERA5 on Open-Meteo days, against 24-hour mean cloud.

usage: fine_tune.py NET.pt DEV.npz FIRST LAST SELECT_FIRST SELECT_LAST OUT.pt --lr 1e-4 --epochs 20 --pull 1 --part all
Keeps the epoch with the lowest loss on the select days. `pull` is a penalty on the squared distance from the starting
weights, since 45 days cannot pin down 5 million weights. `head` trains the learned places and the last decoder block.
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import torch

from features import Frames, Spec
from nets import UNet
from openmeteo_split import FIELD_NAMES
from train_net import weighted

HEAD = ("place", "up.2.", "out.")
BATCH = 8


def trainable(net: UNet, part: str) -> list:
    keep = [p for name, p in net.named_parameters() if part == "all" or name.startswith(HEAD)]
    wanted = {id(p) for p in keep}
    for p in net.parameters():
        p.requires_grad_(id(p) in wanted)
    return keep


def forecast(net: UNet, frames: Frames, days: np.ndarray) -> torch.Tensor:
    x, base = frames.batch(torch.from_numpy(days))
    return base + net(x) * frames.cloud_std


@torch.no_grad()
def loss_on(net: UNet, frames: Frames, truth: torch.Tensor, days: np.ndarray) -> float:
    net.eval()
    index = torch.from_numpy(days).to(truth.device)
    return float(weighted(forecast(net, frames, days).clamp(0, 100), truth[index]).mean())


def tune(net: UNet, frames: Frames, truth: torch.Tensor, fit_days: np.ndarray, select_days: np.ndarray, args) -> tuple:
    """Returns (best weights, best select loss, rows of epoch, fit loss, select loss)."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    params = trainable(net, args.part)
    start = [p.detach().clone() for p in params]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    best = loss_on(net, frames, truth, select_days)
    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    rows = [(0, loss_on(net, frames, truth, fit_days), best)]
    for epoch in range(1, args.epochs + 1):
        net.train()
        for part in np.array_split(rng.permutation(fit_days), max(1, len(fit_days) // BATCH)):
            want = truth[torch.from_numpy(part).to(truth.device)]
            away = sum(((p - s) ** 2).sum() for p, s in zip(params, start))
            loss = weighted(forecast(net, frames, part), want).mean() + args.pull * away
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        checked = loss_on(net, frames, truth, select_days)
        rows.append((epoch, loss_on(net, frames, truth, fit_days), checked))
        print(*(round(v, 2) for v in rows[-1]), flush=True)
        if checked < best:
            best, best_state = checked, {k: v.cpu().clone() for k, v in net.state_dict().items()}
    return best_state, best, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("net", "dev", "first", "last", "select_first", "select_last", "out"):
        parser.add_argument(name)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pull", type=float, default=1.0)
    parser.add_argument("--part", choices=("all", "head"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.net, map_location="cpu", weights_only=False)
    spec = Spec(tuple(saved["spec"]["variables"]), saved["spec"]["steps"], saved["spec"]["lags"])
    held = np.load(args.dev)
    first = date.fromisoformat(str(held["first_date"]))
    index = lambda name: (date.fromisoformat(name) - first).days
    frames = Frames({name: held[name] for name in FIELD_NAMES}, first, spec, saved["stats"], device)
    truth = torch.from_numpy(held["cloud_day"].astype(np.float32)).to(device)
    net = UNet(spec.channels, saved["width"], saved["place"]).to(device)
    net.load_state_dict(saved["state"])
    fit_days = np.arange(index(args.first), index(args.last) + 1)
    select_days = np.arange(index(args.select_first), index(args.select_last) + 1)
    state, best, rows = tune(net, frames, truth, fit_days, select_days, args)
    print("start", round(rows[0][2], 2), "best select loss", round(best, 2))
    torch.save({**saved, "state": state, "tune": vars(args)}, args.out)


if __name__ == "__main__":
    main()

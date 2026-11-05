"""Fits a linear model on the training years with the network's inputs and a wider stencil.

usage: fit_linear.py ERA5_DIR STATS.npz VAL.npz OUT.npz --vars cloud --steps 0 --lags 3 --radius 1 --per band
Each cell reads its neighbours within --radius cells. --per band shares weights across a latitude band as production
does, and --per cell gives every cell its own. The penalty is the one with the lowest validation loss.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

from common import load_year
from features import Frames, LATS, Spec
from nets import wrap
from train_net import FIRST_YEAR, LAST_TRAIN, weighted

PENALTIES = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
BANDS = (np.abs(LATS) < 30, (np.abs(LATS) >= 30) & (np.abs(LATS) < 60), np.abs(LATS) >= 60)


def stencil(x: torch.Tensor, radius: int) -> torch.Tensor:
    """(B, C, 24, 48) to (B, 1152, C * (2r+1)^2)."""
    wide = F.unfold(wrap(x, radius) if radius else x, 2 * radius + 1)
    return wide.transpose(1, 2)


def groups(per: str) -> np.ndarray:
    """The weight set of each of the 1152 cells."""
    if per == "cell":
        return np.arange(1152)
    band = np.zeros(24, dtype=int)
    for k, members in enumerate(BANDS):
        band[members] = k
    return np.repeat(band, 48)


def normal_equations(frames: Frames, days: np.ndarray, spec: Spec, radius: int, group: np.ndarray, size: int = 128) -> tuple:
    """Area-weighted X'X and X'y per group, from normalised departures from the base forecast."""
    count, device = int(group.max()) + 1, frames.device
    area = torch.from_numpy(np.repeat(np.cos(np.radians(LATS)), 48)).to(device)
    index = torch.from_numpy(group).to(device)
    features = spec.dynamic * (2 * radius + 1) ** 2
    xx = torch.zeros(count, features, features, dtype=torch.float64, device=device)
    xy = torch.zeros(count, features, dtype=torch.float64, device=device)
    for part in np.array_split(days, max(1, len(days) // size)):
        day = torch.from_numpy(part)
        inputs, base = frames.batch(day)
        x = stencil(inputs[:, :spec.dynamic], radius).double()
        y = ((frames.truth(day) - base) / frames.cloud_std).reshape(len(part), -1).double()
        for g in range(count):
            cells = index == g
            one = x[:, cells].reshape(-1, features)
            weight = area[cells].repeat(len(part))
            xx[g] += (one * weight[:, None]).T @ one
            xy[g] += (one * weight[:, None]).T @ y[:, cells].ravel()
    return xx, xy


def solve(xx: torch.Tensor, xy: torch.Tensor, penalty: float) -> torch.Tensor:
    # A penalty in units of the mean diagonal, so one grid fits any stencil and any group size.
    scale = xx.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] / 1e6
    return torch.linalg.solve(xx + penalty * scale * torch.eye(xx.shape[1], dtype=xx.dtype, device=xx.device), xy[..., None])[..., 0]


def apply(inputs: torch.Tensor, base: torch.Tensor, weights: torch.Tensor, group: torch.Tensor, spec: Spec, radius: int, std: float) -> torch.Tensor:
    x = stencil(inputs[:, :spec.dynamic], radius).to(weights.dtype)
    delta = (x * weights[group]).sum(-1).reshape(len(x), 24, 48)
    return base + delta.to(base.dtype) * std


@torch.no_grad()
def validation(frames: Frames, days: np.ndarray, weights: torch.Tensor, group: np.ndarray, spec: Spec, radius: int) -> float:
    index, total = torch.from_numpy(group).to(frames.device), 0.0
    for part in np.array_split(days, max(1, len(days) // 128)):
        day = torch.from_numpy(part)
        inputs, base = frames.batch(day)
        total += float(weighted(apply(inputs, base, weights, index, spec, radius, frames.cloud_std).clamp(0, 100), frames.truth(day)).sum())
    return total / len(days)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("era5", "stats", "val", "out"):
        parser.add_argument(name)
    parser.add_argument("--vars", default="cloud")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--per", default="band")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    spec = Spec(tuple(args.vars.split(",")), args.steps, args.lags)
    stats = dict(np.load(args.stats))
    names = set(spec.variables) | {"cloud", "temp"}
    fields = {n: np.concatenate([load_year(args.era5, n, y) for y in range(FIRST_YEAR, LAST_TRAIN + 1)]) for n in names}
    frames = Frames(fields, date(FIRST_YEAR, 1, 1), spec, stats, args.device)
    del fields
    held = np.load(args.val)
    start = date.fromisoformat(str(held["first_date"]))
    val = Frames({n: held[n] for n in names}, start, spec, stats, args.device)
    val_days = np.arange((date.fromisoformat(str(held["first_target"])) - start).days, len(val.daily["cloud"]))
    group = groups(args.per)
    xx, xy = normal_equations(frames, np.arange(frames.first_day(), len(frames.daily["cloud"])), spec, args.radius, group)
    best, best_weights, best_penalty = float("inf"), None, None
    for penalty in PENALTIES:
        weights = solve(xx, xy, penalty)
        found = validation(val, val_days, weights, group, spec, args.radius)
        print("penalty", penalty, "validation loss", round(found, 2), flush=True)
        if found < best:
            best, best_weights, best_penalty = found, weights, penalty
    keep = {k: v for k, v in stats.items() if k in ("land_sea_mask", "geopotential_at_surface") or k.split("_", 1)[-1] in names}
    np.savez(args.out, kind="linear", weights=best_weights.cpu().numpy().astype(np.float32), group=group, radius=args.radius,
             variables=np.array(spec.variables), steps=spec.steps, lags=spec.lags, penalty=best_penalty, **keep)
    print("best validation loss", round(best, 2), "penalty", best_penalty, asdict(spec))


if __name__ == "__main__":
    main()

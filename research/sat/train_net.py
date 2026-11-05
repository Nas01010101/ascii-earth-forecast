"""Deep residual network for the correction, trained on the score's own weighting.

usage: train_net.py FIT_PARQUET HOLDOUT_X OUT_DIR NAME [--depth N] [--width N] [--epochs N] [--seed N] [--no-history] [--no-weather] [--solar-ratio] [--fixed-dt] [--extra-shards N] [--max-drift KM] [--start-year Y]
Writes val_NAME.csv (5 group scores), oof_NAME.npy (validation corrections), hold_NAME.npy and NAME.pt.

The network does not output kilometres. It outputs a gain on the object's own drift estimate and a
fraction of an error scale for that horizon, so objects whose errors differ by orders of magnitude
share one network, and zero outputs mean raw SGP4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from build_samples import FEATURES, HISTORY_FEATURES, WEATHER_FEATURES
from data import HISTORY_TAIL, day, hide_history, inputs, load, load_folder, metric_weights, own_drift
from score import group_scores

TRAIN_SHARDS = set(range(16))
VAL_SHARDS = set(range(32))
VAL_START, END = day(2024), day(2025)
BATCH = 8192


def train_shards(extra: int) -> set:
    """Extra shards come from above the validation shards, so shards 16-31 stay unseen objects."""
    return TRAIN_SHARDS | set(range(max(VAL_SHARDS) + 1, min(max(VAL_SHARDS) + 1 + extra, 64)))


class Block(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class Net(nn.Module):
    def __init__(self, n_in: int, depth: int, width: int, dropout: float):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.first = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[Block(width, dropout) for _ in range(depth)])
        self.last = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))
        nn.init.zeros_(self.last[1].weight)
        nn.init.zeros_(self.last[1].bias)

    def forward(self, x, drift, scale):
        o = self.last(self.blocks(self.first((x - self.mean) / self.std)))
        along = o[:, 0] * drift + o[:, 1] * scale
        return torch.stack([o[:, 2] * scale * 0.1, along, o[:, 3] * scale * 0.1], dim=1)


def miss(radius, truth, c):
    def point(d):
        rho = radius + d[:, 0]
        u, w = d[:, 1] / radius, d[:, 2] / radius
        return torch.stack([rho * torch.cos(w) * torch.cos(u), rho * torch.cos(w) * torch.sin(u), rho * torch.sin(w)], dim=1)
    return torch.sqrt(((point(truth) - point(c)) ** 2).sum(1) + 1e-6)


def error_scale(drift: np.ndarray) -> np.ndarray:
    """Size of the along-track error the object's own history implies at dt, at least 1 km."""
    return np.maximum(np.abs(drift), 1.0).astype(np.float32)


def drop_columns(x: np.ndarray, names: list) -> np.ndarray:
    x = x.copy()
    x[:, [FEATURES.index(n) for n in names]] = 0.0
    return x


def tensors(d: dict, args) -> tuple:
    if args.fixed_dt:  # dt depends on when the truth elements arrive; the nominal horizon does not
        d = {**d, "dt": d["horizon"].astype(d["dt"].dtype)}
    x = inputs(d)
    if args.no_weather:
        x = drop_columns(x, WEATHER_FEATURES)
    elif args.solar_ratio:  # log flux relative to its 81-day mean: the raw level can leave the training range
        x[:, FEATURES.index("f107")] -= x[:, FEATURES.index("f107_81")]
        x[:, FEATURES.index("f107_81")] = 0.0
    if args.no_history:
        x = drop_columns(x, HISTORY_FEATURES)
        x[:, -HISTORY_TAIL:] = 0.0  # the drift estimate and the missing-hindcast flags come from history too
        drift = np.zeros(len(x), np.float32)
        scale = np.maximum(d["dt"].astype(np.float32) ** 2 * 0.1, 1.0)
    else:
        # a diverged hindcast implies drifts up to 1e13 km; one such row can set the mean training loss
        drift = np.clip(own_drift(d), -args.max_drift, args.max_drift)
        scale = error_scale(drift)
    return (torch.from_numpy(x), torch.from_numpy(drift), torch.from_numpy(scale))


def predict(net, tup, device) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tup[0]), 65536):
            out.append(net(*(t[i:i + 65536].to(device) for t in tup)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def fit(train: dict, args, device) -> Net:
    torch.manual_seed(args.seed)
    # kept on the device: copying each batch from host memory took most of an epoch
    tup = tuple(t.to(device) for t in tensors(train, args))
    y = torch.from_numpy(train["y"]).to(device)
    radius = torch.from_numpy(train["radius"]).to(device)
    w = torch.from_numpy(metric_weights(train)).to(device)
    net = Net(tup[0].shape[1], args.depth, args.width, args.dropout).to(device)
    net.mean.copy_(tup[0].mean(0))
    net.std.copy_(tup[0].std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(y) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(y), generator=gen)
        total = 0.0
        for b in range(len(y) // BATCH):
            i = order[b * BATCH:(b + 1) * BATCH].to(device)
            loss = (w[i] * miss(radius[i], y[i], net(*(t[i] for t in tup)))).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch} loss {total / max(1, len(y) // BATCH):.5f}", flush=True)
    return net


def train_rows(args, start: float, end: float, shards: set) -> dict:
    if args.train_samples:
        rows = load_folder(args.train_samples, start, end, shards, rate=args.rate, seed=args.seed)
    else:
        rows = load(args.fit, start, end, shards=shards, rate=args.rate, seed=args.seed)
    return hide_history(rows, args.history_dropout, args.seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fit"), p.add_argument("holdout"), p.add_argument("out"), p.add_argument("name")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--extra-shards", type=int, default=0)
    p.add_argument("--max-drift", type=float, default=float("inf"), help="cap on the drift estimate and scale, km")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--solar-ratio", action="store_true")
    p.add_argument("--fixed-dt", action="store_true")
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--train-samples", default="", help="folder of shard files to train on, in place of FIT_PARQUET; validation stays on FIT_PARQUET")
    p.add_argument("--history-dropout", type=float, default=0.0, help="share of training rows shown with few days of history")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = train_shards(args.extra_shards)
    start = day(args.start_year)
    train = train_rows(args, start, VAL_START, shards)
    val = load(args.fit, VAL_START, END, shards=VAL_SHARDS)
    print(f"train {len(train['dt'])} val {len(val['dt'])} rows on {device}", flush=True)
    net = fit(train, args, device)
    c = predict(net, tensors(val, args), device)
    scores = group_scores(val["norad"], val["horizon"], val["radius"], val["y"].astype(np.float64), c.astype(np.float64))
    np.save(out_dir / f"oof_{args.name}.npy", c)
    (out_dir / f"val_{args.name}.csv").write_text(",".join(f"{s:.4f}" for s in scores) + "\n")
    print(args.name, "val", np.round(scores, 2), "mean", round(float(scores.mean()), 2), flush=True)
    del train
    if args.skip_holdout:
        torch.save({"state": net.state_dict(), "args": vars(args)}, out_dir / f"{args.name}.pt")
        return

    refit = train_rows(args, start, END, shards)
    net = fit(refit, args, device)
    del refit, val
    hold = load(args.holdout, END, day(2026), labelled=False)
    np.save(out_dir / f"hold_{args.name}.npy", predict(net, tensors(hold, args), device))
    torch.save({"state": net.state_dict(), "args": vars(args)}, out_dir / f"{args.name}.pt")


if __name__ == "__main__":
    main()

"""Scores saved networks on the 2025 samples of a once-a-day download. Run only after the reveal.

usage: daily_check.py FIT_PARQUET DAILY_SAMPLES_DIR OUT_CSV NET.pt [NET.pt ...]
Each .pt was refit through 2024, so 2025 stays out of time. The mean of all nets is scored as "mean".
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from build_samples import HORIZONS
from catalogue_ci import HEADER, summary
from data import load_folder
from daily_samples import END, START
from head_to_head import month_blocks
from train_net import Net, predict, tensors


def load_net(path: str, device: str) -> tuple:
    saved = torch.load(path, map_location=device, weights_only=False)
    args = SimpleNamespace(**saved["args"])
    n_in = saved["state"]["mean"].shape[0]
    net = Net(n_in, args.depth, args.width, args.dropout).to(device)
    net.load_state_dict(saved["state"])
    return net, args


def main(fit: str, folder: str, out: str, paths: list) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit_ids = np.unique(pq.read_table(fit, columns=["norad"])["norad"].to_numpy())
    d = load_folder(folder, START, END)
    months = month_blocks(d["epoch"])
    y, radius = d["y"].astype(np.float64), d["radius"].astype(np.float64)
    corrections, rows = {}, []
    for path in paths:
        net, args = load_net(path, device)
        corrections[Path(path).stem] = predict(net, tensors(d, args), device).astype(np.float64)
    corrections["mean"] = np.mean(list(corrections.values()), axis=0)
    for name, c in corrections.items():
        for h in HORIZONS:
            m = d["horizon"] == h
            rows.append([name, h, *summary(d["norad"][m], months[m], radius[m], y[m], c[m], fit_ids)])
            print(*rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])

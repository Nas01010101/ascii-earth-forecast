"""Saves trained networks as the numpy weights the daily job loads.

usage: export_weights.py OUT.npz NET.pt [NET.pt ...]
"""
from __future__ import annotations

import sys

import numpy as np
import torch


def export(out: str, paths: list) -> None:
    arrays = {}
    for seed, path in enumerate(paths):
        state = torch.load(path, map_location="cpu", weights_only=False)["state"]
        arrays.update({f"{seed}/{name}": value.numpy().astype(np.float32) for name, value in state.items()})
    np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2:])

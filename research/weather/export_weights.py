"""Saves a trained network and its scales as the numpy file the daily job loads.

usage: export_weights.py OUT.npz NET.pt [float16|float32] [--layers]
Layer weights take the given type. Scales stay float32, because pressure near 1000 hPa needs the digits.
--layers leaves the scales out, for a network that reads them from the cloud file: they are the same numbers.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from features import LATS

VARIABLES = ("cloud", "temp", "precip", "pressure", "u", "v", "water")


def export(out: str, path: str, kind: str = "float16", layers: bool = False) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    arrays = {name: value.numpy().astype(kind) for name, value in saved["state"].items()}
    if layers:
        np.savez_compressed(out, **arrays)
        return
    stats = saved["stats"]
    for name in VARIABLES:
        arrays[f"clim_{name}"] = np.asarray(stats[f"clim_{name}"], dtype=np.float32)
        arrays[f"std_{name}"] = np.float32(stats[f"std_{name}"])
    lat = np.radians(LATS)[:, None] * np.ones((24, 48))
    arrays["fixed"] = np.stack([np.asarray(stats["land_sea_mask"]), np.asarray(stats["geopotential_at_surface"]), np.sin(lat), np.cos(lat)]).astype(np.float32)
    np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    flags = sys.argv[1:]
    only_layers = "--layers" in flags
    export(*[f for f in flags if f != "--layers"], layers=only_layers)

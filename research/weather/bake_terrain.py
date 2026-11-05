"""Bakes the height bands the site's weather map colours land with.

usage: bake_terrain.py OUT.txt
Reads surface geopotential from WeatherBench 2 and writes base64 for mapmath.js: 2 bits per 2.5 degree cell, south edge
first, west to east from 180W, the order planet.js keeps its land mask in. Sea is not told apart; the mask decides that.
"""
from __future__ import annotations

import base64
import sys

import numcodecs
import numpy as np

from fetch_era5 import NLAT, NLON, get

GRAVITY = 9.80665
ROWS, COLS, PER_CELL = 72, 144, 5
LIMITS = (300.0, 1000.0, 2500.0)


def cell_heights(height: np.ndarray) -> np.ndarray:
    """height (lon, lat) on the store's 1.5 degree grid, with poles, to the mean over each 2.5 degree cell."""
    inner = (np.arange(PER_CELL) + .5) / PER_CELL * 2.5
    lats = (np.arange(ROWS)[:, None] * 2.5 - 90 + inner[None]).ravel()
    lons = (np.arange(COLS)[:, None] * 2.5 + inner[None]).ravel() % 360 - 180
    y = (lats + 90) / 1.5
    x = (lons % 360) / 1.5
    y0, x0 = np.minimum(y.astype(int), NLAT - 2), x.astype(int) % NLON
    fy, fx = (y - y0)[None, :], (x - np.floor(x))[:, None]
    x1 = (x0 + 1) % NLON
    fine = ((1 - fx) * (1 - fy) * height[np.ix_(x0, y0)] + fx * (1 - fy) * height[np.ix_(x1, y0)]
            + (1 - fx) * fy * height[np.ix_(x0, y0 + 1)] + fx * fy * height[np.ix_(x1, y0 + 1)])
    # fine is (lon, lat); cells run south to north, then west to east.
    return fine.reshape(COLS, PER_CELL, ROWS, PER_CELL).mean((1, 3)).T


def pack(heights: np.ndarray) -> bytes:
    """(rows, cols) metres to 2 bits a cell, low bits first, as terrainBand reads them."""
    bands = np.digitize(heights.ravel(), LIMITS).astype(np.uint8)
    out = np.zeros((bands.size + 3) // 4, np.uint8)
    np.bitwise_or.at(out, np.arange(bands.size) >> 2, bands << (2 * (np.arange(bands.size) & 3)))
    return out.tobytes()


def main(out: str) -> None:
    raw = numcodecs.blosc.decompress(get("geopotential_at_surface/0.0"))
    height = np.frombuffer(raw, "<f4").reshape(NLON, NLAT) / GRAVITY
    cells = cell_heights(height)
    with open(out, "w") as file:
        file.write(base64.b64encode(pack(cells)).decode())
    places = {"Tibet": (32, 88), "Montreal": (45.5, -73.6), "Andes": (-16, -68), "Sahara": (23, 10), "Rockies": (40, -106)}
    for name, (lat, lon) in places.items():
        print(f"{name} {cells[int((lat + 90) / 2.5), int((lon + 180) / 2.5)]:.0f} m")
    print("max", cells.max(), "min", cells.min())


if __name__ == "__main__":
    main(sys.argv[1])

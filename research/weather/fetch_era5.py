"""Reads ERA5 from WeatherBench 2 and keeps the daily job's 48x24 points, 6-hourly.

usage: fetch_era5.py OUT_DIR MANIFEST.csv FIRST_YEAR LAST_YEAR
Writes OUT_DIR/{variable}_{year}.npy shaped (steps, 24, 48), south to north, west to east. The 1.5 degree store is read
because the job's Open-Meteo values are point samples, and bilinear weights over 1.5 degrees stay close to a point.
The store's last full year is 2022. A file that exists is kept, so a stopped run resumes.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numcodecs
import numpy as np

STORE = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
VARIABLES = ("total_cloud_cover", "2m_temperature", "total_precipitation_6hr", "mean_sea_level_pressure",
             "10m_u_component_of_wind", "10m_v_component_of_wind", "total_column_water_vapour")
STATIC = ("land_sea_mask", "geopotential_at_surface")
STEP = 7.5
LATS = np.arange(-90 + STEP / 2, 90, STEP)
LONS = np.arange(-180 + STEP / 2, 180, STEP)
CHUNK, NLON, NLAT = 8, 240, 121
START = datetime(1959, 1, 1)
THREADS = 8


def get(path: str, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{STORE}/{path}", timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def weights(latitude: np.ndarray, longitude: np.ndarray) -> tuple:
    """Bilinear corner indices and weights from the store's grid to the job's points."""
    lat_low = np.searchsorted(latitude, LATS) - 1
    lat_w = (LATS - latitude[lat_low]) / (latitude[lat_low + 1] - latitude[lat_low])
    wrapped = LONS % 360
    lon_low = np.floor(wrapped / 1.5).astype(int)
    lon_w = (wrapped - longitude[lon_low]) / 1.5
    return lat_low, lat_w, lon_low, lon_w


def to_points(field: np.ndarray, where: tuple) -> np.ndarray:
    """field (..., lon, lat) to (..., 24, 48)."""
    lat_low, lat_w, lon_low, lon_w = where
    west = field[..., lon_low, :]
    east = field[..., (lon_low + 1) % NLON, :]
    along = west * (1 - lon_w)[:, None] + east * lon_w[:, None]
    low, high = along[..., lat_low], along[..., lat_low + 1]
    out = low * (1 - lat_w) + high * lat_w
    return np.swapaxes(out, -1, -2).astype(np.float32)


def read_chunk(name: str, k: int, where: tuple) -> np.ndarray:
    raw = numcodecs.blosc.decompress(get(f"{name}/{k}.0.0"))
    return to_points(np.frombuffer(raw, "<f4").reshape(CHUNK, NLON, NLAT), where)


def year_steps(year: int) -> tuple:
    first = (datetime(year, 1, 1) - START).days * 4
    last = (datetime(year + 1, 1, 1) - START).days * 4
    return first, last


def read_year(name: str, year: int, where: tuple, pool: ThreadPoolExecutor) -> np.ndarray:
    first, last = year_steps(year)
    chunks = range(first // CHUNK, (last - 1) // CHUNK + 1)
    parts = list(pool.map(lambda k: read_chunk(name, k, where), chunks))
    joined = np.concatenate(parts)
    return joined[first - chunks[0] * CHUNK: last - chunks[0] * CHUNK]


def main(out: str, manifest: str, first_year: int, last_year: int) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads(get(".zmetadata"))["metadata"]
    for name in VARIABLES:
        assert meta[f"{name}/.zarray"]["chunks"] == [CHUNK, NLON, NLAT], name
        assert meta[f"{name}/.zarray"]["filters"] is None, name
    latitude = np.frombuffer(numcodecs.blosc.decompress(get("latitude/0")), meta["latitude/.zarray"]["dtype"]).astype(float)
    longitude = np.frombuffer(numcodecs.blosc.decompress(get("longitude/0")), meta["longitude/.zarray"]["dtype"]).astype(float)
    assert latitude[0] == -90 and latitude[-1] == 90 and longitude[0] == 0 and longitude[1] == 1.5
    where = weights(latitude, longitude)
    rows = []
    with ThreadPoolExecutor(THREADS) as pool:
        for name in STATIC:
            path = folder / f"{name}.npy"
            if not path.exists():
                raw = numcodecs.blosc.decompress(get(f"{name}/0.0"))
                np.save(path, to_points(np.frombuffer(raw, "<f4").reshape(NLON, NLAT), where))
            field = np.load(path)
            rows.append([name, "", field.shape[0], float(np.nanmean(field)), float(np.nanmin(field)), float(np.nanmax(field)), int(np.isnan(field).sum())])
        for year in range(first_year, last_year + 1):
            started = time.time()
            for name in VARIABLES:
                path = folder / f"{name}_{year}.npy"
                if not path.exists():
                    temp = path.with_suffix(".tmp.npy")
                    np.save(temp, read_year(name, year, where, pool))
                    temp.rename(path)
                field = np.load(path, mmap_mode="r")
                rows.append([name, year, field.shape[0], float(np.nanmean(field)), float(np.nanmin(field)), float(np.nanmax(field)), int(np.isnan(field).sum())])
            print(year, round(time.time() - started, 1), "s", flush=True)
    with open(manifest, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variable", "year", "steps", "mean", "min", "max", "nan"])
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

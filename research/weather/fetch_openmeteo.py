"""Downloads past Open-Meteo weather for the daily job's 1,152 points and keeps it in the ERA5 files' form.

usage: fetch_openmeteo.py fields OUT.npz FIRST LAST FIRST_TARGET CACHE_DIR
       fetch_openmeteo.py forecast OUT.npz FIRST LAST CACHE_DIR
       fetch_openmeteo.py lagged OUT.npz FIRST LAST CACHE_DIR
`fields` reads the historical forecast endpoint, whose hours matched the job's forecast endpoint on all 1,080 hours
compared at one point. `forecast` reads Open-Meteo's own forecast made a day earlier, for comparison only. `lagged`
adds the one made two days earlier, as inputs for a correction of those forecasts. Each chunk is cached, so a stopped
download resumes.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from forecast import fetch  # noqa: E402

FIELDS_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HOURLY = {"cloud": "cloud_cover", "temp": "temperature_2m", "precip": "precipitation", "pressure": "pressure_msl",
          "speed": "wind_speed_10m", "direction": "wind_direction_10m", "water": "total_column_integrated_water_vapour"}
LAST_RUN = {"cloud": "cloud_cover_previous_day1", "temp": "temperature_2m_previous_day1"}
LAGGED = {**LAST_RUN, "cloud2": "cloud_cover_previous_day2"}
CHUNK = 100
HOURS_PER_STEP = 6
# The free tier allows 5,000 calls an hour and bills a location 1 call per 14 days of data.
CALLS_PER_MINUTE = 80


def query(url: str, points: np.ndarray, names: dict, first: date, last: date) -> str:
    return url + "?" + urllib.parse.urlencode({
        "latitude": ",".join(f"{v:g}" for v in points[:, 0]),
        "longitude": ",".join(f"{v:g}" for v in points[:, 1]),
        "hourly": ",".join(names.values()),
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    })


def to_grid(payloads: list, names: dict, first: date, last: date) -> dict:
    """{name: (hours, 24, 48)} from one JSON location per cell. A missing hour or location stops the run."""
    hours = ((last - first).days + 1) * 24
    shape = (hours, len(fetch.LATS), len(fetch.LONS))
    if len(payloads) != shape[1] * shape[2]:
        raise fetch.SourceError(f"got {len(payloads)} locations")
    found = {}
    for name, api_name in names.items():
        grid = np.array([p["hourly"][api_name] for p in payloads], dtype=np.float32)
        if grid.shape[1] != hours or not np.isfinite(grid).all():
            raise fetch.SourceError(f"{name}: shape {grid.shape} or missing values")
        found[name] = grid.T.reshape(shape)
    return found


def download(url: str, names: dict, first: date, last: date, cache: str, pause=time.sleep) -> dict:
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    points = fetch.cells()
    wait = 60 * CHUNK * max(1, ((last - first).days + 1) / 14) / CALLS_PER_MINUTE
    payloads = []
    for start in range(0, len(points), CHUNK):
        path = folder / f"{start}.json"
        if not path.exists():
            path.write_text(json.dumps(fetch.get_json(query(url, points[start:start + CHUNK], names, first, last))))
            pause(wait)
        payloads += json.loads(path.read_text())
    return to_grid(payloads, names, first, last)


def six_hourly(hourly: dict) -> dict:
    """ERA5's seven fields from hours that begin a day before the first kept step.

    ERA5 precipitation is the 6 hours ending at each step and an Open-Meteo hour holds the hour ending at its stamp.
    """
    picked = {n: hourly[n][24::HOURS_PER_STEP] for n in ("cloud", "temp", "pressure", "water")}
    total = np.concatenate([np.zeros_like(hourly["precip"][:1]), np.cumsum(hourly["precip"], axis=0, dtype=np.float64)])
    ends = np.arange(24, len(hourly["precip"]), HOURS_PER_STEP)
    picked["precip"] = (total[ends + 1] - total[ends + 1 - HOURS_PER_STEP]).astype(np.float32)
    speed = hourly["speed"][24::HOURS_PER_STEP]
    toward = np.radians(hourly["direction"][24::HOURS_PER_STEP])
    # Direction names where the wind comes from, and u and v name where it goes.
    picked["u"], picked["v"] = -speed * np.sin(toward), -speed * np.cos(toward)
    return picked


def day_means(field: np.ndarray) -> np.ndarray:
    return field.reshape(-1, 24, *field.shape[1:]).mean(axis=1)


def fields(out: str, first: str, last: str, first_target: str, cache: str) -> None:
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    hourly = download(FIELDS_URL, HOURLY, start - timedelta(days=1), end, cache)
    np.savez(out, first_date=first, first_target=first_target, **six_hourly(hourly),
             cloud_day=day_means(hourly["cloud"][24:]), temp_day=day_means(hourly["temp"][24:]))


def forecast(out: str, first: str, last: str, cache: str) -> None:
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    hourly = download(FORECAST_URL, LAST_RUN, start, end, cache)
    np.savez(out, first_date=first, cloud_day=day_means(hourly["cloud"]), temp_day=day_means(hourly["temp"]))


def lagged(out: str, first: str, last: str, cache: str) -> None:
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    hourly = download(FORECAST_URL, LAGGED, start, end, cache)
    np.savez(out, first_date=first, **{f"{name}_day": day_means(field) for name, field in hourly.items()})


if __name__ == "__main__":
    {"fields": fields, "forecast": forecast, "lagged": lagged}[sys.argv[1]](*sys.argv[2:])

"""Downloads and validates the two sources. Nothing here writes to disk."""
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

STEP = 7.5
LATS = np.arange(-90 + STEP / 2, 90, STEP)
LONS = np.arange(-180 + STEP / 2, 180, STEP)
VARIABLES = {"cloud": "cloud_cover", "temp": "temperature_2m", "precip": "precipitation", "pressure": "pressure_msl",
             "speed": "wind_speed_10m", "direction": "wind_direction_10m", "water": "total_column_integrated_water_vapour"}
LIMITS = {"cloud": (0, 100), "temp": (-95, 65), "precip": (0, 500), "pressure": (800, 1150),
          "speed": (0, 200), "direction": (0, 360), "water": (0, 200)}
GP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
CHUNK = 100
# Open-Meteo's free tier allows 600 calls a minute and 5,000 an hour, and bills each location
# 1 call per 14 days of data. 45 past days plus 2 ahead over the 1,152 cells is about 3,900 calls.
CALLS_PER_MINUTE = 500
BOOTSTRAP_DAYS = 45
FORECAST_DAYS = 2
MIN_SATELLITES = 5000


class SourceError(RuntimeError):
    """A source was down or returned data that failed validation."""


def get(url, tries=4, wait=20):
    """Returns the body of url, retrying on rate limits, server errors and network failures."""
    request = urllib.request.Request(url, headers={"User-Agent": "ascii-earth-forecast"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code != 429 and err.code < 500:
                raise SourceError(f"{url[:60]}: HTTP {err.code}") from err
            failure = err
        except (OSError, http.client.HTTPException) as err:
            failure = err
        if attempt < tries - 1:
            time.sleep(wait * 2**attempt)
    raise SourceError(f"{url[:60]}: {failure}")


def cells():
    """Every grid cell as (latitude, longitude) rows, south to north and west to east."""
    lat, lon = np.meshgrid(LATS, LONS, indexing="ij")
    return np.column_stack([lat.ravel(), lon.ravel()])


def weather_url(points, past_days):
    query = {
        "latitude": ",".join(f"{v:g}" for v in points[:, 0]),
        "longitude": ",".join(f"{v:g}" for v in points[:, 1]),
        "hourly": ",".join(VARIABLES.values()),
        "wind_speed_unit": "ms",
        "past_days": past_days,
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }
    return f"{WEATHER_URL}?{urllib.parse.urlencode(query)}"


def parse_weather(payloads, shape):
    """Turns one JSON location list into hours and {name: (hours, lat, lon)} float32 arrays."""
    if len(payloads) != shape[0] * shape[1]:
        raise SourceError(f"weather: got {len(payloads)} locations")
    try:
        stamps = payloads[0]["hourly"]["time"]
        # A run crossing 00:00 UTC can return chunks whose hours are a day apart.
        if any(p["hourly"]["time"] != stamps for p in payloads):
            raise SourceError("weather: locations disagree on the hours")
        hours = np.array(stamps, dtype="datetime64[h]")
        fields = {}
        for name, api_name in VARIABLES.items():
            rows = [p["hourly"][api_name] for p in payloads]
            # float32 turns the API's nulls into NaN; ragged rows raise ValueError here.
            grid = np.array(rows, dtype=np.float32).T.reshape(len(hours), *shape)
            low, high = LIMITS[name]
            if np.any((grid < low) | (grid > high)):
                raise SourceError(f"weather: {name} outside {low}..{high}")
            fields[name] = grid
    except (KeyError, TypeError, ValueError) as err:
        raise SourceError(f"weather: malformed response: {err}") from err
    return hours, fields


def weather(past_days, pause=time.sleep):
    """Downloads the whole grid in chunks, paced under the free tier's limit."""
    points = cells()
    calls = CHUNK * max(1, (past_days + FORECAST_DAYS) / 14)
    payloads = []
    for start in range(0, len(points), CHUNK):
        if start:
            pause(60 * calls / CALLS_PER_MINUTE)
        try:
            body = json.loads(get(weather_url(points[start : start + CHUNK], past_days)))
        except ValueError as err:
            raise SourceError(f"weather: invalid JSON: {err}") from err
        payloads += body if isinstance(body, list) else [body]
    return parse_weather(payloads, (len(LATS), len(LONS)))


def elements():
    """Downloads the active satellite list as CSV text."""
    # CelesTrak firewalls addresses that keep requesting after a non-200, so this never retries.
    text = get(GP_URL, tries=1).decode()
    # CelesTrak answers repeat downloads with a plain-text notice rather than data.
    if not text.startswith("OBJECT_NAME,") or text.count("\n") < MIN_SATELLITES:
        raise SourceError(f"elements: unexpected body: {text[:80]!r}")
    return text

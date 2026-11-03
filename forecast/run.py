"""One daily update. Reads and writes a data directory; the workflow publishes it on the data branch.

Each source fails on its own: the last good section stays in forecast.json, flagged stale, and
the run only exits non-zero when neither source could update.
"""
import gzip
import json
import os
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import fetch, orbits, weather

HISTORY_DAYS = 60
SCORED_DAYS = 7
SNAPSHOT_DAYS = 9
MAX_MISSING = 0.01
MIN_HOURS = 18
FRAME_HOURS = 3
RAMP = " .:-=+*#%@"
ERROR_RAMP_MAX = 50


def write_atomic(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def load_history(root):
    path = root / "history.npz"
    if not path.exists():
        return {}
    # NpzFile decompresses on every access, so each array is read once.
    with np.load(path) as saved:
        days, cloud, temp = saved["days"], saved["cloud"], saved["temp"]
    return {str(d): (cloud[i], temp[i]) for i, d in enumerate(days)}


def save_history(root, history):
    days = sorted(history)
    temp = root / "history.tmp.npz"
    np.savez_compressed(temp, days=np.array(days),
                        cloud=np.stack([history[d][0] for d in days]),
                        temp=np.stack([history[d][1] for d in days]))
    os.replace(temp, root / "history.npz")


def add_days(history, hours, fields, today):
    """Adds each complete new UTC day before today as daily means, then trims to the rolling window.

    Stored days are never rewritten, so a day used to score another keeps its value. A day with too
    many missing cells is left out; days_to_fetch asks for it again on the next run.
    """
    stamps = hours.astype("datetime64[D]")
    for day in np.unique(stamps):
        pick = stamps == day
        if day >= np.datetime64(today) or pick.sum() != 24 or str(day) in history:
            continue
        cloud, temp = _daily_mean(fields["cloud"][pick]), _daily_mean(fields["temp"][pick])
        if max(np.isnan(cloud).mean(), np.isnan(temp).mean()) > MAX_MISSING:
            print(f"::warning::weather: over {MAX_MISSING:.0%} of cells missing on {day}", file=sys.stderr)
            continue
        history[str(day)] = (_fill(cloud), _fill(temp))
    for old in sorted(history)[:-HISTORY_DAYS]:
        del history[old]


def _daily_mean(hours):
    """Mean over the hours present; a cell with under MIN_HOURS of them is missing."""
    present = ~np.isnan(hours)
    count = present.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(present, hours, 0).sum(0) / count
    mean[count < MIN_HOURS] = np.nan
    return mean


def days_to_fetch(history, today):
    """How many past days to request: back to the oldest one missing, so a hole heals on the next run."""
    for back in range(fetch.BOOTSTRAP_DAYS, 0, -1):
        if (today - timedelta(days=back)).isoformat() not in history:
            return back
    return 0


def _fill(field):
    """A rare missing cell takes the mean of its latitude row, or the global mean if the row is empty."""
    if not np.isnan(field).any():
        return field
    field = field.copy()
    with np.errstate(invalid="ignore"):
        rows = np.array([np.nanmean(r) if not np.isnan(r).all() else np.nanmean(field) for r in field])
    gaps = np.isnan(field)
    field[gaps] = np.broadcast_to(rows[:, None], field.shape)[gaps]
    return field


def consecutive(history, today):
    """The unbroken run of days ending yesterday, as sorted date strings."""
    days, day = [], today - timedelta(days=1)
    while day.isoformat() in history:
        days.insert(0, day.isoformat())
        day -= timedelta(days=1)
    return days


def picture(field, top):
    """Draws a grid with north on top, one character per cell, darkest at zero."""
    index = np.clip(np.nan_to_num(field) / top * (len(RAMP) - 1), 0, len(RAMP) - 1).round().astype(int)
    return ["".join(RAMP[i] for i in row) for row in index[::-1]]


def weather_section(days, history, today):
    if len(days) <= weather.MIN_TRAIN_DAYS + 3:
        raise fetch.SourceError(f"weather: {len(days)} consecutive days is too few to score")
    cloud = np.stack([history[d][0] for d in days])
    temp = np.stack([history[d][1] for d in days])
    series = {"cloud": cloud, "temp": temp}
    scored = []
    keep = {}
    first = max(len(days) - SCORED_DAYS, weather.MIN_TRAIN_DAYS + 3)
    for target in range(first, len(days)):
        row = {"date": days[target]}
        for name, values in series.items():
            forecast, trailing = weather.predict(values, target)
            row[name] = weather.score(values[target], forecast, values[target - 1], trailing)
            keep[name] = forecast, values[target]
        scored.append(row)
    today_cloud, _ = weather.predict(cloud, len(days))
    today_temp, _ = weather.predict(temp, len(days))
    forecast, actual = keep["cloud"]
    week = {name: {k: round(float(np.mean([r[name][k] for r in scored])), 2) for k in scored[0][name]} for name in series}
    return {
        "days": scored,
        "week": week,
        "train_days": len(days),
        "features": list(weather.FEATURES),
        "today": {
            "date": today.isoformat(),
            "cloud_mean": round(_area_mean(np.clip(today_cloud, 0, 100)), 1),
            "temp_mean": round(_area_mean(today_temp), 1),
            "map": picture(np.clip(today_cloud, 0, 100), 100),
        },
        "last_scored": {
            "date": scored[-1]["date"],
            "forecast": picture(np.clip(forecast, 0, 100), 100),
            "actual": picture(actual, 100),
            "error": picture(np.abs(np.clip(forecast, 0, 100) - actual), ERROR_RAMP_MAX),
        },
    }


def _area_mean(field):
    return float((field * weather.AREA).sum() / (weather.AREA.sum() * field.shape[1]))


def planet_frames(hours, fields, today):
    """Packs today's and tomorrow's cloud cover (percent) and rain (tenths of a mm/h) for the planet."""
    pick = np.flatnonzero((hours >= np.datetime64(today)) & (hours.astype(int) % FRAME_HOURS == 0))
    if not pick.size:
        raise fetch.SourceError("weather: no hours for today")
    for name in ("cloud", "precip"):
        if np.isnan(fields[name][pick]).mean(axis=(1, 2)).max() > MAX_MISSING:
            raise fetch.SourceError(f"weather: over {MAX_MISSING:.0%} of {name} missing in a future frame")
    start = datetime.combine(today, datetime.min.time(), timezone.utc).timestamp() * 1000
    cloud = np.nan_to_num(fields["cloud"][pick]).clip(0, 100).round().astype(np.uint8)
    rain = (np.nan_to_num(fields["precip"][pick]) * 10).clip(0, 255).round().astype(np.uint8)
    header = struct.pack("<dHHHH", start, pick.size, FRAME_HOURS, *cloud.shape[1:])
    return header + cloud.tobytes() + rain.tobytes()


def update_weather(root, today):
    history = load_history(root)
    hours, fields = fetch.weather(days_to_fetch(history, today))
    # The planet needs only today's forecast, so a problem with past days cannot hold it back.
    write_atomic(root / "planet.bin", planet_frames(hours, fields, today))
    add_days(history, hours, fields, today)
    save_history(root, history)
    return weather_section(consecutive(history, today), history, today)


def _snapshot(root, day):
    path = root / "elements" / f"{day.isoformat()}.csv.gz"
    return orbits.parse(gzip.decompress(path.read_bytes()).decode()) if path.exists() else None


def horizon_row(old, new, before_old, before_new, days):
    """Scores one horizon. The drift fit only sees the pair ending yesterday, never today's elements."""
    pair = orbits.compare(old, new)
    if pair is None or not pair["id"].size:
        return None
    row = {"days": days, **orbits.regimes(pair)}
    iss = np.flatnonzero(pair["id"] == orbits.ISS)
    if iss.size:
        row["iss_km"] = round(float(pair["km"][iss[0]]), 2)
    train = orbits.compare(before_old, before_new) if before_old and before_new else None
    weights = orbits.fit_drift(train) if train else None
    if weights is not None and (pair["rev"] >= orbits.LEO_REV_PER_DAY).any():
        row["low_corrected"] = orbits.summary(orbits.corrected_km(pair, weights))
    return row


def update_satellites(root, today):
    folder = root / "elements"
    folder.mkdir(exist_ok=True)
    path = folder / f"{today.isoformat()}.csv.gz"
    # CelesTrak rejects repeat downloads of unchanged data, so a same-day rerun reuses the snapshot.
    if not path.exists():
        text = orbits.clean(fetch.elements())
        write_atomic(root / "sats.bin", orbits.pack(text))
        # Last, because a snapshot tells a rerun that today is done.
        write_atomic(path, gzip.compress(text.encode(), mtime=0))
    for old in sorted(folder.glob("*.csv.gz"))[:-SNAPSHOT_DAYS]:
        old.unlink()
    snaps = {d: _snapshot(root, today - timedelta(days=d)) for d in range(0, SCORED_DAYS + 2)}
    rows = []
    for days in range(1, SCORED_DAYS + 1):
        if snaps[days]:
            row = horizon_row(snaps[days], snaps[0], snaps.get(days + 1), snaps[1], days)
            if row:
                rows.append(row)
    return {"as_of": today.isoformat(), "count": len(snaps[0]), "horizons": rows}


def run(root, today, updates):
    """Runs each named update, keeping the previous section when a source fails. Returns failure count."""
    path = root / "forecast.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    failures = 0
    for name, update in updates.items():
        try:
            section = update(root, today)
            # allow_nan=False makes a NaN in a score fail its source instead of breaking the page's JSON.parse.
            json.dumps(section, allow_nan=False)
            report[name] = section
            report.setdefault("status", {})[name] = {"ok": True, "updated": today.isoformat()}
        # Any crash, expected or not, must leave the other source and the last good section alone.
        except Exception as err:
            failures += 1
            print(f"::warning::{name}: {err}", file=sys.stderr)
            status = report.setdefault("status", {}).setdefault(name, {"updated": None})
            status.update(ok=False, error=f"{type(err).__name__}: {err}"[:200])
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_atomic(path, json.dumps(report, allow_nan=False, separators=(",", ":")).encode())
    return failures


def main(argv=None):
    root = Path((argv or sys.argv)[1])
    root.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    updates = {"weather": update_weather, "satellites": update_satellites}
    return 1 if run(root, today, updates) == len(updates) else 0

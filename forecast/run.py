"""One daily update. Reads and writes a data directory; the workflow publishes it on the data branch.

Each source fails on its own: the last good section stays in forecast.json, flagged stale, and
the run only exits non-zero when neither source could update.
"""
import gzip
import json
import os
import struct
import sys
from collections import namedtuple
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cloudnet, drift, fetch, orbits, weather

HISTORY_DAYS = 60
LOSS_DAYS = 28
STEP_DAYS = cloudnet.DAYS_NEEDED + 8  # the network's window, and a week of replays after a missed run
SNAPSHOT_DAYS = 38  # a 7-day forecast, its 30-day update count, and the day it is checked
MAX_MISSING = 0.01
MIN_HOURS = 18
FRAME_HOURS = 3
RAMP = " .:-=+*#%@"
CLOUD_RANGE = (0, 100)
# net: the network made the cloud. temp_net: the network made the temperature. pressure: None when no network made one.
Forecast = namedtuple("Forecast", "cloud temp issued net pressure temp_net", defaults=(None, False))


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


def load_forecasts(root):
    path = root / "forecasts.npz"
    if not path.exists():
        return {}
    with np.load(path) as saved:
        days, cloud, temp, issued = saved["days"], saved["cloud"], saved["temp"], saved["issued"]
        # Forecasts stored before the network have no flag, and all came from the ridge.
        net = saved["net"] if "net" in saved.files else np.zeros(len(days), dtype=bool)
        # Forecasts stored before the temperature and pressure networks have neither.
        temp_net = saved["temp_net"] if "temp_net" in saved.files else np.zeros(len(days), dtype=bool)
        pressure = saved["pressure"] if "pressure" in saved.files else np.full((len(days), 1, 1), np.nan)
    return {str(d): Forecast(cloud[i], temp[i], bool(issued[i]), bool(net[i]),
                             None if np.isnan(pressure[i]).all() else pressure[i], bool(temp_net[i]))
            for i, d in enumerate(days)}


def save_forecasts(root, forecasts):
    days = sorted(forecasts)
    temp = root / "forecasts.tmp.npz"
    made = [Forecast(*forecasts[d]) for d in days]
    empty = np.full(np.shape(made[0].cloud), np.nan)
    np.savez_compressed(temp, days=np.array(days),
                        cloud=np.stack([forecasts[d][0] for d in days]).astype(np.float32),
                        temp=np.stack([forecasts[d][1] for d in days]).astype(np.float32),
                        issued=np.array([forecasts[d][2] for d in days]),
                        net=np.array([forecasts[d][3] for d in days]),
                        temp_net=np.array([f.temp_net for f in made]),
                        pressure=np.stack([empty if f.pressure is None else f.pressure for f in made]).astype(np.float32))
    os.replace(temp, root / "forecasts.npz")


def load_steps(root):
    path = root / "steps.npz"
    if not path.exists():
        return {}
    with np.load(path) as saved:
        days, values = saved["days"], saved["values"]
    return {str(d): values[i] for i, d in enumerate(days)}


def save_steps(root, steps):
    days = sorted(steps)
    if not days:
        return
    temp = root / "steps.tmp.npz"
    np.savez_compressed(temp, days=np.array(days), values=np.stack([steps[d] for d in days]))
    os.replace(temp, root / "steps.npz")


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


def add_steps(steps, hours, fields, today):
    """Adds each past day the network can read that is not held yet, then trims to its window.

    Same rules as add_days: stored days are never rewritten, and a day with too many missing cells is left out
    and asked for again on the next run.
    """
    for back in range(STEP_DAYS, 0, -1):
        day = today - timedelta(days=back)
        if day.isoformat() in steps:
            continue
        found = cloudnet.day_steps(hours, fields, day)
        if found is None:
            continue
        if np.isnan(found).mean(axis=(2, 3)).max() > MAX_MISSING:
            print(f"::warning::weather: over {MAX_MISSING:.0%} of a 6-hourly field missing on {day}", file=sys.stderr)
            continue
        steps[day.isoformat()] = np.stack([[_fill(field) for field in variable] for variable in found])
    for old in sorted(steps)[:-STEP_DAYS]:
        del steps[old]


def _daily_mean(hours):
    """Mean over the hours present; a cell with under MIN_HOURS of them is missing."""
    present = ~np.isnan(hours)
    count = present.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(present, hours, 0).sum(0) / count
    mean[count < MIN_HOURS] = np.nan
    return mean


def days_to_fetch(history, steps, today):
    """How many past days to request: back to the oldest one missing, so a hole heals on the next run."""
    def oldest(held, window):
        return max((b for b in range(1, window + 1) if (today - timedelta(days=b)).isoformat() not in held), default=0)
    # A day's first 6-hourly rain reads the last hours of the day before it.
    missing_step = oldest(steps, STEP_DAYS)
    return max(oldest(history, fetch.BOOTSTRAP_DAYS), missing_step + 1 if missing_step else 0)


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


def _forecast(cloud, temp, target, when, steps, saved, others=None):
    """Forecasts a day from the days before it, as (cloud, temp, cloud by network, pressure or None, temp by network).

    The networks need all 30 days of 6-hourly steps. When any is missing cloud and temperature come from the ridge, so
    a forecast is still kept, the flags say which model made it, and there is no pressure. `others` holds the networks
    of the other targets, and a target without one takes the same fallback. Cover outside 0..100 cannot happen.
    """
    others = others or {}
    held = cloudnet.view(steps, when)
    if held is None:
        print(f"::warning::weather: 6-hourly days missing before {when}, ridge forecasts only", file=sys.stderr)
        found = weather.predict(cloud, target)[0]
    else:
        found = cloudnet.predict(held, when, saved)
    by_net = {name: cloudnet.predict(held, when, net, name) for name, net in others.items()} if held is not None else {}
    made_temp = by_net["temp"] if "temp" in by_net else weather.predict(temp, target)[0]
    return np.clip(found, *CLOUD_RANGE), made_temp, held is not None, by_net.get("pressure"), "temp" in by_net


def keep_forecasts(forecasts, days, history, steps, today, saved, others=None):
    """Stores today's forecast as issued and replays the recent days that have none. No entry is rewritten.

    A replay runs the same code on the stored days before its target, and stays flagged as not issued.
    """
    cloud = np.stack([history[d][0] for d in days])
    temp = np.stack([history[d][1] for d in days])
    if today.isoformat() not in forecasts:
        made = _forecast(cloud, temp, len(days), today, steps, saved, others)
        forecasts[today.isoformat()] = Forecast(made[0], made[1], True, made[2], made[3], made[4])
    for target in range(max(len(days) - LOSS_DAYS, weather.MIN_TRAIN_DAYS + 3), len(days)):
        if days[target] not in forecasts:
            made = _forecast(cloud, temp, target, date.fromisoformat(days[target]), steps, saved, others)
            forecasts[days[target]] = Forecast(made[0], made[1], False, made[2], made[3], made[4])


def _pressure_of_day(steps, day):
    """A day's pressure as the mean of its 4 samples, the truth the network was scored on, or None if not held."""
    return steps[day][cloudnet.VARIABLES.index("pressure")].mean(0) if day in steps else None


def forecast_losses(forecasts, history, steps=None):
    """Area-weighted squared error of each stored forecast whose day has finished, and of repeating the day before."""
    steps = steps or {}
    rows = []
    for day in sorted(set(forecasts) & set(history))[-LOSS_DAYS:]:
        before = history.get((date.fromisoformat(day) - timedelta(days=1)).isoformat())
        row = {"date": day, "issued": forecasts[day][2], "net": forecasts[day][3]}
        for i, name in enumerate(("cloud", "temp")):
            row[name] = {"model": round(weather.mse(history[day][i], forecasts[day][i]), 1),
                         "repeat": round(weather.mse(history[day][i], before[i]), 1) if before else None}
        made = Forecast(*forecasts[day])
        truth = _pressure_of_day(steps, day)
        if made.pressure is not None and truth is not None:
            earlier = _pressure_of_day(steps, (date.fromisoformat(day) - timedelta(days=1)).isoformat())
            row["pressure"] = {"model": round(weather.mse(truth, made.pressure), 1),
                               "repeat": round(weather.mse(truth, earlier), 1) if earlier is not None else None}
        rows.append(row)
    return rows


def weather_section(days, history, forecasts, today):
    made = Forecast(*forecasts[today.isoformat()])
    cloud, temp, net = made.cloud, made.temp, made.net
    networks = [name for name, by_net in (("cloud", net), ("temp", made.temp_net), ("pressure", made.pressure is not None)) if by_net]
    section = {
        "train_days": len(days),
        "today": {
            "date": today.isoformat(),
            "net": net,
            "places": weather.at_places(cloud, temp),
            "map": picture(cloud, 100),
            # Whole percent and whole degrees, south to north and west to east, for the viewer's map.
            "cloud": np.clip(cloud, *CLOUD_RANGE).round().astype(int).ravel().tolist(),
            "temp": np.asarray(temp).round().astype(int).ravel().tolist(),
            "networks": networks,
        },
        "last_scored": {"date": days[-1], "actual": picture(history[days[-1]][0], 100)},
    }
    if made.pressure is not None:
        # A tenth of a hPa keeps the viewer's isobars from stepping.
        section["today"]["pressure"] = np.asarray(made.pressure, dtype=np.float64).round(1).ravel().tolist()
    return section


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
    history, steps = load_history(root), load_steps(root)
    hours, fields = fetch.weather(days_to_fetch(history, steps, today))
    # The planet needs only today's forecast, so a problem with past days cannot hold it back.
    write_atomic(root / "planet.bin", planet_frames(hours, fields, today))
    add_days(history, hours, fields, today)
    save_history(root, history)
    add_steps(steps, hours, fields, today)
    save_steps(root, steps)
    days = consecutive(history, today)
    if len(days) <= weather.MIN_TRAIN_DAYS + 3:
        raise fetch.SourceError(f"weather: {len(days)} consecutive days is too few to forecast")
    forecasts = load_forecasts(root)
    saved = cloudnet.load()
    keep_forecasts(forecasts, days, history, steps, today, saved, cloudnet.load_others(saved))
    save_forecasts(root, forecasts)
    return {**weather_section(days, history, forecasts, today), "losses": forecast_losses(forecasts, history, steps)}


def stored_losses(root):
    path = root / "forecast.json"
    return json.loads(path.read_text()).get("satellites", {}).get("losses", []) if path.exists() else []


def satellite_losses(rows, snapshots, today):
    """Adds a row for each held day that has none, keeping the last LOSS_DAYS. No row is rewritten.

    A row replayed from stored snapshots reads only what was held on its day, and stays flagged as not issued.
    """
    done = {row["date"] for row in rows}
    nets = drift.load_nets()
    rows = list(rows)
    for day in sorted(snapshots):
        if day.isoformat() not in done:
            days = drift.score_day(snapshots, day, nets)
            if days:
                rows.append({"date": day.isoformat(), "issued": day == today, "days": days})
    return sorted(rows, key=lambda row: row["date"])[-LOSS_DAYS:]


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
    snapshots = {date.fromisoformat(p.name.split(".")[0]): orbits.parse(gzip.decompress(p.read_bytes()).decode())
                 for p in folder.glob("*.csv.gz")}
    # After the snapshot, so a rerun that finds the snapshot still writes the viewer's file.
    write_atomic(root / "corrections.bin", drift.pack_corrections(snapshots, today, drift.load_nets()))
    return {"as_of": today.isoformat(), "count": len(snapshots[today]),
            "losses": satellite_losses(stored_losses(root), snapshots, today)}


def done_today(report, name, today):
    status = report.get("status", {}).get(name, {})
    return status.get("ok") is True and status.get("updated") == today.isoformat()


def run(root, today, updates):
    """Runs each named update not yet done today, keeping the previous section when a source fails. Returns failure count.

    The workflow runs twice a day because GitHub can drop a scheduled run. The later run fetches only what is missing.
    """
    path = root / "forecast.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    failures = 0
    for name, update in updates.items():
        if done_today(report, name, today):
            print(f"{name}: done today", file=sys.stderr)
            continue
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

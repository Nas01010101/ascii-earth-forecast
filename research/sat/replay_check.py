"""Runs the daily job's scorer on 2025 elements rebuilt as daily snapshots, and reports what the page would show.

usage: replay_check.py SHARDS_DIR WEIGHTS.npz OUT.csv NET.pt SHARD [SHARD ...]
Rows are one target day per week of the burned year, and nothing here feeds a choice of model. The first net is
also compared with the torch file it was exported from, on the rows of the first target day.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# The forecast package sits two levels up in the repo and one level up in the copy on the GPU box.
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from forecast import drift  # noqa: E402

from daily_samples import SNAPSHOT_HOUR  # noqa: E402
from sat_common import COLUMNS, SGP4_EPOCH_JD  # noqa: E402
from train_net import Net, predict  # noqa: E402

WINDOW_DAYS = 38
FIRST, LAST = date(2025, 2, 10), date(2025, 12, 29)


def epoch_of(day: date) -> float:
    """SGP4 epoch days at the hour the daily job downloads on this calendar day."""
    moment = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() / 86400
    return moment + drift.JD_UNIX_EPOCH - SGP4_EPOCH_JD + SNAPSHOT_HOUR / 24


def load_sets(folder: str, shards: list) -> dict:
    """{norad: column arrays}, oldest set first, for the years a 2025 window reads."""
    found = {}
    for shard in shards:
        for year in (2024, 2025):
            path = Path(folder) / f"shard={shard}" / f"tle_{year}.parquet"
            table = pq.read_table(path, columns=list(COLUMNS)).to_pandas()
            for norad, rows in table.groupby("norad"):
                found.setdefault(int(norad), []).append(rows.sort_values("epoch"))
    out = {}
    for norad, parts in found.items():
        rows = pd.concat(parts)
        out[norad] = {c: rows[c].to_numpy() for c in COLUMNS}
    return out


def csv_row(columns: dict, k: int, norad: int) -> dict:
    """One set as the CelesTrak CSV row the daily job stores. Field meanings match parse_pair."""
    moment = datetime(1970, 1, 1) + timedelta(days=float(columns["epoch"][k]) + SGP4_EPOCH_JD - drift.JD_UNIX_EPOCH)
    return {
        "NORAD_CAT_ID": str(norad), "OBJECT_ID": "2000-001A", "OBJECT_NAME": "REPLAY", "EPHEMERIS_TYPE": "0",
        "CLASSIFICATION_TYPE": "U", "ELEMENT_SET_NO": "999", "REV_AT_EPOCH": "100", "EPOCH": moment.isoformat(timespec="microseconds"),
        "MEAN_MOTION": repr(float(columns["no_rev"][k])), "ECCENTRICITY": repr(float(columns["ecco"][k])),
        "INCLINATION": repr(float(columns["inclo"][k])), "RA_OF_ASC_NODE": repr(float(columns["nodeo"][k])),
        "ARG_OF_PERICENTER": repr(float(columns["argpo"][k])), "MEAN_ANOMALY": repr(float(columns["mo"][k])),
        "MEAN_MOTION_DOT": repr(float(columns["ndot"][k])), "MEAN_MOTION_DDOT": repr(float(columns["nddot"][k])),
        "BSTAR": repr(float(columns["bstar"][k])),
    }


def snapshots_for(sets: dict, target: date) -> dict:
    """{date: {norad: newest row held that day}}, for the window of snapshots the job keeps on the target day."""
    out = {}
    for back in range(WINDOW_DAYS):
        day = target - timedelta(days=back)
        limit = epoch_of(day)
        held = {}
        for norad, columns in sets.items():
            k = int(np.searchsorted(columns["epoch"], limit, side="right")) - 1
            # An object with no set for 30 days has left the active list.
            if k >= 0 and limit - columns["epoch"][k] <= drift.UPDATE_WINDOW:
                held[norad] = csv_row(columns, k, norad)
        out[day] = held
    return out


def forward_gap(nets: list, path: str, x: np.ndarray, horizon: float) -> float:
    """Largest km difference between the numpy port of the first net and its torch source."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**saved["args"])
    net = Net(saved["state"]["mean"].shape[0], args.depth, args.width, args.dropout)
    net.load_state_dict(saved["state"])
    dt = np.full(len(x), horizon, np.float32)
    drifted = np.clip(drift.own_drift(x, dt), -drift.MAX_DRIFT_KM, drift.MAX_DRIFT_KM)
    scale = np.maximum(np.abs(drifted), 1.0)
    theirs = predict(net, (torch.from_numpy(drift.inputs(x, dt)), torch.from_numpy(drifted), torch.from_numpy(scale)), "cpu")
    ours = drift.forward(nets[0], drift.inputs(x, dt), drifted, scale)
    return float(np.max(np.abs(theirs - ours)))


def first_rows(snapshots: dict, day: date, horizon: int) -> np.ndarray:
    known = drift.sets_by_object([snapshots[d] for d in sorted(snapshots) if d <= day])
    begun = day - timedelta(days=horizon)
    x = []
    for norad, row in snapshots[begun].items():
        epochs, rows = known[norad]
        x.append(drift.features(epochs, rows, int(np.searchsorted(epochs, drift.epoch_day(row))), 30))
    return np.array(x, dtype=np.float32)


def main(folder: str, weights: str, out: str, net_path: str, shards: list) -> None:
    nets = drift.load_nets(weights)
    sets = load_sets(folder, shards)
    print("objects", len(sets), flush=True)
    rows = []
    target = FIRST
    checked = False
    while target <= LAST:
        snapshots = snapshots_for(sets, target)
        started = time.time()
        scored = drift.score_day(snapshots, target, nets)
        seconds = time.time() - started
        if not checked:
            print("largest km gap, numpy against torch:", forward_gap(nets, net_path, first_rows(snapshots, target, 3), 3.0), flush=True)
            checked = True
        for horizon, row in scored.items():
            rows.append([target.isoformat(), horizon, row["n"], row["sgp4_km"], row["model_km"], round(seconds, 1), len(snapshots[target])])
            print(*rows[-1], flush=True)
        target += timedelta(days=7)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "horizon", "n", "sgp4_km", "model_km", "seconds_for_day", "objects_held"])
        writer.writerows(rows)
    for horizon in {r[1] for r in rows}:
        part = [r for r in rows if r[1] == horizon]
        wins = sum(r[4] < r[3] for r in part)
        print("horizon", horizon, "days", len(part), "model beat sgp4", wins, "median sgp4", np.median([r[3] for r in part]),
              "median model", np.median([r[4] for r in part]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])

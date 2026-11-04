import csv
import gzip
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from forecast import drift, fetch, orbits, run

SAMPLE = (Path(__file__).parent / "active-sample.csv").read_text()
TODAY = date(2026, 11, 20)
NETS = drift.load_nets()


def later(rows, hours):
    """The same elements with a later epoch, which SGP4 reads as a large error."""
    out = {}
    for key, row in rows.items():
        moment = datetime.fromisoformat(row["EPOCH"]) + timedelta(hours=hours)
        out[key] = {**row, "EPOCH": moment.isoformat()}
    return out


def daily(days):
    """Snapshots for the given offsets from TODAY, with a new element set each day."""
    rows = orbits.parse(SAMPLE)
    return {TODAY + timedelta(days=d): later(rows, 24 * d + 12) for d in days}


def as_csv(snapshot):
    rows = list(snapshot.values())
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def test_the_shipped_weights_are_three_networks_of_the_chosen_shape():
    assert len(NETS) == 3
    for net in NETS:
        assert net["mean"].shape == (45,)
        assert net["first.weight"].shape == (256, 45)
        assert net["last.1.weight"].shape == (4, 256)


def test_a_day_scores_each_horizon_it_has_a_start_for():
    scored = drift.score_day(daily([-3, -1, 0]), TODAY, NETS)
    assert sorted(scored) == ["1", "3"]
    for row in scored.values():
        assert row["n"] > 0
        assert row["sgp4_km"] > 0
        assert np.isfinite(row["model_km"])


def test_a_day_with_no_earlier_snapshot_scores_nothing():
    assert drift.score_day(daily([0]), TODAY, NETS) == {}


def test_a_replayed_day_does_not_read_later_snapshots():
    held = daily([-3, -2, -1, 0, 1, 2])
    cut = {d: s for d, s in held.items() if d <= TODAY}
    assert drift.score_day(held, TODAY, NETS) == drift.score_day(cut, TODAY, NETS)


def test_a_set_seen_on_several_days_counts_once():
    rows = orbits.parse(SAMPLE)
    epochs, kept = drift.sets_by_object([rows, rows])[next(iter(rows))]
    assert len(epochs) == len(kept) == 1


def test_a_missing_later_set_leaves_the_satellite_out():
    held = daily([-1, 0])
    complete = drift.score_day(held, TODAY, NETS)["1"]["n"]
    # Pick one the complete run scored: its later set is the only one past the horizon.
    start = held[TODAY - timedelta(days=1)]
    known = drift.sets_by_object(held.values())
    scored = next(n for n in start if drift.score_start(known, {n: start[n]}, 1, 1, NETS))
    del held[TODAY][scored]
    assert drift.score_day(held, TODAY, NETS)["1"]["n"] == complete - 1


def test_rows_mark_only_the_current_day_as_issued_and_are_never_rewritten():
    held = daily([-2, -1, 0])
    old = {"date": (TODAY - timedelta(days=1)).isoformat(), "issued": True, "days": {"1": {"n": 1, "sgp4_km": 9.0, "model_km": 8.0}}}
    rows = run.satellite_losses([old], held, TODAY)
    assert [r["date"] for r in rows] == [(TODAY - timedelta(days=1)).isoformat(), TODAY.isoformat()]
    assert rows[0] == old
    assert rows[1]["issued"] is True


def test_a_missed_run_is_filled_and_flagged_as_replayed():
    held = daily([-2, -1, 0])
    rows = run.satellite_losses([], held, TODAY)
    assert [r["issued"] for r in rows] == [False, True]


def test_only_recent_rows_are_kept():
    stale = [{"date": (TODAY - timedelta(days=d)).isoformat(), "issued": True, "days": {}} for d in range(1, 60)]
    assert len(run.satellite_losses(stale, daily([0]), TODAY)) == run.LOSS_DAYS


def test_the_update_scores_today_from_stored_snapshots(tmp_path, monkeypatch):
    folder = tmp_path / "elements"
    folder.mkdir()
    yesterday = TODAY - timedelta(days=1)
    (folder / f"{yesterday.isoformat()}.csv.gz").write_bytes(gzip.compress(as_csv(daily([-1])[yesterday]).encode(), mtime=0))
    monkeypatch.setattr(fetch, "elements", lambda: as_csv(daily([0])[TODAY]))
    section = run.update_satellites(tmp_path, TODAY)
    assert section["losses"][0]["date"] == TODAY.isoformat()
    assert section["losses"][0]["issued"] is True
    json.dumps(section, allow_nan=False)

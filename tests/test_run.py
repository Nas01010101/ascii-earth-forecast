import csv
import io
import json
import struct
from datetime import date, timedelta

import pytest

from forecast import fetch, run
from synthetic import TODAY


def test_satellites_score_only_with_history(tmp_path, monkeypatch):
    text = open("tests/active-sample.csv").read()
    monkeypatch.setattr(fetch, "elements", lambda: text)
    first = run.update_satellites(tmp_path, TODAY)
    assert first["losses"] == []
    calls = []
    monkeypatch.setattr(fetch, "elements", lambda: calls.append(1) or text)
    run.update_satellites(tmp_path, TODAY)
    assert calls == []  # A same-day rerun does not download again.


def test_a_bad_row_does_not_stop_the_satellite_update(tmp_path, monkeypatch):
    rows = list(csv.DictReader(io.StringIO(open("tests/active-sample.csv").read())))
    rows[0]["MEAN_MOTION"] = ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    monkeypatch.setattr(fetch, "elements", lambda: out.getvalue())
    assert run.update_satellites(tmp_path, TODAY)["count"] == len(rows) - 1
    assert (tmp_path / "sats.bin").exists()
    assert run.update_satellites(tmp_path, TODAY)["count"] == len(rows) - 1


def test_a_failed_pack_leaves_no_snapshot_so_a_rerun_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "elements", lambda: open("tests/active-sample.csv").read())
    def broken(text):
        raise struct.error("no")
    monkeypatch.setattr(run.orbits, "pack", broken)
    with pytest.raises(struct.error):
        run.update_satellites(tmp_path, TODAY)
    assert not list((tmp_path / "elements").glob("*.csv.gz"))


def test_an_unexpected_crash_stays_inside_its_source(tmp_path):
    ok = lambda root, today: {"value": 1}
    def crash(root, today):
        raise KeyError("surprise")
    assert run.run(tmp_path, TODAY, {"weather": crash, "satellites": ok}) == 1
    report = json.loads((tmp_path / "forecast.json").read_text())
    assert report["status"]["weather"]["ok"] is False
    assert report["satellites"] == {"value": 1}


def test_one_failing_source_keeps_the_last_good_section(tmp_path):
    ok = lambda root, today: {"value": 1}
    def down(root, today):
        raise fetch.SourceError("down")
    assert run.run(tmp_path, TODAY, {"weather": ok, "satellites": ok}) == 0
    tomorrow = TODAY + timedelta(days=1)
    assert run.run(tmp_path, tomorrow, {"weather": down, "satellites": ok}) == 1
    report = json.loads((tmp_path / "forecast.json").read_text())
    assert report["weather"] == {"value": 1}
    assert report["status"]["weather"]["ok"] is False
    assert report["status"]["weather"]["updated"] == TODAY.isoformat()
    assert report["status"]["satellites"]["ok"] is True


def test_nan_in_a_score_fails_that_source_and_keeps_the_last_section(tmp_path):
    (tmp_path / "forecast.json").write_text('{"weather": {"keep": 1}}')
    assert run.run(tmp_path, TODAY, {"weather": lambda r, t: {"bad": float("nan")}}) == 1
    report = json.loads((tmp_path / "forecast.json").read_text())
    assert report["weather"] == {"keep": 1}
    assert report["status"]["weather"]["ok"] is False


def test_an_update_done_today_is_not_run_again(tmp_path):
    calls = []
    def update(root, today):
        calls.append(today)
        return {}
    today = date(2026, 11, 5)
    assert run.run(tmp_path, today, {"weather": update}) == 0
    assert run.run(tmp_path, today, {"weather": update}) == 0
    assert calls == [today]
    assert run.run(tmp_path, today + timedelta(days=1), {"weather": update}) == 0
    assert len(calls) == 2


def test_force_runs_an_update_done_today_again(tmp_path):
    calls = []
    def update(root, today):
        calls.append(today)
        return {}
    today = date(2026, 11, 5)
    assert run.run(tmp_path, today, {"weather": update}) == 0
    assert run.run(tmp_path, today, {"weather": update}, force=True) == 0
    assert calls == [today, today]


def test_a_later_run_retries_only_the_source_that_failed(tmp_path):
    calls = {"weather": 0, "satellites": 0}
    def good(root, today):
        calls["weather"] += 1
        return {}
    outcomes = [RuntimeError("down"), {}]
    def flaky(root, today):
        calls["satellites"] += 1
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    today = date(2026, 11, 5)
    updates = {"weather": good, "satellites": flaky}
    assert run.run(tmp_path, today, updates) == 1
    assert run.run(tmp_path, today, updates) == 0
    assert calls == {"weather": 1, "satellites": 2}

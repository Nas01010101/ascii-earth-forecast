import csv
import io
import json
import struct
from datetime import date, timedelta

import numpy as np
import pytest

from forecast import fetch, run

TODAY = date(2026, 11, 3)
SHAPE = (len(fetch.LATS), len(fetch.LONS))


def hourly(days_back, seed=0):
    rng = np.random.default_rng(seed)
    start = np.datetime64(TODAY - timedelta(days=days_back), "h")
    hours = start + np.arange((days_back + fetch.FORECAST_DAYS) * 24)
    fields = {"cloud": rng.uniform(0, 100, (hours.size, *SHAPE)).astype(np.float32),
              "temp": rng.uniform(-20, 30, (hours.size, *SHAPE)).astype(np.float32),
              "precip": rng.uniform(0, 3, (hours.size, *SHAPE)).astype(np.float32),
              "pressure": rng.uniform(980, 1030, (hours.size, *SHAPE)).astype(np.float32),
              "speed": rng.uniform(0, 20, (hours.size, *SHAPE)).astype(np.float32),
              "direction": rng.uniform(0, 360, (hours.size, *SHAPE)).astype(np.float32),
              "water": rng.uniform(0, 60, (hours.size, *SHAPE)).astype(np.float32)}
    return hours, fields


def test_only_complete_days_before_today_enter_history():
    hours, fields = hourly(3)
    history = {}
    run.add_days(history, hours[5:], {k: v[5:] for k, v in fields.items()}, TODAY)
    # The first day lost five hours and today and tomorrow are not complete past days.
    assert sorted(history) == ["2026-11-01", "2026-11-02"]


def test_history_is_trimmed_to_the_window():
    hours, fields = hourly(run.HISTORY_DAYS + 5)
    history = {}
    run.add_days(history, hours, fields, TODAY)
    assert len(history) == run.HISTORY_DAYS


def test_a_day_with_too_many_missing_cells_is_skipped_not_fatal():
    hours, fields = hourly(2)
    fields["cloud"][:24] = np.nan
    history = {}
    run.add_days(history, hours, fields, TODAY)
    assert sorted(history) == ["2026-11-02"]


def test_one_null_hour_does_not_lose_the_cell():
    hours, fields = hourly(1)
    fields["cloud"][3] = np.nan
    history = {}
    run.add_days(history, hours, fields, TODAY)
    assert not np.isnan(history["2026-11-02"][0]).any()


def test_stored_days_are_never_rewritten():
    hours, fields = hourly(1)
    history = {"2026-11-02": ("kept", "kept")}
    run.add_days(history, hours, fields, TODAY)
    assert history["2026-11-02"] == ("kept", "kept")


def past(days):
    return {(TODAY - timedelta(days=d)).isoformat(): None for d in range(1, days + 1)}


def test_fetch_asks_only_for_missing_days_and_heals_holes():
    steps = past(run.STEP_DAYS)
    assert run.days_to_fetch({}, steps, TODAY) == fetch.BOOTSTRAP_DAYS
    full = past(fetch.BOOTSTRAP_DAYS)
    assert run.days_to_fetch(full, steps, TODAY) == 0
    del full[(TODAY - timedelta(days=1)).isoformat()]
    assert run.days_to_fetch(full, steps, TODAY) == 1
    del full[(TODAY - timedelta(days=5)).isoformat()]
    assert run.days_to_fetch(full, steps, TODAY) == 5


def test_a_missing_step_day_also_fetches_the_day_before_for_its_rain():
    full = past(fetch.BOOTSTRAP_DAYS)
    assert run.days_to_fetch(full, {}, TODAY) == run.STEP_DAYS + 1
    steps = past(run.STEP_DAYS)
    del steps[(TODAY - timedelta(days=3)).isoformat()]
    assert run.days_to_fetch(full, steps, TODAY) == 4


def test_a_gap_stops_the_consecutive_run():
    history = {d: None for d in ("2026-11-02", "2026-11-01", "2026-10-30")}
    assert run.consecutive(history, TODAY) == ["2026-11-01", "2026-11-02"]


def test_picture_puts_north_first():
    field = np.zeros((2, 2))
    field[1] = 100
    assert run.picture(field, 100) == ["@@", "  "]


def test_frames_cover_today_and_tomorrow_every_three_hours():
    hours, fields = hourly(1)
    data = run.planet_frames(hours, fields, TODAY)
    frames = 16
    assert len(data) == 16 + 2 * frames * SHAPE[0] * SHAPE[1]


def test_frames_with_missing_future_hours_are_refused():
    hours, fields = hourly(1)
    fields["cloud"][hours >= np.datetime64(TODAY)] = np.nan
    with pytest.raises(fetch.SourceError):
        run.planet_frames(hours, fields, TODAY)


def test_planet_survives_a_day_that_cannot_be_scored(tmp_path, monkeypatch):
    def holed(past):
        hours, fields = hourly(past)
        fields["temp"][hours.astype("datetime64[D]") == np.datetime64(TODAY - timedelta(days=1))] = np.nan
        return hours, fields
    monkeypatch.setattr(fetch, "weather", holed)
    with pytest.raises(fetch.SourceError):
        run.update_weather(tmp_path, TODAY)
    assert (tmp_path / "planet.bin").exists()


def test_weather_update_end_to_end_and_incremental(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "weather", lambda past: hourly(past))
    first = run.update_weather(tmp_path, TODAY)
    assert first["today"]["net"] is True
    assert len(first["today"]["map"]) == SHAPE[0]
    assert len(first["last_scored"]["actual"][0]) == SHAPE[1]
    assert len(run.load_steps(tmp_path)) == run.STEP_DAYS
    json.dumps(first, allow_nan=False)
    # A rerun on the same day has nothing to backfill.
    asked = []
    monkeypatch.setattr(fetch, "weather", lambda past: (asked.append(past), hourly(past))[1])
    run.update_weather(tmp_path, TODAY)
    assert asked == [0]


def test_today_names_places_with_the_stored_forecast():
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(30, 0, -1)]
    history = {d: (field(90), field(0)) for d in days}
    forecasts = {TODAY.isoformat(): (field(100), field(150), True, True)}
    today = run.weather_section(days, history, forecasts, TODAY)["today"]
    assert [p["place"] for p in today["places"]] == [c[0] for c in run.weather.CELLS]
    assert {p["cloud"] for p in today["places"]} == {100.0}
    assert {p["temp"] for p in today["places"]} == {150.0}


def test_a_field_with_too_many_missing_cells_keeps_the_ridge_and_says_so(tmp_path, monkeypatch):
    def no_water(past):
        hours, fields = hourly(past)
        fields["water"][:] = np.nan
        return hours, fields
    monkeypatch.setattr(fetch, "weather", no_water)
    today = run.update_weather(tmp_path, TODAY)["today"]
    assert today["net"] is False
    assert run.load_steps(tmp_path) == {}
    assert run.load_forecasts(tmp_path)[TODAY.isoformat()][3] is False


def test_a_rare_missing_cell_is_filled_in_a_stored_step():
    hours, fields = hourly(2)
    fields["water"][hours == np.datetime64(TODAY - timedelta(days=1), "h"), 3, 4] = np.nan
    steps = {}
    run.add_steps(steps, hours, fields, TODAY)
    assert sorted(steps) == [(TODAY - timedelta(days=1)).isoformat()]
    assert not np.isnan(steps[(TODAY - timedelta(days=1)).isoformat()]).any()


def test_stored_steps_are_never_rewritten_and_the_window_is_trimmed():
    hours, fields = hourly(run.STEP_DAYS + 5)
    old = (TODAY - timedelta(days=run.STEP_DAYS + 3)).isoformat()
    held = (TODAY - timedelta(days=2)).isoformat()
    steps = {old: "old", held: "kept"}
    run.add_steps(steps, hours, fields, TODAY)
    assert old not in steps and steps[held] == "kept"
    assert len(steps) == run.STEP_DAYS


def test_steps_round_trip_through_the_data_directory(tmp_path):
    values = np.arange(7 * 4 * 2 * 2, dtype=np.float32).reshape(7, 4, 2, 2)
    run.save_steps(tmp_path, {"2026-11-02": values, "2026-11-01": values + 1})
    saved = run.load_steps(tmp_path)
    assert sorted(saved) == ["2026-11-01", "2026-11-02"]
    assert np.array_equal(saved["2026-11-01"], values + 1)
    assert not list(tmp_path.glob("*.tmp*"))


def test_a_forecast_needs_every_day_of_steps_for_the_network():
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(30, 0, -1)]
    cloud, temp = (np.stack([field(v)] * 30) for v in (50, 5))
    steps = {d: np.zeros((7, 4, *SHAPE), dtype=np.float32) for d in days[1:]}
    assert run._forecast(cloud, temp, 30, TODAY, steps, None)[2] is False


def test_a_network_forecast_is_clipped_to_the_cover_range(monkeypatch):
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(30, 0, -1)]
    cloud, temp = (np.stack([field(v)] * 30) for v in (50, 5))
    steps = {d: np.zeros((7, 4, *SHAPE), dtype=np.float32) for d in days}
    monkeypatch.setattr(run.cloudnet, "predict", lambda held, when, saved: field(150))
    found, _, net = run._forecast(cloud, temp, 30, TODAY, steps, None)
    assert net is True and found.max() == 100


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
    assert run.run(tmp_path, TODAY, {"weather": down, "satellites": ok}) == 1
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


def field(value):
    return np.full(SHAPE, value, dtype=np.float32)


def test_losses_score_the_stored_forecast_and_predict_nothing(monkeypatch):
    def no_prediction(*args):
        raise AssertionError("a past forecast was computed again")
    monkeypatch.setattr(run.weather, "predict", no_prediction)
    history = {"2026-11-01": (field(70), field(1)), "2026-11-02": (field(50), field(4))}
    forecasts = {"2026-11-02": (field(60), field(2), True, True), "2026-11-03": (field(0), field(0), True, True)}
    # The forecast for today has no actual yet, so only yesterday's is scored.
    assert run.forecast_losses(forecasts, history) == [{
        "date": "2026-11-02", "issued": True, "net": True,
        "cloud": {"model": 100.0, "repeat": 400.0},
        "temp": {"model": 4.0, "repeat": 9.0},
    }]


def test_a_day_with_no_day_before_it_has_no_repeat_loss():
    rows = run.forecast_losses({"2026-11-02": (field(60), field(2), False, False)}, {"2026-11-02": (field(50), field(4))})
    assert rows[0]["cloud"]["repeat"] is None
    json.dumps(rows, allow_nan=False)


def test_losses_cover_the_last_few_weeks_while_older_forecasts_stay_stored():
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(run.LOSS_DAYS + 5, 0, -1)]
    history = {d: (field(50), field(0)) for d in days}
    rows = run.forecast_losses({d: (field(60), field(0), True, False) for d in days}, history)
    assert [r["date"] for r in rows] == days[-run.LOSS_DAYS:]


def test_first_run_issues_today_and_replays_the_past(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "weather", lambda past: hourly(past))
    section = run.update_weather(tmp_path, TODAY)
    forecasts = run.load_forecasts(tmp_path)
    assert forecasts[TODAY.isoformat()][2] is True
    assert TODAY.isoformat() not in [r["date"] for r in section["losses"]]
    assert section["losses"] and not any(r["issued"] for r in section["losses"])
    assert forecasts[TODAY.isoformat()][0].min() >= 0 and forecasts[TODAY.isoformat()][0].max() <= 100


def test_a_rerun_keeps_the_first_forecast_of_the_day(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "weather", lambda past: hourly(past))
    run.update_weather(tmp_path, TODAY)
    first = run.load_forecasts(tmp_path)[TODAY.isoformat()][0].copy()
    real = run.cloudnet.predict
    monkeypatch.setattr(run.cloudnet, "predict", lambda *args: real(*args) + 1)
    run.update_weather(tmp_path, TODAY)
    assert np.array_equal(run.load_forecasts(tmp_path)[TODAY.isoformat()][0], first)


def test_a_stored_day_is_never_replayed():
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(25, 0, -1)]
    history = {d: (field(50), field(0)) for d in days}
    forecasts = {days[-1]: (field(1), field(2), True, True)}
    run.keep_forecasts(forecasts, days, history, {}, TODAY, None)
    assert forecasts[days[-1]][0][0, 0] == 1 and forecasts[days[-1]][2] is True
    assert forecasts[days[-2]][2] is False


def test_a_replay_does_not_see_its_own_day_or_later():
    rng = np.random.default_rng(1)
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(25, 0, -1)]
    history = {d: (rng.uniform(0, 100, SHAPE), rng.uniform(-20, 30, SHAPE)) for d in days}
    plain = {}
    run.keep_forecasts(plain, days, history, {}, TODAY, None)
    changed = dict(history, **{d: (field(100), field(30)) for d in days[-3:]})
    other = {}
    run.keep_forecasts(other, days, changed, {}, TODAY, None)
    assert np.array_equal(plain[days[-3]][0], other[days[-3]][0])
    assert not np.array_equal(plain[days[-2]][0], other[days[-2]][0])


def test_a_network_replay_does_not_see_its_own_day_or_later(monkeypatch):
    """Every stored day is reachable from a later target, so only the view decides what a replay reads."""
    days = [(TODAY - timedelta(days=d)).isoformat() for d in range(40, 0, -1)]
    history = {d: (field(50), field(0)) for d in days}
    steps = {d: np.full((7, 4, *SHAPE), i, dtype=np.float32) for i, d in enumerate(days)}
    seen = {}
    def record(held, when, saved):
        seen[when.isoformat()] = held[0, -1, 0, 0]
        return field(50)
    monkeypatch.setattr(run.cloudnet, "predict", record)
    run.keep_forecasts({}, days, history, steps, TODAY, None)
    assert seen[TODAY.isoformat()] == steps[days[-1]][0, 0, 0, 0]
    assert seen[days[-1]] == steps[days[-2]][0, 0, 0, 0]


def test_forecasts_round_trip_through_the_data_directory(tmp_path):
    run.save_forecasts(tmp_path, {"2026-11-02": (field(60), field(2), True, True), "2026-11-01": (field(10), field(3), False, False)})
    saved = run.load_forecasts(tmp_path)
    assert sorted(saved) == ["2026-11-01", "2026-11-02"]
    assert saved["2026-11-02"][2] is True and saved["2026-11-01"][2] is False
    assert saved["2026-11-02"][3] is True and saved["2026-11-01"][3] is False
    assert saved["2026-11-01"][0][0, 0] == 10 and saved["2026-11-01"][1][0, 0] == 3
    assert not list(tmp_path.glob("*.tmp*"))


def test_forecasts_stored_before_the_network_load_as_the_ridge(tmp_path):
    np.savez_compressed(tmp_path / "forecasts.npz", days=np.array(["2026-11-02"]), cloud=np.stack([field(60)]),
                        temp=np.stack([field(2)]), issued=np.array([True]))
    saved = run.load_forecasts(tmp_path)["2026-11-02"]
    assert saved[2] is True and saved[3] is False


def test_tomorrow_scores_todays_issued_forecast(tmp_path, monkeypatch):
    def from_day(day):
        def weather_on(past):
            start = np.datetime64(day - timedelta(days=past), "h")
            stamps = start + np.arange((past + fetch.FORECAST_DAYS) * 24)
            rng = np.random.default_rng(past)
            return stamps, {name: rng.uniform(0, 50, (stamps.size, *SHAPE)).astype(np.float32) for name in fetch.VARIABLES}
        return weather_on
    monkeypatch.setattr(fetch, "weather", from_day(TODAY))
    run.update_weather(tmp_path, TODAY)
    issued = run.load_forecasts(tmp_path)[TODAY.isoformat()]
    next_day = TODAY + timedelta(days=1)
    monkeypatch.setattr(fetch, "weather", from_day(next_day))
    last = run.update_weather(tmp_path, next_day)["losses"][-1]
    assert last["date"] == TODAY.isoformat() and last["issued"] is True
    actual = run.load_history(tmp_path)[TODAY.isoformat()][0]
    assert last["cloud"]["model"] == round(run.weather.mse(actual, issued[0]), 1)

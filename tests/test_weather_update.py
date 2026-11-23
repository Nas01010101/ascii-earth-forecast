import json
from datetime import timedelta

import numpy as np
import pytest

from forecast import fetch, run, weather
from synthetic import SHAPE, TODAY, field, hourly, past_days


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
    days = past_days(30)
    history = {d: (field(90), field(0)) for d in days}
    forecasts = {TODAY.isoformat(): (field(100), field(150), True, True)}
    today = run.weather_section(days, history, forecasts, TODAY)["today"]
    assert [p["place"] for p in today["places"]] == [c[0] for c in run.weather.CELLS]
    assert {p["cloud"] for p in today["places"]} == {100.0}
    assert {p["temp"] for p in today["places"]} == {150.0}


def test_today_carries_the_network_cloud_grid_as_whole_percent():
    days = past_days(30)
    history = {d: (field(90), field(0)) for d in days}
    cloud = field(40.4)
    cloud[0, 0], cloud[-1, -1] = -3.0, 120.0
    forecasts = {TODAY.isoformat(): (cloud, field(150), True, True)}
    grid = run.weather_section(days, history, forecasts, TODAY)["today"]["cloud"]
    assert len(grid) == cloud.size
    assert all(type(v) is int for v in grid)
    assert (grid[0], grid[1], grid[-1]) == (0, 40, 100)


def test_today_carries_a_temperature_grid_in_whole_degrees():
    days = past_days(30)
    history = {d: (field(90), field(0)) for d in days}
    temp = field(4.6)
    temp[0, 0], temp[-1, -1] = -31.4, 28.9
    forecasts = {TODAY.isoformat(): (field(40), temp, True, True)}
    grid = run.weather_section(days, history, forecasts, TODAY)["today"]["temp"]
    assert len(grid) == temp.size
    assert all(type(v) is int for v in grid)
    assert (grid[0], grid[1], grid[-1]) == (-31, 5, 29)


def test_the_page_reads_the_same_cell_from_the_grids_as_from_the_places():
    days = past_days(30)
    history = {d: (field(90), field(0)) for d in days}
    rng = np.random.default_rng(3)
    cloud, temp = rng.uniform(0, 100, SHAPE), rng.uniform(-20, 30, SHAPE)
    forecasts = {TODAY.isoformat(): (cloud, temp, True, True)}
    today = run.weather_section(days, history, forecasts, TODAY)["today"]
    for place, lat, lon in weather.CELLS:
        row, col = int(np.floor((lat + 90) / fetch.STEP)), int(np.floor((lon + 180) / fetch.STEP))
        found = next(p for p in today["places"] if p["place"] == place)
        assert abs(today["cloud"][row * SHAPE[1] + col] - found["cloud"]) <= 0.55
        assert abs(today["temp"][row * SHAPE[1] + col] - found["temp"]) <= 0.55


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

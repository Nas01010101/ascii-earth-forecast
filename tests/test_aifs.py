import json
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import numpy as np

from forecast import fetch, run
from synthetic import SHAPE, TODAY, field, hourly


def aifs_hours(day=TODAY, value=(40.0, 5.0, 1010.0)):
    hours = np.datetime64(day, "h") + np.arange(24)
    return hours, {name: np.full((24, *SHAPE), v, np.float32) for name, v in zip(fetch.AIFS_VARIABLES, value)}


def test_the_url_names_the_aifs_model_and_only_its_three_fields():
    url = fetch.weather_url(fetch.cells()[:2], 0, fetch.AIFS_VARIABLES, fetch.AIFS_MODEL, 1)
    query = parse_qs(urlparse(url).query)
    assert query["models"] == ["ecmwf_aifs025_single"]
    assert query["hourly"] == ["cloud_cover,temperature_2m,pressure_msl"]
    assert query["forecast_days"] == ["1"]
    assert "models" not in parse_qs(urlparse(fetch.weather_url(fetch.cells()[:2], 0)).query)


def test_the_aifs_download_parses_its_fields_and_waits_before_its_first_chunk(monkeypatch):
    hours = [f"2026-11-03T{h:02d}:00" for h in range(24)]
    one = {"hourly": {"time": hours, "cloud_cover": [40] * 24, "temperature_2m": [5.0] * 24, "pressure_msl": [1010.0] * 24}}
    waits = []
    monkeypatch.setattr(fetch, "get", lambda url: json.dumps([one] * (parse_qs(urlparse(url).query)["latitude"][0].count(",") + 1)).encode())
    found_hours, fields = fetch.aifs(pause=waits.append)
    assert set(fields) == {"cloud", "temp", "pressure"}
    assert fields["pressure"].shape == (24, *SHAPE)
    # One wait before each of the 12 chunks, the first included.
    assert len(waits) == -(-len(fetch.cells()) // fetch.CHUNK)
    assert waits[0] > 0


def test_a_day_of_aifs_hours_becomes_its_mean_in_stored_order():
    found = run.aifs_day(*aifs_hours(), TODAY)
    assert found.shape == (3, *SHAPE)
    assert found[:, 0, 0].tolist() == [40.0, 5.0, 1010.0]
    assert run.aifs_day(*aifs_hours(TODAY - timedelta(days=1)), TODAY) is None


def test_aifs_is_stored_once_and_a_failed_download_is_only_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "aifs", lambda: (_ for _ in ()).throw(fetch.SourceError("down")))
    assert run.keep_aifs(tmp_path, TODAY) == {}
    assert "::warning::aifs: down" in capsys.readouterr().err
    monkeypatch.setattr(fetch, "aifs", lambda: aifs_hours())
    first = run.keep_aifs(tmp_path, TODAY)
    monkeypatch.setattr(fetch, "aifs", lambda: aifs_hours(value=(0.0, 0.0, 1000.0)))
    assert run.keep_aifs(tmp_path, TODAY)[TODAY.isoformat()][0, 0, 0] == first[TODAY.isoformat()][0, 0, 0] == 40.0


def test_losses_score_aifs_beside_the_network():
    history = {"2026-11-01": (field(70), field(1)), "2026-11-02": (field(50), field(4))}
    forecasts = {"2026-11-02": (field(60), field(2), True, True)}
    aifs = {"2026-11-02": np.stack([field(55), field(3), field(1000)])}
    row = run.forecast_losses(forecasts, history, aifs=aifs)[0]
    assert row["cloud"] == {"model": 100.0, "repeat": 400.0, "aifs": 25.0}
    assert row["temp"]["aifs"] == 1.0
    # A day with no stored AIFS forecast has no AIFS score.
    assert "aifs" not in run.forecast_losses(forecasts, history)[0]["cloud"]


def test_the_daily_job_publishes_aifs_and_names_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "weather", lambda past: hourly(past))
    monkeypatch.setattr(fetch, "aifs", lambda: aifs_hours())
    today = run.update_weather(tmp_path, TODAY)["today"]
    assert today["aifs"]["model"] == "ECMWF AIFS Single"
    assert today["aifs"]["licence"] == "CC BY 4.0"
    assert len(today["aifs"]["cloud"]) == len(today["aifs"]["pressure"]) == SHAPE[0] * SHAPE[1]
    assert (tmp_path / "aifs.npz").exists()

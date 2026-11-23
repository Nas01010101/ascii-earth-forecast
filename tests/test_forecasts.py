import json
from datetime import timedelta

import numpy as np

from forecast import run
from synthetic import SHAPE, TODAY, field, past_days


def test_a_forecast_needs_every_day_of_steps_for_the_network():
    days = past_days(30)
    cloud, temp = (np.stack([field(v)] * 30) for v in (50, 5))
    steps = {d: np.zeros((7, 4, *SHAPE), dtype=np.float32) for d in days[1:]}
    assert run._forecast(cloud, temp, 30, TODAY, steps, None).net is False


def test_a_network_forecast_is_clipped_to_the_cover_range(monkeypatch):
    days = past_days(30)
    cloud, temp = (np.stack([field(v)] * 30) for v in (50, 5))
    steps = {d: np.zeros((7, 4, *SHAPE), dtype=np.float32) for d in days}
    monkeypatch.setattr(run.cloudnet, "predict", lambda held, when, saved: field(150))
    made = run._forecast(cloud, temp, 30, TODAY, steps, None)
    assert made.net is True and made.cloud.max() == 100


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
    days = past_days(run.LOSS_DAYS + 5)
    history = {d: (field(50), field(0)) for d in days}
    rows = run.forecast_losses({d: (field(60), field(0), True, False) for d in days}, history)
    assert [r["date"] for r in rows] == days[-run.LOSS_DAYS:]


def test_a_stored_day_is_never_replayed():
    days = past_days(25)
    history = {d: (field(50), field(0)) for d in days}
    forecasts = {days[-1]: (field(1), field(2), True, True)}
    run.keep_forecasts(forecasts, days, history, {}, TODAY, None)
    assert forecasts[days[-1]][0][0, 0] == 1 and forecasts[days[-1]][2] is True
    assert forecasts[days[-2]][2] is False


def test_a_replay_does_not_see_its_own_day_or_later():
    rng = np.random.default_rng(1)
    days = past_days(25)
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
    days = past_days(40)
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


def _others(saved):
    return {"temp": dict(saved), "pressure": dict(saved)}


def test_the_networks_make_temperature_and_pressure_when_their_days_are_held(monkeypatch):
    days = past_days(30)
    seen = []

    def predict(held, when, saved, target="cloud"):
        seen.append(target)
        return field({"cloud": 40.0, "temp": 7.0, "pressure": 1012.34}[target])

    monkeypatch.setattr(run.cloudnet, "predict", predict)
    monkeypatch.setattr(run.cloudnet, "view", lambda steps, when: np.zeros(1))
    history = {d: (field(90), field(0)) for d in days}
    forecasts = {}
    run.keep_forecasts(forecasts, days, history, {}, TODAY, {}, _others({}))
    assert seen[:3] == ["cloud", "temp", "pressure"]
    made = forecasts[TODAY.isoformat()]
    assert made.net and made.temp_net and made.temp[0, 0] == 7.0 and made.pressure[0, 0] == 1012.34
    today = run.weather_section(days, history, forecasts, TODAY)["today"]
    assert today["networks"] == ["cloud", "temp", "pressure"]
    assert len(today["pressure"]) == field(0).size and today["pressure"][0] == 1012.3
    assert {p["temp"] for p in today["places"]} == {7.0}


def test_without_their_days_temperature_comes_from_the_ridge_and_there_is_no_pressure(monkeypatch):
    days = past_days(30)
    history = {d: (field(90), field(3)) for d in days}
    forecasts = {}
    run.keep_forecasts(forecasts, days, history, {}, TODAY, {}, _others({}))
    made = forecasts[TODAY.isoformat()]
    assert not made.net and not made.temp_net and made.pressure is None
    today = run.weather_section(days, history, forecasts, TODAY)["today"]
    assert today["networks"] == [] and "pressure" not in today


def test_a_target_with_no_weights_file_keeps_the_ridge(monkeypatch, tmp_path):
    days = past_days(30)
    monkeypatch.setattr(run.cloudnet, "predict", lambda held, when, saved, target="cloud": field(40.0))
    monkeypatch.setattr(run.cloudnet, "view", lambda steps, when: np.zeros(1))
    history = {d: (field(90), field(3)) for d in days}
    forecasts = {}
    missing = {"temp": tmp_path / "none.npz", "pressure": tmp_path / "none.npz"}
    run.keep_forecasts(forecasts, days, history, {}, TODAY, {}, run.cloudnet.load_others({}, missing))
    made = forecasts[TODAY.isoformat()]
    assert made.net and not made.temp_net and made.pressure is None


def test_an_other_network_keeps_its_own_layers_over_the_shared_scales(tmp_path):
    saved = {"clim_temp": np.zeros(1, dtype=np.float32), "out.weight": np.zeros(1, dtype=np.float32)}
    np.savez(tmp_path / "t.npz", **{"out.weight": np.ones(1)})
    found = run.cloudnet.load_others(saved, {"temp": tmp_path / "t.npz"})["temp"]
    assert len(found) == 1 and found[0]["out.weight"][0] == 1 and found[0]["clim_temp"] is saved["clim_temp"]
    assert saved["out.weight"][0] == 0


def test_a_file_of_seeds_loads_one_network_per_seed(tmp_path):
    saved = {"clim_temp": np.zeros(1, dtype=np.float32)}
    np.savez(tmp_path / "t.npz", **{"seed0/out.weight": np.ones(1), "seed1/out.weight": np.full(1, 2.0)})
    found = run.cloudnet.load_others(saved, {"temp": tmp_path / "t.npz"})["temp"]
    assert [net["out.weight"][0] for net in found] == [1, 2]
    assert all(net["clim_temp"] is saved["clim_temp"] for net in found)


def test_stored_forecasts_keep_pressure_and_read_older_files(tmp_path):
    a, b = TODAY.isoformat(), (TODAY - timedelta(days=1)).isoformat()
    forecasts = {a: run.Forecast(field(1), field(2), True, True, field(1000), True), b: (field(1), field(2), False, False)}
    run.save_forecasts(tmp_path, forecasts)
    back = run.load_forecasts(tmp_path)
    assert back[a].pressure[0, 0] == 1000 and back[a].temp_net is True
    assert back[b].pressure is None and back[b].temp_net is False
    with np.load(tmp_path / "forecasts.npz") as saved:
        np.savez(tmp_path / "forecasts.npz", **{k: saved[k] for k in ("days", "cloud", "temp", "issued", "net")})
    assert run.load_forecasts(tmp_path)[a].pressure is None


def test_the_pressure_loss_reads_the_mean_of_the_day_samples():
    day, before = TODAY.isoformat(), (TODAY - timedelta(days=1)).isoformat()
    def steps_of(value):
        out = np.zeros((len(run.cloudnet.VARIABLES), 4, *SHAPE))
        out[run.cloudnet.VARIABLES.index("pressure")] = value
        return out
    forecasts = {day: run.Forecast(field(1), field(2), True, True, field(1003), True)}
    history = {day: (field(1), field(2)), before: (field(1), field(2))}
    row = run.forecast_losses(forecasts, history, {day: steps_of(1000), before: steps_of(1004)})[0]
    assert row["pressure"] == {"model": 9.0, "repeat": 16.0}
    assert "pressure" not in run.forecast_losses(forecasts, history)[0]

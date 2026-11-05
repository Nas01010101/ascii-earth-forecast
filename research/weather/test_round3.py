"""The temperature and pressure labels, forecasts and scorer, on synthetic days."""
import io
from datetime import date, timedelta

import numpy as np
import pytest

import field_labels
import fields_score
import openmeteo_split as split
import predict_fields
from common import STEPS_PER_DAY
from forecast import weather

START = date(2026, 6, 18)
JOIN = date(2026, 9, 7)
DAYS = 139  # 2026-06-18 to 2026-11-03


def steps(seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(k, 3, (DAYS * STEPS_PER_DAY, 24, 48)).astype(np.float32) for k, name in enumerate(split.FIELD_NAMES)}


def npz(arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    buffer.seek(0)
    return np.load(buffer)


def files(fields, last_dev=111):
    """dev ends on the last select day, late starts 81 days in, as the real files do."""
    temp_day = fields["temp"].reshape(DAYS, STEPS_PER_DAY, 24, 48).mean(1) + 0.5  # a 24-hour mean differs from 4 samples
    join = (JOIN - START).days
    dev = {n: f[:last_dev * STEPS_PER_DAY] for n, f in fields.items()}
    dev.update(temp_day=temp_day[:last_dev], first_date=START.isoformat())
    late = {n: f[join * STEPS_PER_DAY:] for n, f in fields.items()}
    late.update(temp_day=temp_day[join:], first_date=JOIN.isoformat())
    return npz(dev), npz(late), temp_day


def test_labels_hold_the_dates_of_the_cloud_round_and_the_right_truth():
    fields = steps()
    dev, late, temp_day = files(fields)
    found = field_labels.labels(dev, late)
    select, test = found["select_y"], found["test_y"]
    assert (str(select["days"][0]), str(select["days"][-1]), len(select["days"])) == ("2026-09-01", "2026-10-06", 36)
    assert (str(test["days"][0]), str(test["days"][-1]), len(test["days"])) == ("2026-10-07", "2026-11-03", 28)
    assert np.allclose(test["temp"], temp_day[-28:])
    pressure = fields["pressure"].reshape(DAYS, STEPS_PER_DAY, 24, 48).mean(1)
    assert np.allclose(test["pressure"], pressure[-28:])
    assert np.allclose(select["pressure"], pressure[75:111])


def test_the_ridge_label_is_the_production_forecast_from_the_60_days_before():
    fields = steps()
    dev, late, temp_day = files(fields)
    test = field_labels.labels(dev, late)["test_y"]
    day = DAYS - 1
    assert np.allclose(test["ridge_temp"][-1], weather.predict(temp_day[day - 60:day], 60)[0], atol=1e-4)


def test_files_that_disagree_are_refused():
    fields = steps()
    dev, late, _ = files(fields)
    broken = {k: late[k] for k in late.files}
    broken["pressure"] = broken["pressure"] + 1
    with pytest.raises(ValueError, match="pressure"):
        field_labels.labels(dev, npz(broken))


@pytest.mark.parametrize("name", ["repeat", "month", "ridge"])
def test_a_forecast_never_reads_the_day_it_forecasts(name):
    fields = steps()
    when = date(2026, 10, 10)
    model = predict_fields.forecaster(name, "pressure")
    days = [when.isoformat()]
    before = predict_fields.predict(model, fields, START, days)
    cut = (when - START).days * STEPS_PER_DAY
    later = {n: f.copy() for n, f in fields.items()}
    later["pressure"][cut:] += 50
    assert np.array_equal(before, predict_fields.predict(model, later, START, days))
    earlier = {n: f.copy() for n, f in fields.items()}
    earlier["pressure"][cut - 1] += 50
    assert not np.array_equal(before, predict_fields.predict(model, earlier, START, days))


def _labels(days=8):
    rng = np.random.default_rng(1)
    out = {"days": np.array([(date(2026, 9, 1) + timedelta(days=i)).isoformat() for i in range(days)])}
    for name in field_labels.TARGETS:
        out[name] = rng.normal(0, 3, (days, 24, 48))
        out[f"ridge_{name}"] = out[name] + rng.normal(0, 2, (days, 24, 48))
    return out


def test_the_ridge_scores_zero_and_the_truth_a_hundred():
    want = _labels()
    ridge = {"days": want["days"], "temp": want["ridge_temp"], "pressure": want["ridge_pressure"]}
    truth = {"days": want["days"], "temp": want["temp"], "pressure": want["pressure"]}
    assert np.allclose(fields_score.scores(ridge, want)["temp"], 0)
    assert np.allclose(fields_score.scores(truth, want)["pressure"], 100)
    assert len(fields_score.scores(truth, want)["temp"]) == 2


def test_the_printed_score_is_the_mean_of_the_two_fields(tmp_path, capsys):
    want = _labels()
    half = {"days": want["days"], "temp": want["temp"], "pressure": want["ridge_pressure"]}
    np.savez(tmp_path / "got.npz", **half)
    np.savez(tmp_path / "want.npz", **want)
    fields_score.main(str(tmp_path / "got.npz"), str(tmp_path / "want.npz"))
    assert np.allclose([float(s) for s in capsys.readouterr().out.strip().split(",")], 50)


def test_a_score_refuses_other_days_or_a_broken_forecast():
    want = _labels()
    broken = {"days": want["days"], "temp": want["temp"] * np.nan, "pressure": want["pressure"]}
    with pytest.raises(ValueError, match="temp"):
        fields_score.scores(broken, want)
    with pytest.raises(ValueError, match="days"):
        fields_score.scores({**broken, "days": want["days"][:-1]}, want)


def test_a_network_for_another_target_is_refused(monkeypatch):
    class Wrong:
        class spec:
            target = "cloud"
    monkeypatch.setattr(predict_fields.candidate, "load", lambda path: Wrong())
    with pytest.raises(ValueError, match="forecasts cloud"):
        predict_fields.forecaster("net.pt", "temp")

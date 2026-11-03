import numpy as np
import pytest

from forecast import weather
from forecast.fetch import LATS, LONS

SHAPE = (len(LATS), len(LONS))


def ar_series(days=40, phi=0.8, seed=0):
    """Fields with day-to-day memory, so a forecast has real signal to find."""
    rng = np.random.default_rng(seed)
    out = np.zeros((days, *SHAPE), dtype=np.float32)
    for d in range(1, days):
        out[d] = phi * out[d - 1] + rng.normal(0, 5, SHAPE)
    return out + 50


def test_later_days_cannot_change_the_forecast():
    series = ar_series()
    day = 30
    before, _ = weather.predict(series, day)
    tampered = series.copy()
    tampered[day:] = 1e6
    after, _ = weather.predict(tampered, day)
    assert np.array_equal(before, after)


def test_model_beats_baselines_when_there_is_memory():
    series = ar_series()
    day = 35
    forecast, trailing = weather.predict(series, day)
    scores = weather.score(series[day], forecast, series[day - 1], trailing)
    assert scores["model"] < scores["persistence"]
    assert scores["model"] < scores["trailing_mean"]


def test_refuses_too_little_history():
    with pytest.raises(ValueError):
        weather.predict(ar_series(days=16), 16)


def test_mae_weights_by_area():
    a = np.zeros(SHAPE)
    polar = a.copy()
    polar[0] = 10
    equator = a.copy()
    equator[len(LATS) // 2] = 10
    assert weather.mae(a, polar) < weather.mae(a, equator)

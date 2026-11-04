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


def test_published_weights_minimise_the_published_loss():
    series = ar_series()
    _, anomaly, fits = weather.fit(series, 30)
    x, y, area = weather.training_rows(anomaly, weather.BANDS[1])
    objective = lambda w: (area * (y - x @ w) ** 2).sum() + weather.RIDGE * w @ w
    w = fits[1]["weights"]
    assert objective(w) == pytest.approx(fits[1]["loss"])
    for nudge in np.eye(len(w)) * 1e-3:
        assert objective(w) < objective(w + nudge)
        assert objective(w) < objective(w - nudge)


def test_worked_cells_match_the_forecast():
    series = ar_series()
    day = 30
    forecast, _ = weather.predict(series, day)
    for band in weather.explain(series, day, (0, 100)):
        cell = band["cell"]
        i, j = np.abs(LATS - cell["lat"]).argmin(), np.abs(LONS - cell["lon"]).argmin()
        assert cell["predicted"] == pytest.approx(np.clip(forecast[i, j], 0, 100), abs=0.051)
        assert cell["actual"] == pytest.approx(series[day, i, j], abs=0.051)
        assert cell["mean"] + cell["change"] == pytest.approx(forecast[i, j], abs=0.11)
        assert len(cell["inputs"]) == len(band["weights"]) == len(weather.FEATURES)


def test_worked_cell_sits_in_its_band():
    for name, band, (_, lat, _) in zip(weather.BAND_NAMES, weather.BANDS, weather.CELLS):
        assert band[np.abs(LATS - lat).argmin()], name


def test_places_read_their_own_cell():
    cloud, temp = np.zeros(SHAPE), np.zeros(SHAPE)
    for k, (_, lat, lon) in enumerate(weather.CELLS, start=1):
        i, j = np.abs(LATS - lat).argmin(), np.abs(LONS - lon).argmin()
        cloud[i, j], temp[i, j] = 10 * k, -k
    assert weather.at_places(cloud, temp) == [
        {"place": place, "cloud": 10.0 * k, "temp": -1.0 * k}
        for k, (place, _, _) in enumerate(weather.CELLS, start=1)
    ]


def test_mse_weights_each_row_by_its_area():
    actual = np.zeros(SHAPE)
    forecast = np.zeros(SHAPE)
    forecast[weather.LATS.size // 2] = 10
    share = float(weather.AREA[weather.LATS.size // 2, 0] / weather.AREA.sum())
    assert weather.mse(actual, forecast) == pytest.approx(100 * share)

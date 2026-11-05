from datetime import date, timedelta

import numpy as np
import pytest

from forecast import cloudnet, fetch

WHEN = date(2026, 11, 3)
SHAPE = (len(fetch.LATS), len(fetch.LONS))


@pytest.fixture(scope="module")
def saved():
    return cloudnet.load()


def held(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (len(cloudnet.VARIABLES), cloudnet.DAYS_NEEDED * cloudnet.STEPS_PER_DAY, *SHAPE)).astype(np.float32)


def typical(saved, seed=0):
    """Fields near each variable's climatology, as a real day would be."""
    scale = np.stack([saved[f"std_{n}"] * np.ones(1) for n in cloudnet.VARIABLES])[:, :, None, None]
    start = cloudnet._doy(WHEN - timedelta(days=cloudnet.DAYS_NEEDED))
    means = np.stack([saved[f"clim_{n}"][(start + np.arange(cloudnet.DAYS_NEEDED).repeat(4)) % 366] for n in cloudnet.VARIABLES])
    return means + held(seed) * scale.astype(np.float32)


def test_the_shipped_file_makes_a_finite_map_in_a_sane_range(saved):
    forecast = cloudnet.predict(typical(saved), WHEN, saved)
    assert forecast.shape == SHAPE
    assert np.isfinite(forecast).all()
    # Clipping to 0..100 is the caller's, so this allows some overshoot and catches a wrong scale.
    assert forecast.min() > -60 and forecast.max() < 160


def test_a_changed_input_changes_the_forecast(saved):
    assert not np.array_equal(cloudnet.predict(typical(saved, 1), WHEN, saved), cloudnet.predict(typical(saved, 2), WHEN, saved))


def test_the_network_starts_from_the_trailing_base(saved):
    """Its output is a change from the base forecast, so a forecast stays near the base rather than on its own."""
    x, base = cloudnet.inputs(typical(saved), WHEN, saved)
    change = cloudnet.predict(typical(saved), WHEN, saved) - base
    assert 0 < np.abs(change).mean() < 50
    assert x.shape == (58, *SHAPE)


def test_view_reads_exactly_the_30_days_before_the_target_oldest_first():
    days = [(WHEN - timedelta(days=b)).isoformat() for b in range(cloudnet.DAYS_NEEDED, 0, -1)]
    steps = {d: np.full((7, 4, 1, 1), i, dtype=np.float32) for i, d in enumerate(days)}
    steps[WHEN.isoformat()] = np.full((7, 4, 1, 1), 999, dtype=np.float32)
    found = cloudnet.view(steps, WHEN)
    assert found.shape == (7, 120, 1, 1)
    assert found[0, 0, 0, 0] == 0 and found[0, -1, 0, 0] == 29


def test_view_refuses_a_gap():
    days = [(WHEN - timedelta(days=b)).isoformat() for b in range(cloudnet.DAYS_NEEDED, 0, -1)]
    steps = {d: np.zeros((7, 4, 1, 1)) for d in days}
    del steps[days[10]]
    assert cloudnet.view(steps, WHEN) is None


def test_day_steps_sum_rain_and_turn_wind_into_where_it_goes():
    hours = np.datetime64("2026-11-01T00", "h") + np.arange(72)
    zero = np.zeros((72, 1, 1), dtype=np.float32)
    fields = {n: zero.copy() for n in ("cloud", "temp", "precip", "pressure", "speed", "direction", "water")}
    fields["precip"][24 - 5 : 24 + 1] = 1  # the 6 hours ending at 00 UTC on the 2nd
    fields["speed"][:] = 10
    fields["direction"][:] = 270  # from the west, so blowing east
    found = cloudnet.day_steps(hours, fields, date(2026, 11, 2))
    names = list(cloudnet.VARIABLES)
    assert found.shape == (7, 4, 1, 1)
    assert found[names.index("precip"), 0, 0, 0] == 6 and found[names.index("precip"), 1, 0, 0] == 0
    assert found[names.index("u"), 0, 0, 0] == pytest.approx(10)
    assert found[names.index("v"), 0, 0, 0] == pytest.approx(0, abs=1e-5)


def test_day_steps_refuse_a_day_whose_hours_are_not_held():
    hours = np.datetime64("2026-11-02T00", "h") + np.arange(48)
    fields = {n: np.zeros((48, 1, 1), dtype=np.float32) for n in ("cloud", "temp", "precip", "pressure", "speed", "direction", "water")}
    assert cloudnet.day_steps(hours, fields, date(2026, 11, 2)) is None  # no hours from the day before for rain
    assert cloudnet.day_steps(hours, fields, date(2026, 11, 3)) is not None
    assert cloudnet.day_steps(hours, fields, date(2026, 11, 4)) is None

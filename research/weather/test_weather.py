"""The judge and the file builders, on synthetic fields."""
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import candidate
import common
import fetch_era5
import replay_days
import score
import split
from forecast import weather


def noise(days=120, seed=1):
    rng = np.random.default_rng(seed)
    return np.float32(50 + 30 * rng.standard_normal((days * common.STEPS_PER_DAY, 24, 48)).clip(-1.5, 1.5))


def sample(days=120, first="2020-11-02", target="2021-01-01"):
    data = {name: noise(days) for name in common.FIELDS}
    data.update(first_date=np.array(first), first_target=np.array(target))
    return data


def test_bilinear_keeps_a_constant_and_a_slope():
    latitude = np.arange(-90, 90.1, 1.5)
    longitude = np.arange(0, 360, 1.5)
    where = fetch_era5.weights(latitude, longitude)
    assert np.allclose(fetch_era5.to_points(np.full((240, 121), 7.0), where), 7.0)
    slope = np.broadcast_to(latitude, (240, 121))
    assert np.allclose(fetch_era5.to_points(slope, where)[:, 0], fetch_era5.LATS, atol=1e-4)


def test_points_land_on_their_longitude_past_the_date_line():
    latitude = np.arange(-90, 90.1, 1.5)
    longitude = np.arange(0, 360, 1.5)
    where = fetch_era5.weights(latitude, longitude)
    field = np.broadcast_to(np.sin(np.radians(longitude))[:, None], (240, 121))
    assert np.allclose(fetch_era5.to_points(field, where)[12], np.sin(np.radians(fetch_era5.LONS)), atol=1e-3)


def test_the_store_grid_rows_run_south_to_north():
    assert fetch_era5.LATS[0] < 0 < fetch_era5.LATS[-1]
    assert weather.LATS[0] == fetch_era5.LATS[0] and weather.LONS[0] == fetch_era5.LONS[0]


def test_daily_refuses_a_partial_day():
    assert common.daily(np.arange(8.0)).tolist() == [1.5, 5.5]
    with pytest.raises(AssertionError):
        common.daily(np.arange(6.0))


def test_the_ridge_scores_zero_against_itself():
    found = score.losses(candidate.Ridge(), sample())
    assert score.percent_lower(found["2021Q1"]) == pytest.approx(0.0)


def test_the_ridge_matches_production():
    data = sample()
    series = common.daily(data["cloud"])
    ours = candidate.Ridge().cloud({"cloud": data["cloud"][:90 * common.STEPS_PER_DAY]}, date(2021, 1, 31))
    assert np.allclose(ours, weather.predict(series[30:90], 60)[0])


def test_a_candidate_never_sees_its_target_day():
    class Spy:
        def cloud(self, fields, when):
            first = date(2020, 11, 2)
            seen.append((when - first).days * common.STEPS_PER_DAY == len(fields["cloud"]))
            return np.zeros((24, 48))
    seen = []
    score.losses(Spy(), sample())
    assert seen and all(seen)


def test_a_perfect_forecast_scores_one_hundred():
    data = sample()
    truth = common.daily(data["cloud"])

    class Oracle:
        def cloud(self, fields, when):
            return truth[len(fields["cloud"]) // common.STEPS_PER_DAY]
    assert score.percent_lower(score.losses(Oracle(), data)["2021Q1"]) == pytest.approx(100.0)


def test_each_quarter_is_one_part():
    found = score.losses(candidate.Repeat(), sample(days=200))
    assert sorted(found) == ["2021Q1", "2021Q2"]
    assert len(found["2021Q1"]) == 90


def test_split_cuts_where_it_says_and_removes_the_holdout_years(tmp_path):
    for year in range(2018, 2023):
        for store, _, _ in common.FIELDS.values():
            days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
            start = (date(year, 1, 1) - date(2018, 1, 1)).days
            np.save(tmp_path / f"{store}_{year}.npy", np.arange(start, start + days, dtype=np.float32).repeat(4).reshape(-1, 1, 1))
    split.main(str(tmp_path), str(tmp_path / "v.npz"), str(tmp_path / "h.npz"))
    val, held = np.load(tmp_path / "v.npz"), np.load(tmp_path / "h.npz")
    assert str(val["first_date"]) == "2018-11-02" and str(val["first_target"]) == "2019-01-01"
    assert str(held["first_date"]) == "2020-11-02" and str(held["first_target"]) == "2021-01-01"
    first = (date(2020, 11, 2) - date(2018, 1, 1)).days
    assert held["u"][0, 0, 0] == first and held["u"].shape[0] == ((date(2023, 1, 1) - date(2020, 11, 2)).days) * 4
    assert (val["u"].shape[0]) == ((date(2021, 1, 1) - date(2018, 11, 2)).days) * 4
    assert not list(Path(tmp_path).glob("*_2021.npy")) and not list(Path(tmp_path).glob("*_2022.npy"))
    assert list(Path(tmp_path).glob("*_2020.npy"))


def test_units():
    assert common.FIELDS["temp"][2] == -273.15 and common.FIELDS["cloud"][1] == 100


def test_replay_scores_ridge_and_repeat():
    cloud = common.daily(noise(40))
    days = [f"2026-10-{d:02d}" for d in range(1, 32)] + [f"2026-11-{d:02d}" for d in range(1, 10)]
    found = replay_days.replay(candidate.Repeat(), cloud, cloud, days, 5)
    assert found["model"] == found["repeat"] and found["ridge"] != found["repeat"]

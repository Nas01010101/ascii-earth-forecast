"""The Open-Meteo download's conversion to ERA5's form, and the replay that reads it."""
import ast
from datetime import date, timedelta

import numpy as np
import pytest

import candidate
import fetch_openmeteo as om
import nwp_loss
import replay_days
from forecast import fetch

DAYS = 3
SHAPE = (DAYS * 24, 24, 48)


def _flat(*values):
    return np.stack([np.full((24, 48), value, np.float32) for value in values])


def test_nwp_loss_holds_cloud_to_its_range_and_means_the_days():
    got = nwp_loss.losses(_flat(50, 50), _flat(0, 0), _flat(150, 40), _flat(2, 2))
    assert got["cloud"] == pytest.approx((50 ** 2 + 10 ** 2) / 2) and got["temp"] == pytest.approx(4)


def test_nwp_loss_scores_the_last_days_and_refuses_files_that_start_apart(tmp_path, capsys):
    history = tmp_path / "history.npz"
    np.savez(history, days=np.array(["2026-01-01", "2026-01-02", "2026-01-03"]), cloud=_flat(0, 50, 50), temp=_flat(0, 0, 0))
    nwp = tmp_path / "nwp.npz"
    np.savez(nwp, first_date="2026-01-02", cloud_day=_flat(60, 60), temp_day=_flat(1, 1))
    nwp_loss.main(str(history), str(nwp), "2")
    assert ast.literal_eval(capsys.readouterr().out)["cloud"] == pytest.approx(100)
    with pytest.raises(SystemExit, match="stored days start 2026-01-01"):
        nwp_loss.main(str(history), str(nwp), "3")


def hours(**values):
    out = {name: np.full(SHAPE, 1.0, np.float32) for name in om.HOURLY}
    out.update(values)
    return out


def test_steps_start_at_00_utc_of_the_day_after_the_first_hours():
    cloud = np.arange(SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(SHAPE, np.float32)
    got = om.six_hourly(hours(cloud=cloud))["cloud"]
    assert got.shape == ((DAYS - 1) * 4, 24, 48)
    assert got[0, 0, 0] == 24 and got[1, 0, 0] == 30 and got[-1, 0, 0] == 66


def test_precipitation_sums_the_six_hours_ending_at_the_step():
    rain = np.zeros(SHAPE, np.float32)
    rain[19:25] = 1.0  # the six hours ending at step 24
    rain[25:31] = 2.0  # the six hours ending at step 30
    got = om.six_hourly(hours(precip=rain))["precip"]
    assert got[0, 3, 3] == pytest.approx(6.0) and got[1, 3, 3] == pytest.approx(12.0) and got[2, 3, 3] == 0


@pytest.mark.parametrize("direction, u, v", [(270, 5, 0), (90, -5, 0), (180, 0, 5), (0, 0, -5)])
def test_wind_from_a_direction_blows_toward_the_opposite_one(direction, u, v):
    got = om.six_hourly(hours(speed=np.full(SHAPE, 5.0, np.float32), direction=np.full(SHAPE, direction, np.float32)))
    assert got["u"][0, 0, 0] == pytest.approx(u, abs=1e-4) and got["v"][0, 0, 0] == pytest.approx(v, abs=1e-4)


def test_the_seven_fields_are_the_judges():
    from common import FIELDS
    assert set(om.six_hourly(hours())) == set(FIELDS)


def payloads(value=1.0, count=24 * 48, length=48):
    return [{"hourly": {"cloud_cover": [value] * length}} for _ in range(count)]


def test_a_null_hour_stops_the_run():
    rows = payloads()
    rows[7]["hourly"]["cloud_cover"][3] = None
    with pytest.raises(fetch.SourceError):
        om.to_grid(rows, {"cloud": "cloud_cover"}, date(2026, 1, 1), date(2026, 1, 2))


def test_a_short_series_and_a_missing_location_stop_the_run():
    for rows in (payloads(length=47), payloads(count=100)):
        with pytest.raises((fetch.SourceError, ValueError)):
            om.to_grid(rows, {"cloud": "cloud_cover"}, date(2026, 1, 1), date(2026, 1, 2))


def test_the_grid_keeps_the_cell_order():
    rows = [{"hourly": {"cloud_cover": [float(i)] * 24}} for i in range(24 * 48)]
    grid = om.to_grid(rows, {"cloud": "cloud_cover"}, date(2026, 1, 1), date(2026, 1, 1))["cloud"]
    assert grid[5, 1, 2] == 48 + 2


class Spy:
    """Records what a model reads and predicts its last value."""

    def __init__(self):
        self.ends = []

    def cloud(self, view, when):
        self.ends.append(len(view["cloud"]))
        return np.float32(view["cloud"][-1])


def stored(days=40, seed=2):
    rng = np.random.default_rng(seed)
    cloud = np.float32(50 + 10 * rng.standard_normal((days, 24, 48)))
    temp = np.float32(10 + 5 * rng.standard_normal((days, 24, 48)))
    names = [(date(2026, 1, 1) + timedelta(days=d)).isoformat() for d in range(days)]
    return cloud, temp, names


def test_six_hourly_replay_reads_only_steps_before_the_target_with_an_offset_start():
    cloud, temp, names = stored()
    # The fetched file starts 5 days after the stored one.
    fields = {"cloud": np.full((35 * 4, 24, 48), 40.0, np.float32), "first_date": "2026-01-06", "first_target": "2026-02-05"}

    class Files(dict):
        files = list(fields)

    spy = Spy()
    replay_days.replay(spy, cloud, temp, names, 5, Files(fields))
    assert spy.ends == [(day - 5) * 4 for day in range(35, 40)]


def test_a_target_day_change_does_not_move_the_forecast():
    cloud, temp, names = stored()
    fields = {"cloud": np.full((35 * 4, 24, 48), 40.0, np.float32), "first_date": "2026-01-06"}

    class Files(dict):
        files = list(fields)

    first = replay_days.replay(Spy(), cloud, temp, names, 1, Files(fields))
    fields["cloud"][34 * 4:] = 999.0
    assert replay_days.replay(Spy(), cloud, temp, names, 1, Files(fields)) == first


def test_held_replay_still_matches_the_ridge_on_itself():
    cloud, temp, names = stored()
    got = replay_days.replay(candidate.Ridge(), cloud, temp, names, 5)
    assert got["model"] == pytest.approx(got["ridge"])

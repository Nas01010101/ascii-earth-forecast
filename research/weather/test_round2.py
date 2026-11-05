"""The split and the scorer for candidates fitted on Open-Meteo days."""
import argparse
from datetime import date, timedelta

import numpy as np
import pytest
import torch

import features
import fetch_openmeteo as om
import fine_tune
import nets
import openmeteo_split as split
import parts_score
import predict_net
import rescale_stats
import stack
from common import STEPS_PER_DAY
from forecast import weather

GRID = (24, 48)


def _fields(first, days, seed):
    rng = np.random.default_rng(seed)
    out = {name: rng.random((days * STEPS_PER_DAY, *GRID), np.float32) * 100 for name in split.FIELD_NAMES}
    out.update(first_date=first.isoformat(), cloud_day=rng.random((days, *GRID), np.float32) * 100,
               temp_day=rng.random((days, *GRID), np.float32))
    return out


def _sources():
    early = _fields(date(2026, 6, 18), 81, 0)
    late = _fields(date(2026, 9, 7), 58, 1)
    rng = np.random.default_rng(2)
    days = np.array([(date(2026, 9, 20) + timedelta(days=i)).isoformat() for i in range(45)])
    return early, late, {"days": days, "cloud": rng.random((45, *GRID), np.float32) * 100, "temp": rng.random((45, *GRID), np.float32)}


def test_dev_stops_before_the_first_test_day():
    got = split.split(*_sources())
    assert got["dev"]["cloud_day"].shape[0] == (split.FIRST_TEST - date(2026, 6, 18)).days
    assert got["dev"]["cloud"].shape[0] == got["dev"]["cloud_day"].shape[0] * STEPS_PER_DAY
    assert got["select_y"]["days"][-1] == (split.FIRST_TEST - timedelta(days=1)).isoformat()
    assert got["select_y"]["days"][0] == split.FIRST_SELECT.isoformat()


def test_no_test_day_reaches_the_dev_file():
    early, late, history = _sources()
    before = split.split(early, late, history)["dev"]["cloud"].copy()
    keep = (split.FIRST_TEST - date(2026, 9, 7)).days * STEPS_PER_DAY
    changed = dict(late, cloud=late["cloud"].copy())
    changed["cloud"][keep:] = -1
    after = split.split(early, changed, history)["dev"]["cloud"]
    assert np.array_equal(before, after)


def test_the_test_labels_are_the_stored_days():
    early, late, history = _sources()
    got = split.split(early, late, history)["test_y"]
    assert got["days"][0] == "2026-10-07" and len(got["days"]) == 28
    assert np.array_equal(got["cloud"], history["cloud"][-28:])


def test_files_that_do_not_touch_are_refused():
    early, late, history = _sources()
    with pytest.raises(ValueError, match="touch"):
        split.split(early, dict(late, first_date="2026-09-08"), history)


def test_stored_days_that_start_elsewhere_are_refused():
    early, late, history = _sources()
    with pytest.raises(ValueError, match="stored test days"):
        split.split(early, late, dict(history, days=np.roll(history["days"], 1)))


def test_inputs_stop_before_the_target_day():
    fields = _fields(date(2026, 9, 7), 58, 1)
    got = split.inputs_of(fields, date(2026, 9, 7), date(2026, 10, 7))
    assert got["cloud"].shape[0] == 30 * STEPS_PER_DAY
    assert np.array_equal(got["cloud"][-1], fields["cloud"][30 * STEPS_PER_DAY - 1])
    with pytest.raises(ValueError):
        split.inputs_of(fields, date(2026, 9, 7), date(2026, 9, 7))


def test_a_ridge_forecast_ignores_its_own_day_and_later():
    cloud = np.random.default_rng(3).random((45, *GRID), np.float32) * 100
    base = split.walk_forward_ridge(cloud, 3)
    changed = cloud.copy()
    changed[-1] = 0
    moved = split.walk_forward_ridge(changed, 3)
    assert np.array_equal(base[:3], moved[:3])
    changed[-2] = 0
    assert np.array_equal(base[:2], split.walk_forward_ridge(changed, 3)[:2])


def test_parts_are_disjoint_and_score_against_the_ridge():
    truth = np.full((8, *GRID), 50.0, np.float32)
    ridge = truth + 10
    half = truth + 5  # a quarter of the ridge's squared error
    scores = parts_score.part_scores(half, truth, ridge)
    assert len(scores) == 2 and scores == pytest.approx([75.0, 75.0])
    worse = half.copy()
    worse[4:] = truth[4:] + 20  # only the second part changes
    assert parts_score.part_scores(worse, truth, ridge) == pytest.approx([75.0, -300.0])


def test_part_scores_clip_to_the_cloud_range():
    truth = np.full((4, *GRID), 100.0, np.float32)
    ridge = truth - 10
    assert parts_score.part_scores(truth + 50, truth, ridge) == pytest.approx([100.0])
    assert weather.mse(truth, truth) == 0


def test_a_forecast_reads_only_steps_before_its_day():
    fields = _fields(date(2026, 9, 7), 58, 1)
    model = predict_net.Month()
    days = ["2026-10-07", "2026-10-08"]
    base = predict_net.predict(model, fields, date(2026, 9, 7), days)
    changed = {**fields, "cloud": fields["cloud"].copy()}
    changed["cloud"][31 * STEPS_PER_DAY:] = 0  # the second target day and later
    assert np.array_equal(base, predict_net.predict(model, changed, date(2026, 9, 7), days))
    changed["cloud"][30 * STEPS_PER_DAY:] = 0  # now the first target day's own steps
    assert np.array_equal(base[0], predict_net.predict(model, changed, date(2026, 9, 7), days)[0])
    assert not np.array_equal(base[1], predict_net.predict(model, changed, date(2026, 9, 7), days)[1])


def test_the_month_is_the_mean_of_the_last_30_days():
    steps = np.zeros((40 * STEPS_PER_DAY, *GRID), np.float32)
    steps[-30 * STEPS_PER_DAY:] = 60
    got = predict_net.Month().cloud({"cloud": steps}, None)
    assert np.allclose(got, 60)


def test_the_day_prefix_reads_daily_means_only_before_the_target(tmp_path):
    fields = _fields(date(2026, 9, 7), 58, 1)
    path, out = tmp_path / "f.npz", tmp_path / "p.npz"
    np.savez(path, **fields)
    predict_net.main("day:month", str(path), "2026-10-07", "2026-10-07", str(out))
    want = fields["cloud_day"][:30].mean(axis=0)
    assert np.allclose(np.load(out)["cloud"][0], want)
    fields["cloud_day"][30:] = 0  # the target day and later
    np.savez(path, **fields)
    predict_net.main("day:month", str(path), "2026-10-07", "2026-10-07", str(out))
    assert np.allclose(np.load(out)["cloud"][0], want)


def test_the_mean_of_forecasts_refuses_other_days(tmp_path):
    a, b, out = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "m.npz"
    np.savez(a, days=np.array(["x", "y"]), cloud=np.full((2, *GRID), 10.0))
    np.savez(b, days=np.array(["x", "y"]), cloud=np.full((2, *GRID), 30.0))
    stack.mean(str(out), str(a), str(b))
    assert np.allclose(np.load(out)["cloud"], 20) and list(np.load(out)["days"]) == ["x", "y"]
    np.savez(b, days=np.array(["y", "x"]), cloud=np.full((2, *GRID), 30.0))
    with pytest.raises(ValueError):
        stack.mean(str(out), str(a), str(b))


def _tune_setup(pull=0.0, epochs=3, part="all", lr=1e-3):
    spec = features.Spec(("cloud", "temp"), 4, 3)
    fields = _fields(date(2026, 7, 1), 50, 7)
    stats = {f"clim_{n}": np.full((366, *GRID), 50.0, np.float32) for n in ("cloud", "temp")}
    stats.update(std_cloud=np.float32(30), std_temp=np.float32(30), land_sea_mask=np.zeros(GRID), geopotential_at_surface=np.zeros(GRID))
    frames = features.Frames(fields, date(2026, 7, 1), spec, stats)
    torch.manual_seed(0)
    net = nets.UNet(spec.channels, 4, 2)
    truth = torch.from_numpy(fields["cloud_day"])
    args = argparse.Namespace(seed=0, part=part, lr=lr, pull=pull, epochs=epochs)
    return net, frames, truth, args


def test_fine_tuning_lowers_the_fit_loss_and_keeps_the_best_select_epoch():
    net, frames, truth, args = _tune_setup()
    state, best, rows = fine_tune.tune(net, frames, truth, np.arange(30, 42), np.arange(42, 50), args)
    assert rows[-1][1] < rows[0][1]
    assert best == min(row[2] for row in rows)
    net.load_state_dict(state)
    assert fine_tune.loss_on(net, frames, truth, np.arange(42, 50)) == pytest.approx(best)


def test_fine_tuning_does_not_read_the_truth_of_select_days_for_its_steps():
    net, frames, truth, args = _tune_setup(epochs=1)
    fine_tune.tune(net, frames, truth, np.arange(30, 42), np.arange(42, 50), args)
    moved = [p.detach().clone() for p in net.parameters()]
    other, frames2, truth2, args2 = _tune_setup(epochs=1)
    truth2[42:] = 0
    fine_tune.tune(other, frames2, truth2, np.arange(30, 42), np.arange(42, 50), args2)
    assert all(torch.equal(a, b) for a, b in zip(moved, other.parameters()))


def test_a_large_pull_keeps_the_weights_near_the_start_and_head_freezes_the_rest():
    net, frames, truth, args = _tune_setup(pull=1e9, lr=1e-6)
    start = [p.detach().clone() for p in net.parameters()]
    fine_tune.tune(net, frames, truth, np.arange(30, 42), np.arange(42, 50), args)
    assert sum(float(((a - b) ** 2).sum()) for a, b in zip(start, net.parameters())) < 1e-4
    net, frames, truth, args = _tune_setup(part="head")
    start = {name: p.detach().clone() for name, p in net.named_parameters()}
    fine_tune.tune(net, frames, truth, np.arange(30, 42), np.arange(42, 50), args)
    for name, p in net.named_parameters():
        assert torch.equal(start[name], p) != name.startswith(fine_tune.HEAD)


def _stack_data(seed=4, days=60):
    rng = np.random.default_rng(seed)
    a, b = rng.random((days, *GRID)) * 100, rng.random((days, *GRID)) * 100
    x = np.stack([a, b, np.full_like(a, stack.INTERCEPT)], axis=-1)
    return x, a, b


def test_a_large_strength_gives_the_global_weights_on_every_row():
    x, a, b = _stack_data()
    coef = stack.fit_rows(x, 0.25 * a + 0.75 * b, 1e12)
    assert np.allclose(coef, coef[0], atol=1e-6)
    assert coef[0] == pytest.approx([0.25, 0.75, 0.0], abs=1e-3)


def test_a_small_strength_recovers_each_rows_own_weights():
    x, a, b = _stack_data()
    share = np.linspace(0, 1, 24)[None, :, None]
    y = share * a + (1 - share) * b + 5
    coef = stack.fit_rows(x, y, 1e-6)
    assert coef[:, 0] == pytest.approx(share.ravel(), abs=1e-4)
    assert coef[:, 2] == pytest.approx(np.full(24, 5 / stack.INTERCEPT), abs=1e-4)


def test_a_row_with_little_data_stays_near_the_global_weights():
    x, a, b = _stack_data(days=3)
    y = 0.25 * a + 0.75 * b
    y[:, 0] = a[:, 0]  # one row disagrees with the rest
    coef = stack.fit_rows(x, y, 1e8)
    assert coef[1] == pytest.approx(coef[2], abs=1e-3)
    assert coef[0][0] == pytest.approx(coef[1][0], abs=0.1)


def test_applying_weights_clips_and_uses_each_row():
    x, a, b = _stack_data(days=2)
    coef = np.zeros((24, 3))
    coef[0, 0] = 1
    coef[1, 2] = 300 / stack.INTERCEPT
    got = stack.apply_rows(coef, x)
    assert np.allclose(got[:, 0], a[:, 0]) and np.allclose(got[:, 1], 100) and np.allclose(got[:, 2:], 0)


def test_smoothing_wraps_longitude_and_keeps_a_flat_field():
    assert np.allclose(stack.smooth(np.full((1, *GRID), 30.0)), 30)
    spot = np.zeros((1, *GRID))
    spot[0, 10, 0] = 9
    got = stack.smooth(spot)[0]
    assert got[10, 47] == 1 and got[10, 1] == 1 and got[9, 0] == 1 and got[11, 0] == 1 and got.sum() == pytest.approx(9)


def test_the_fit_takes_truth_from_its_own_days(tmp_path, capsys):
    days = split.days_from(date(2026, 7, 18), 12)
    rng = np.random.default_rng(5)
    feature = rng.random((12, *GRID), np.float32) * 100
    truth = 0.5 * feature + 10
    lead = np.zeros((2, *GRID), np.float32)
    np.savez(tmp_path / "dev.npz", first_date="2026-07-16", cloud_day=np.concatenate([lead, truth]))
    np.savez(tmp_path / "f.npz", days=np.array(days), cloud=feature)
    out = tmp_path / "coef.npz"
    stack.fit(str(tmp_path / "dev.npz"), days[0], days[7], str(out), "1,1000", days[8], days[11], str(tmp_path / "f.npz"))
    held = np.load(out)
    assert list(held["days"]) == days[8:]
    assert np.allclose(held["cloud"], truth[8:], atol=0.05)
    assert "kept strength" in capsys.readouterr().out
    stack.apply(str(out), days[0], days[1], str(tmp_path / "p.npz"), str(tmp_path / "f.npz"))
    assert np.allclose(np.load(tmp_path / "p.npz")["cloud"], truth[:2], atol=0.05)


def test_the_lagged_file_holds_both_forecasts_as_daily_means(tmp_path, monkeypatch):
    hours = np.arange(48 * 24 * 48, dtype=np.float32).reshape(48, 24, 48) % 100
    monkeypatch.setattr(om, "download", lambda url, names, first, last, cache: {name: hours + i for i, name in enumerate(names)})
    out = tmp_path / "lag.npz"
    om.lagged(str(out), "2026-07-18", "2026-07-19", str(tmp_path))
    held = np.load(out)
    assert str(held["first_date"]) == "2026-07-18"
    assert held["cloud2_day"].shape == (2, *GRID)
    assert np.allclose(held["cloud_day"][1], hours[24:].mean(0))
    assert np.allclose(held["cloud2_day"][0] - held["cloud_day"][0], list(om.LAGGED).index("cloud2") - list(om.LAGGED).index("cloud"))


def test_rescaled_stats_give_inputs_mean_zero_and_spread_one():
    rng = np.random.default_rng(6)
    first = date(2026, 7, 18)
    clim = rng.random((366, *GRID), np.float32) * 10
    doys = rescale_stats.step_days(first, 20)
    held = clim[doys] + 3 + 2 * rng.standard_normal((20 * STEPS_PER_DAY, *GRID), np.float32)
    stats = {"clim_cloud": clim, "std_cloud": np.float32(2.0)}
    got = rescale_stats.rescaled(stats, {"cloud": held}, first, 5, 20, ("cloud",))
    scaled = (held[5 * STEPS_PER_DAY:] - got["clim_cloud"][doys[5 * STEPS_PER_DAY:]]) / float(got["std_cloud"])
    assert scaled.mean() == pytest.approx(0, abs=1e-4) and scaled.std() == pytest.approx(1, abs=1e-4)
    assert float(stats["std_cloud"]) == 2.0  # the source stats stay as they were


def test_the_scorer_refuses_other_days(tmp_path, capsys):
    labels, pred = tmp_path / "y.npz", tmp_path / "p.npz"
    truth = np.full((4, *GRID), 50.0, np.float32)
    days = np.array(["a", "b", "c", "d"])
    np.savez(labels, days=days, cloud=truth, ridge=truth + 10)
    np.savez(pred, days=days, cloud=truth + 5)
    parts_score.main(str(pred), str(labels))
    assert capsys.readouterr().out.strip() == "75.0000"
    np.savez(pred, days=days[::-1], cloud=truth)
    with pytest.raises(ValueError):
        parts_score.main(str(pred), str(labels))
    np.savez(pred, days=days, cloud=truth * np.nan)
    with pytest.raises(ValueError):
        parts_score.main(str(pred), str(labels))


def test_lagged_forecasts_keep_their_own_lead_and_dates():
    import lagged_features

    grid = np.ones((3, 24, 48))
    one, two = lagged_features.split({"first_date": "2026-07-18", "cloud_day": grid * 10, "cloud2_day": grid * 20})
    assert list(one["days"]) == ["2026-07-18", "2026-07-19", "2026-07-20"]
    assert list(two["days"]) == list(one["days"])
    assert one["cloud"][0, 0, 0] == 10 and two["cloud"][0, 0, 0] == 20

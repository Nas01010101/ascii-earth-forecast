"""Checks the shared transforms, the baselines and the network start, on synthetic rows."""
import argparse

import numpy as np
import pytest
import torch

from baselines import linear_walk
from build_samples import FEATURES, HISTORY_FEATURES, LOOKBACKS, WEATHER_FEATURES
from data import HISTORY_TAIL, inputs, metric_weights, own_drift
from score import group_scores
from train_net import Net, miss, tensors


def rows(n=40, c=0.5, seed=0, step=5.0, objects=7):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, len(FEATURES)), dtype=np.float32)
    x[:, FEATURES.index("no_rev")] = 15.0
    for a in LOOKBACKS:
        x[:, FEATURES.index(f"age_{int(a)}")] = a
        x[:, FEATURES.index(f"along_{int(a)}")] = c * a ** 2
    dt = rng.choice([1.0, 3.0, 7.0], n).astype(np.float32)
    y = np.zeros((n, 3), dtype=np.float32)
    y[:, 1] = c * dt ** 2
    return {"norad": np.arange(n, dtype=np.int32) % objects + 1, "epoch": np.arange(n, dtype=np.float64) * step, "horizon": dt.copy(),
            "dt": dt, "radius": np.full(n, 7000.0, np.float32), "x": x, "y": y}


def test_own_drift_recovers_a_quadratic_error():
    d = rows()
    assert own_drift(d) == pytest.approx(d["y"][:, 1], rel=1e-5)


def test_own_drift_ignores_missing_hindcasts_and_gives_zero_with_none():
    d = rows()
    for name in HISTORY_FEATURES:
        if name.endswith("_15"):
            d["x"][:, FEATURES.index(name)] = np.nan
    assert own_drift(d) == pytest.approx(d["y"][:, 1], rel=1e-5)
    d["x"][:, [FEATURES.index(n) for n in HISTORY_FEATURES]] = np.nan
    assert (own_drift(d) == 0).all()


def test_inputs_are_finite_with_missing_history_and_tail_marks_history():
    d = rows()
    full = inputs(d)
    d["x"][:, [FEATURES.index(n) for n in HISTORY_FEATURES]] = np.nan
    empty = inputs(d)
    assert np.isfinite(empty).all()
    assert (empty[:, -len(LOOKBACKS):] == 1).all() and (full[:, -len(LOOKBACKS):] == 0).all()
    assert (empty[:, -HISTORY_TAIL] == 0).all() and (full[:, -HISTORY_TAIL] != 0).all()
    # dt stays: it is known without history.
    assert (empty[:, -HISTORY_TAIL - 1] == full[:, -HISTORY_TAIL - 1]).all()


def test_no_history_removes_every_trace_of_history():
    args = argparse.Namespace(no_history=True, no_weather=False, fixed_dt=False, solar_ratio=False, max_drift=np.inf)
    a, b = rows(seed=1), rows(seed=1)
    b["x"][:, [FEATURES.index(n) for n in HISTORY_FEATURES]] *= 9.0
    b["x"][:, FEATURES.index("along_1")] = np.nan
    for left, right in zip(tensors(a, args), tensors(b, args)):
        assert torch.equal(left, right)


def test_no_weather_blanks_only_weather():
    args = argparse.Namespace(no_history=False, no_weather=True, fixed_dt=False, solar_ratio=False, max_drift=np.inf)
    a, b = rows(seed=1), rows(seed=1)
    b["x"][:, [FEATURES.index(n) for n in WEATHER_FEATURES]] = 100.0
    assert torch.equal(tensors(a, args)[0], tensors(b, args)[0])


def test_fixed_dt_removes_the_arrival_time_of_the_truth_elements():
    args = argparse.Namespace(no_history=False, no_weather=False, fixed_dt=True, solar_ratio=False, max_drift=np.inf)
    a, b = rows(), rows()
    b["dt"] = b["dt"] + 0.7
    for left, right in zip(tensors(a, args), tensors(b, args)):
        assert torch.equal(left, right)


def test_metric_weights_match_the_score_gradient_per_object():
    d = rows(n=100)
    w = metric_weights(d)
    key = d["norad"].astype(np.int64) * 100 + d["horizon"].astype(np.int64)
    raw = np.linalg.norm(d["y"], axis=1)
    totals = [float((w[key == k] * raw[key == k]).sum()) for k in np.unique(key)]
    assert np.allclose(totals, totals[0], rtol=1e-4)


def test_network_starts_at_raw_sgp4():
    d = rows()
    args = argparse.Namespace(no_history=False, no_weather=False, fixed_dt=False, solar_ratio=False, max_drift=np.inf)
    x, drift, scale = tensors(d, args)
    out = Net(x.shape[1], 3, 16, 0.0)(x, drift, scale)
    assert (out == 0).all()


def test_miss_is_zero_for_a_perfect_correction_and_has_finite_gradient():
    truth = torch.tensor([[1.0, 50.0, -2.0]])
    c = truth.clone().requires_grad_()
    m = miss(torch.tensor([7000.0]), truth, c)
    assert m.item() < 1e-2
    m.backward()
    assert torch.isfinite(c.grad).all()


def test_linear_walk_uses_only_earlier_hindcasts():
    # Six hours apart, so 30 days hold 480 hindcasts, past the 100 the fit needs.
    d = rows(n=400, step=0.25)
    first_week = d["epoch"] < d["epoch"].min() + 7.0
    c = linear_walk(d)
    assert (c[first_week] == 0).all()  # nothing known yet
    later = d["epoch"] > d["epoch"].min() + 40.0
    assert c[later, 1] == pytest.approx(d["y"][later, 1], rel=2e-2)
    # Changing a row's hindcast must not alter any correction before it is known.
    e = rows(n=400, step=0.25)
    e["x"][300:, FEATURES.index("along_1")] *= 50.0
    assert (linear_walk(e)[:300] == c[:300]).all()


def test_baseline_with_true_correction_scores_full_reduction():
    d = rows(n=2000, objects=100)
    scores = group_scores(d["norad"], np.full(2000, 15.0), d["radius"], d["y"].astype(np.float64), d["y"].astype(np.float64))
    assert scores.mean() == pytest.approx(100.0, abs=1e-6)


def test_extra_shards_never_touch_the_unseen_validation_objects():
    from train_net import TRAIN_SHARDS, VAL_SHARDS, train_shards

    unseen = VAL_SHARDS - TRAIN_SHARDS
    assert train_shards(0) == TRAIN_SHARDS
    assert train_shards(32) & unseen == set()
    assert len(train_shards(32)) == 48
    assert max(train_shards(99)) == 63


def test_ensemble_mean_is_elementwise():
    from ensemble import mean_correction

    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
    assert mean_correction([a, b]).tolist() == [[2.0, 3.0, 4.0]]


def test_save_mean_writes_the_mean_without_labels(tmp_path):
    from ensemble import main

    a, b, out = tmp_path / "a.npy", tmp_path / "b.npy", tmp_path / "m.npy"
    np.save(a, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    np.save(b, np.array([[3.0, 4.0, 5.0]], dtype=np.float32))
    main("--save-mean", str(out), str(a), str(b))
    assert np.load(out).tolist() == [[2.0, 3.0, 4.0]]
    assert np.load(out).dtype == np.float32


def test_solar_ratio_keeps_no_absolute_flux_level():
    d = rows()
    d["x"][:, FEATURES.index("f107")] = 150.0
    d["x"][:, FEATURES.index("f107_81")] = 150.0
    args = argparse.Namespace(no_weather=False, no_history=False, fixed_dt=False, solar_ratio=True, max_drift=np.inf)
    x = tensors(d, args)[0]
    assert x[:, FEATURES.index("f107")].abs().max() == 0
    assert x[:, FEATURES.index("f107_81")].abs().max() == 0


def test_inputs_are_the_same_in_chunks(monkeypatch):
    import data

    d = rows(n=50)
    whole = inputs(d)
    monkeypatch.setattr(data, "CHUNK", 7)
    assert np.array_equal(inputs(d), whole)


def test_max_drift_caps_a_diverged_hindcast_and_its_scale():
    d = rows()
    d["x"][0, FEATURES.index("along_1")] = 1e13
    args = argparse.Namespace(no_weather=False, no_history=False, fixed_dt=False, solar_ratio=False, max_drift=4e4)
    _, drift, scale = tensors(d, args)
    assert drift.abs().max() <= 4e4
    assert scale.max() <= 4e4
    assert drift[0].abs() == 4e4


def test_percent_gain_pools_error_within_an_object():
    from head_to_head import percent_gain

    assert percent_gain(np.array([10.0, 30.0]), np.array([5.0, 15.0])) == 50.0


def test_month_bootstrap_centres_on_the_gain_and_keeps_its_spread():
    from head_to_head import month_blocks, month_bootstrap, percent_gain

    epoch = np.arange(0.0, 540.0, 5.0)
    raw = 10.0 + np.sin(epoch / 40.0)
    model = raw * (0.7 + 0.1 * np.sin(epoch / 90.0))
    months = month_blocks(epoch)
    labels = np.unique(months)
    picks = np.random.default_rng(0).integers(0, len(labels), size=(500, len(labels)))
    boot = month_bootstrap(raw, model, months, labels, picks)
    assert abs(boot.mean() - percent_gain(raw, model)) < 1.0
    assert boot.std() > 0.0


def test_month_bootstrap_keeps_objects_on_shared_months():
    from head_to_head import month_blocks, month_bootstrap

    # One object starts a month later: a grid anchored on each object's first row would misalign them.
    a, b = np.arange(0.0, 360.0, 5.0), np.arange(31.0, 360.0, 5.0)
    assert month_blocks(np.array([60.0]))[0] == month_blocks(np.array([60.1]))[0]
    labels = np.unique(month_blocks(a))
    picks = np.random.default_rng(0).integers(0, len(labels), size=(500, len(labels)))
    # Both objects are bad in the same months, so their draws must move together.
    gain = lambda t: 0.5 + 0.3 * np.sin(t / 30.0)
    boot_a = month_bootstrap(np.ones_like(a), gain(a), month_blocks(a), labels, picks)
    boot_b = month_bootstrap(np.ones_like(b), gain(b), month_blocks(b), labels, picks)
    assert np.corrcoef(boot_a, boot_b)[0, 1] > 0.9


def test_drop_top_row_removes_only_each_objects_largest():
    from catalogue_ci import drop_top_row

    norad = np.array([2, 1, 2, 1, 1])
    raw = np.array([9.0, 1.0, 3.0, 7.0, 2.0])
    assert drop_top_row(norad, raw).tolist() == [False, True, True, False, True]


def test_block_interval_covers_the_gain_and_widens_with_a_bad_month():
    from catalogue_ci import block_interval

    norad = np.repeat([1, 2, 3], 120)
    months = np.tile(np.repeat(np.arange(12), 10), 3)
    raw = np.ones(360)
    steady = block_interval(norad, months, raw, raw * 0.7, draws=300)
    assert steady == pytest.approx((30.0, 30.0))
    uneven = raw * 0.7
    uneven[months == 0] = 3.0
    lo, hi = block_interval(norad, months, raw, uneven, draws=300)
    assert lo < 100.0 * (1.0 - uneven.sum() / raw.sum()) < hi and hi - lo > 5.0


def test_block_interval_skips_objects_with_too_few_rows_in_a_draw():
    from catalogue_ci import block_interval

    norad = np.r_[np.repeat(1, 60), np.repeat(2, 3)]
    months = np.r_[np.repeat(np.arange(6), 10), np.array([0, 0, 0])]
    raw = np.ones(63)
    model = np.r_[np.full(60, 0.5), np.full(3, 9.0)]
    # Object 2 never reaches MIN_SAMPLES unless block 0 is drawn twice, so most of the mass stays near 50.
    lo, _ = block_interval(norad, months, raw, model, draws=300)
    assert np.isfinite(lo)


def test_per_object_inputs_use_the_nominal_horizon_not_the_arrival():
    from per_object import all_inputs

    d = rows()
    late = {**d, "dt": d["dt"] + 0.9}
    assert np.array_equal(all_inputs(d), all_inputs(late))


def test_holdout_labels_must_line_up_with_features():
    from report_holdout import aligned_labels

    d = {"norad": np.array([1, 2]), "horizon": np.array([15.0, 15.0])}
    labels = {"norad": np.array([1, 2]), "horizon": np.array([15.0, 15.0]), "radius": np.array([7000.0, 7000.0]),
              "t_height": np.zeros(2), "t_along": np.array([1.0, 2.0]), "t_cross": np.zeros(2)}
    y, radius = aligned_labels(d, labels)
    assert y[:, 1].tolist() == [1.0, 2.0] and radius.tolist() == [7000.0, 7000.0]
    with pytest.raises(AssertionError):
        aligned_labels({**d, "norad": np.array([2, 1])}, labels)


def test_trailing_bias_ignores_truth_that_has_not_arrived():
    from calibrate import trailing_bias

    epoch = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    dt = np.full(5, 12.0)
    residual = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    bias = trailing_bias(epoch, dt, residual, half_life=1e6, prior=0.0)
    assert bias[:3].tolist() == [0.0, 0.0, 0.0]  # the first truth arrives at day 12
    assert bias[3] == pytest.approx(100.0) and bias[4] == pytest.approx(100.0)


def test_trailing_bias_does_not_read_a_later_residual():
    from calibrate import trailing_bias

    epoch = np.arange(0.0, 50.0, 5.0)
    dt = np.full(10, 7.0)
    residual = np.arange(10.0)
    a = trailing_bias(epoch, dt, residual, 30.0, 2.0)
    changed = residual.copy()
    changed[6:] = 1e6
    b = trailing_bias(epoch, dt, changed, 30.0, 2.0)
    assert np.array_equal(a[:7], b[:7])  # row 6 arrives at day 37, after the t0 of row 7 at day 35


def test_trailing_bias_limits_a_manoeuvre_and_shrinks_with_few_rows():
    from calibrate import trailing_bias

    epoch = np.arange(0.0, 60.0, 5.0)
    dt = np.full(12, 1.0)
    residual = np.ones(12)
    residual[5] = 1e6
    bias = trailing_bias(epoch, dt, residual, 1e6, 0.0)
    assert abs(bias[6]) < 10.0
    assert trailing_bias(epoch, dt, np.ones(12), 1e6, 2.0)[1] == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_calibrated_keeps_row_order_and_only_moves_along_track():
    from calibrate import calibrated

    d = {"norad": np.array([2, 1, 2, 1, 2, 1]), "horizon": np.full(6, 1.0), "dt": np.full(6, 1.0, dtype=np.float32),
         "epoch": np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0]), "y": np.zeros((6, 3), dtype=np.float32)}
    d["y"][:, 1] = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
    out = calibrated(d, np.zeros((6, 3)), 1e6, 0.0)
    assert out[:, 1].tolist() == pytest.approx([0.0, 0.0, 10.0, 20.0, 10.0, 20.0])
    assert np.all(out[:, [0, 2]] == 0.0)


def test_walk_forward_trains_only_on_truth_that_has_arrived():
    from per_object import MIN_ROWS, MONTH_DAYS, walk_forward

    n = 200
    epoch = np.arange(n, dtype=np.float64) * 5.0
    d = {"norad": np.full(n, 7), "horizon": np.full(n, 15.0), "dt": np.full(n, 15.0, dtype=np.float32), "epoch": epoch,
         "y": np.zeros((n, 1), dtype=np.float32)}
    d["y"][:, 0] = np.arange(n, dtype=np.float32)
    start = epoch[120]
    latest = {}

    class Spy:
        def fit(self, x, y):
            self.top = y.max()
            return self

        def predict(self, x):
            latest[float(x.min())] = self.top
            return np.full(len(x), self.top)

    c = walk_forward(d, epoch[:, None], Spy, start, epoch[160])
    for first_epoch, top in latest.items():
        month = start + np.floor((first_epoch - start) / MONTH_DAYS) * MONTH_DAYS
        assert epoch[int(top)] + 15.0 < month  # the newest training row had arrived before the month began
    assert MIN_ROWS <= 120
    assert np.all(c[:120] == 0.0) and np.all(c[160:] == 0.0)


def test_walk_forward_leaves_short_histories_uncorrected():
    from per_object import walk_forward

    n = 20
    d = {"norad": np.full(n, 7), "horizon": np.full(n, 1.0), "dt": np.ones(n, dtype=np.float32), "epoch": np.arange(n) * 5.0,
         "y": np.ones((n, 1), dtype=np.float32)}

    class Never:
        def fit(self, x, y):
            raise AssertionError("fitted on too few rows")

    assert np.all(walk_forward(d, np.zeros((n, 1)), Never, 50.0, 100.0) == 0.0)


def test_ridge_recovers_a_line():
    from per_object import Ridge

    x = np.arange(50.0)[:, None]
    assert Ridge(1e-6).fit(x, 3.0 * x[:, 0] + 2.0).predict(np.array([[100.0]]))[0] == pytest.approx(302.0, abs=1e-3)


def test_shuffled_folds_holds_out_each_row_and_uses_later_rows():
    from per_object import shuffled_folds

    n = 200
    epoch = np.arange(n, dtype=np.float64)
    d = {"norad": np.full(n, 7), "horizon": np.full(n, 15.0), "dt": np.full(n, 15.0, dtype=np.float32), "epoch": epoch,
         "y": np.arange(n, dtype=np.float32)[:, None]}
    seen = []

    class Spy:
        def fit(self, x, y):
            self.train = set(y.astype(int).tolist())
            return self

        def predict(self, x):
            test = set(x[:, 0].astype(int).tolist())
            assert not test & self.train  # a row never trains its own prediction
            seen.append((test, self.train))
            return np.full(len(x), 1.0)

    c = shuffled_folds(d, epoch[:, None], Spy, 100.0, 150.0)
    assert np.all(c[100:150] == 1.0) and np.all(c[:100] == 0.0) and np.all(c[150:] == 0.0)
    assert all(max(train) > max(test) for test, train in seen)  # later rows are in the training set


def test_daily_view_keeps_the_newest_set_before_each_download():
    from daily_samples import SNAPSHOT_HOUR, daily_view

    cut = SNAPSHOT_HOUR / 24
    # Two sets before day 10's download, one after it and before day 11's, none near day 12's, one at the end.
    epoch = np.array([10 + cut - 0.5, 10 + cut - 0.1, 10 + cut + 0.2, 13 + cut - 0.3])
    assert daily_view(epoch).tolist() == [1, 2, 3]
    assert daily_view(np.array([5.0])).tolist() == [0]


def test_load_folder_joins_only_the_chosen_shards(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from build_samples import FEATURES, TARGETS
    from data import load_folder

    for shard, norad in ((1, 65), (2, 66), (3, 67)):
        columns = {"norad": [norad], "epoch": [10.0], "horizon": [1.0], "dt": [1.0], "radius": [7000.0]}
        columns.update({name: [0.5] for name in TARGETS + FEATURES})
        pq.write_table(pa.table(columns), tmp_path / f"shard_{shard:02d}.parquet")
    d = load_folder(str(tmp_path), 0.0, 20.0, shards={1, 3})
    assert sorted(d["norad"].tolist()) == [65, 67]
    assert d["x"].shape == (2, len(FEATURES))
    assert d["y"].shape == (2, len(TARGETS))

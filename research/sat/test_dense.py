"""Samples at any time gap, and the scoring that compares them with the horizons the shipped file holds."""
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_samples import FEATURES, HORIZONS, TARGETS, object_samples
from data import load_folder, metric_weights
from dense_eval import in_steps
from dense_samples import EDGES, dense_targets, tolerance_of
from train_net import train_rows

WEATHER = {day: (150.0, 150.0, 10.0) for day in range(-7306 + 25990, -7306 + 27000)}
COLUMN_AT = {"horizon": 2, "dt": 3}


def elements(n=400, step=0.05, start=26000.0):
    """One LEO object with a set every `step` days, so a truth set is close to any time asked for."""
    return {
        "epoch": start + step * np.arange(n), "inclo": np.full(n, 51.6), "nodeo": np.linspace(100.0, 80.0, n),
        "ecco": np.full(n, 0.0005), "argpo": np.full(n, 90.0), "mo": np.linspace(0.0, 300.0, n), "no_rev": np.full(n, 15.5),
        "ndot": np.full(n, 1e-4), "nddot": np.zeros(n), "bstar": np.full(n, 1e-4),
    }


def test_the_default_targets_keep_the_fixed_horizons():
    rows = object_samples(25544, elements(n=800), WEATHER)
    labels = {r[COLUMN_AT["horizon"]] for r in rows}
    assert labels == set(HORIZONS)
    for r in rows:
        assert r[COLUMN_AT["horizon"]] <= r[COLUMN_AT["dt"]] <= r[COLUMN_AT["horizon"]] + 1


def test_dense_targets_fall_in_their_buckets_and_repeat_for_one_t0():
    first = dense_targets(25544, 26000.5)
    assert [t[0] for t in first] == list(EDGES[:-1])
    for label, target, tolerance in first:
        assert label <= target < EDGES[EDGES.index(label) + 1]
        assert tolerance == tolerance_of(target)
    assert first == dense_targets(25544, 26000.5)
    assert first != dense_targets(25544, 26005.5)
    assert first != dense_targets(25545, 26000.5)


def test_tolerance_stays_small_beside_the_gap():
    assert tolerance_of(0.125) == 0.125
    assert tolerance_of(2.0) == 0.5
    assert tolerance_of(29.0) == 1.0


def test_dense_rows_use_a_set_close_to_the_time_asked_for():
    rows = object_samples(25544, elements(n=800), WEATHER, dense_targets)
    assert {r[COLUMN_AT["horizon"]] for r in rows} == set(EDGES[:-1])
    for r in rows:
        low, high = r[COLUMN_AT["horizon"]], EDGES[EDGES.index(r[COLUMN_AT["horizon"]]) + 1]
        assert low <= r[COLUMN_AT["dt"]] <= high + tolerance_of(high)


def test_a_late_set_is_refused_rather_than_stretched():
    row = elements(n=200, step=1.0)  # a set a day: nothing arrives within an eighth of a day of most short gaps
    rows = object_samples(25544, row, WEATHER, lambda norad, t: [(0.125, 0.2, 0.125)])
    assert rows == []


def test_metric_weights_keep_sub_day_buckets_apart():
    n = 4
    d = {"norad": np.full(n, 7), "horizon": np.array([0.125, 0.125, 0.25, 0.5], np.float32),
         "y": np.array([[0, 1, 0], [0, 3, 0], [0, 10, 0], [0, 100, 0]], np.float32)}
    w = metric_weights(d)
    # three buckets of one object: each weighs inversely to its own total raw error
    assert np.allclose(w[0] * 1, w[1] * 1)
    assert np.allclose(w[0] / w[2], 10 / 4)
    assert np.allclose(w[2] / w[3], 10)


def table_file(path, norad, epochs):
    n = len(epochs)
    cols = {"norad": pa.array(np.full(n, norad, np.int32)), "epoch": pa.array(np.asarray(epochs, np.float64)),
            "horizon": pa.array(np.full(n, 1.0, np.float32)), "dt": pa.array(np.full(n, 1.0, np.float32)), "radius": pa.array(np.full(n, 7000.0, np.float32))}
    for c in TARGETS + FEATURES:
        cols[c] = pa.array(np.zeros(n, np.float32))
    pq.write_table(pa.table(cols), path)


def test_load_folder_thins_by_rate(tmp_path):
    table_file(tmp_path / "shard_00.parquet", 64, np.linspace(10, 20, 4000, endpoint=False))
    all_rows = load_folder(str(tmp_path), 10, 30)
    some = load_folder(str(tmp_path), 10, 30, rate=0.25, seed=1)
    assert len(all_rows["dt"]) == 4000
    assert 800 < len(some["dt"]) < 1200


def test_train_rows_read_the_dense_folder_when_given(tmp_path):
    table_file(tmp_path / "shard_00.parquet", 64, np.linspace(10, 20, 100, endpoint=False))
    table_file(tmp_path / "shard_16.parquet", 80, np.linspace(10, 20, 100, endpoint=False))
    args = argparse.Namespace(train_samples=str(tmp_path), rate=1.0, seed=0, fit="unused", history_dropout=0.0)
    rows = train_rows(args, 10, 30, {0})
    assert set(rows["norad"]) == {64}


def test_in_steps_blends_in_squared_time_and_is_silent_outside():
    d = np.array([0.5, 1.0, 2.0, 3.0, 20.0])
    at = {1.0: np.full((5, 3), 1.0), 3.0: np.full((5, 3), 9.0)}
    got = in_steps(d, at)
    assert got[0].tolist() == [0, 0, 0]  # before the first step, the file holds nothing
    assert got[1, 0] == 1.0
    assert np.isclose(got[2, 0], 4.0)  # 1 + (4 - 1) / (9 - 1) * 8
    assert got[3, 0] == 9.0
    assert got[4].tolist() == [0, 0, 0]  # past the last step, nothing either

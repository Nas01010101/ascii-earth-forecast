"""Tests for training on rows that hide part of their history, as a job with few snapshots sees them."""
import argparse

import numpy as np

from build_samples import FEATURES, LOOKBACKS
from data import hide_history, own_drift, with_days_held
from history_check import rows_of
from train_net import train_rows

AGE = {back: FEATURES.index(f"age_{int(back)}") for back in LOOKBACKS}
UPDATES = FEATURES.index("updates_30d")


def rows(n=1000):
    x = np.ones((n, len(FEATURES)), np.float32)
    for back in LOOKBACKS:
        x[:, AGE[back]] = back
    return {"x": x, "dt": np.full(n, 3.0, np.float32), "norad": np.arange(n)}


def test_days_held_takes_one_number_for_all_rows():
    seen = with_days_held(rows(4), 3)["x"]
    assert np.isfinite(seen[:, AGE[1.0]]).all() and np.isfinite(seen[:, AGE[3.0]]).all()
    assert np.isnan(seen[:, AGE[7.0]]).all() and np.isnan(seen[:, AGE[15.0]]).all()
    assert (seen[:, UPDATES] == 1).all()


def test_days_held_takes_one_number_per_row():
    seen = with_days_held(rows(3), np.array([0.0, 3.0, np.inf]))["x"]
    assert np.isnan(seen[0, AGE[1.0]]) and seen[0, UPDATES] == 0
    assert np.isfinite(seen[1, AGE[3.0]]) and np.isnan(seen[1, AGE[7.0]]) and seen[1, UPDATES] == 1
    assert np.isfinite(seen[2, AGE[15.0]]) and seen[2, UPDATES] == 1


def test_days_held_leaves_the_rows_it_was_given():
    d = rows(3)
    with_days_held(d, 0)
    assert np.isfinite(d["x"]).all()


def test_no_history_leaves_no_drift_estimate():
    assert (own_drift(with_days_held(rows(3), 0)) == 0).all()


def test_hide_history_touches_only_the_share_asked_for():
    d = rows(10000)
    seen = hide_history(d, 0.3, seed=1)["x"]
    changed = np.isnan(seen[:, AGE[15.0]])
    assert 0.27 < changed.mean() < 0.33
    assert np.array_equal(seen[~changed], d["x"][~changed])


def test_hide_history_covers_new_objects_and_repeats_for_a_seed():
    d = rows(10000)
    first = hide_history(d, 0.3, seed=1)["x"]
    assert np.isnan(first[:, AGE[1.0]]).any()  # days held 0
    assert (np.isnan(first[:, AGE[15.0]]) & np.isfinite(first[:, AGE[3.0]])).any()  # days held 4 or 8
    assert np.array_equal(np.isnan(first), np.isnan(hide_history(d, 0.3, seed=1)["x"]))
    assert not np.array_equal(np.isnan(first), np.isnan(hide_history(d, 0.3, seed=2)["x"]))


def test_a_share_of_zero_changes_nothing():
    d = rows(10)
    assert hide_history(d, 0.0, seed=1)["x"] is d["x"]


def test_train_rows_hide_history_when_asked(tmp_path, monkeypatch):
    import train_net
    monkeypatch.setattr(train_net, "load", lambda *a, **k: rows(2000))
    args = argparse.Namespace(train_samples="", fit="unused", rate=1.0, seed=0, history_dropout=0.5)
    assert np.isnan(train_rows(args, 0, 1, {0})["x"][:, AGE[15.0]]).any()
    args.history_dropout = 0.0
    assert not np.isnan(train_rows(args, 0, 1, {0})["x"]).any()


def test_rows_of_a_parquet_are_the_2024_validation(monkeypatch):
    import history_check
    seen = {}
    monkeypatch.setattr(history_check, "load", lambda path, start, end, shards: seen.update(start=start, end=end, shards=shards) or "rows")
    assert rows_of("fit.parquet") == "rows"
    assert seen["start"] == history_check.VAL_START and seen["end"] == history_check.VAL_END
    assert seen["shards"] <= set(range(32)) and 16 in seen["shards"]

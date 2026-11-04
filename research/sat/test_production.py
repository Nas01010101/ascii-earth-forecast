"""The daily job's numpy port against the training code, on one synthetic object's history."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast import drift, orbits  # noqa: E402

import build_samples  # noqa: E402
import replay_check  # noqa: E402
from data import inputs as research_inputs  # noqa: E402
from data import own_drift as research_drift  # noqa: E402
from export_weights import export  # noqa: E402
from sat_common import SGP4_EPOCH_JD  # noqa: E402
from train_net import Net, predict, tensors  # noqa: E402

RECIPE = SimpleNamespace(fixed_dt=True, no_weather=True, no_history=False, solar_ratio=False, max_drift=drift.MAX_DRIFT_KM)
START = 27300.0  # about October 2024 in SGP4 epoch days
WEATHER = {day: (150.0, 140.0, 8.0) for day in range(19000, 21000)}


def history(n=45, seed=3):
    """Element columns of a slowly decaying low orbit, one set a day."""
    rng = np.random.default_rng(seed)
    epoch = START + np.cumsum(rng.uniform(0.7, 1.3, n))
    no_rev = 15.1 + 2e-4 * (epoch - START) + rng.normal(0, 2e-5, n)
    return {
        "epoch": epoch, "inclo": np.full(n, 51.6), "nodeo": (100 - 5 * (epoch - START)) % 360, "ecco": np.full(n, 0.0005),
        "argpo": (40 + 3 * (epoch - START)) % 360, "mo": (360 * no_rev * (epoch - START)) % 360, "no_rev": no_rev,
        "ndot": np.full(n, 1e-4), "nddot": np.zeros(n), "bstar": 3e-4 + rng.normal(0, 1e-5, n),
    }


def csv_rows(columns):
    base = datetime(1970, 1, 1, tzinfo=timezone.utc)
    rows = []
    for k, epoch in enumerate(columns["epoch"]):
        moment = base + timedelta(days=epoch + SGP4_EPOCH_JD - drift.JD_UNIX_EPOCH)
        rows.append({
            "NORAD_CAT_ID": "12345", "OBJECT_ID": "2000-001A", "OBJECT_NAME": "TEST", "EPHEMERIS_TYPE": "0", "CLASSIFICATION_TYPE": "U",
            "ELEMENT_SET_NO": "999", "REV_AT_EPOCH": "100", "EPOCH": moment.replace(tzinfo=None).isoformat(),
            "MEAN_MOTION": repr(float(columns["no_rev"][k])), "ECCENTRICITY": repr(float(columns["ecco"][k])),
            "INCLINATION": repr(float(columns["inclo"][k])), "RA_OF_ASC_NODE": repr(float(columns["nodeo"][k])),
            "ARG_OF_PERICENTER": repr(float(columns["argpo"][k])), "MEAN_ANOMALY": repr(float(columns["mo"][k])),
            "MEAN_MOTION_DOT": repr(float(columns["ndot"][k])), "MEAN_MOTION_DDOT": repr(float(columns["nddot"][k])),
            "BSTAR": repr(float(columns["bstar"][k])),
        })
    return rows


def research_rows(columns):
    return build_samples.object_samples(12345, columns, WEATHER)


def as_dict(samples):
    """The arrays research's load() would return for these samples."""
    names = ["norad", "epoch", "horizon", "dt", "radius"] + build_samples.TARGETS + build_samples.FEATURES
    table = {n: np.array([s[k] for s in samples]) for k, n in enumerate(names)}
    return {
        "norad": table["norad"].astype(np.int32), "epoch": table["epoch"], "horizon": table["horizon"].astype(np.float32),
        "dt": table["dt"].astype(np.float32), "radius": table["radius"].astype(np.float32),
        "x": np.stack([table[f] for f in build_samples.FEATURES], axis=1).astype(np.float32),
        "y": np.stack([table[t] for t in build_samples.TARGETS], axis=1).astype(np.float32),
    }


def production_x(columns, samples):
    rows = csv_rows(columns)
    out = []
    for s in samples:
        i = int(np.searchsorted(columns["epoch"], s[1]))
        # A long history means the 30-day window is full, so no scaling applies.
        out.append(drift.features(columns["epoch"], rows, i, held_days=30))
    return np.array(out, dtype=np.float32)


def test_the_feature_names_match():
    assert drift.FEATURES == build_samples.FEATURES


def test_the_sets_read_from_csv_rows_are_the_same_orbits():
    columns = history()
    rows = csv_rows(columns)
    assert drift.epoch_day(rows[5]) == pytest.approx(columns["epoch"][5], abs=2e-5)


def test_features_match_the_samples_the_network_trained_on():
    columns = history()
    samples = research_rows(columns)
    assert samples
    research = as_dict(samples)["x"].copy()
    research[:, [drift.INDEX[w] for w in drift.WEATHER]] = 0.0
    assert np.allclose(production_x(columns, samples), research, rtol=1e-3, atol=2e-3, equal_nan=True)


def test_inputs_match_without_the_weather_columns():
    columns = history()
    samples = research_rows(columns)
    d = as_dict(samples)
    dt = d["horizon"]
    weather = [drift.INDEX[w] for w in drift.WEATHER]
    d["x"][:, weather] = 0.0
    theirs = research_inputs({**d, "dt": dt})
    ours = drift.inputs(d["x"], dt)
    assert np.allclose(ours, theirs, atol=1e-6)
    assert np.allclose(drift.own_drift(d["x"], dt), research_drift({**d, "dt": dt}))


def test_forward_matches_torch(tmp_path):
    columns = history()
    d = as_dict(research_rows(columns))
    torch.manual_seed(0)
    net = Net(45, depth=6, width=32, dropout=0.0)
    with torch.no_grad():
        net.last[1].weight.normal_(0, 0.1)
        net.last[1].bias.normal_(0, 0.1)
    path = tmp_path / "net.pt"
    torch.save({"state": net.state_dict(), "args": vars(RECIPE)}, path)
    export(str(tmp_path / "w.npz"), [str(path), str(path)])
    nets = drift.load_nets(tmp_path / "w.npz")
    assert len(nets) == 2
    theirs = predict(net, tensors(d, RECIPE), "cpu")
    d["x"][:, [drift.INDEX[w] for w in drift.WEATHER]] = 0.0
    ours = drift.correct(nets, d["x"], d["horizon"])
    assert np.allclose(ours, theirs, rtol=1e-3, atol=1e-3)


def test_a_zero_network_leaves_the_miss_unchanged(tmp_path):
    columns = history()
    rows = csv_rows(columns)
    known = {12345: (columns["epoch"], rows)}
    torch.manual_seed(0)
    path = tmp_path / "net.pt"
    torch.save({"state": Net(45, depth=2, width=8, dropout=0.0).state_dict(), "args": {}}, path)
    export(str(tmp_path / "w.npz"), [str(path)])
    row = drift.score_start(known, {12345: rows[20]}, 3.0, 30, drift.load_nets(tmp_path / "w.npz"))
    assert row["n"] == 1
    assert row["model_km"] == row["sgp4_km"] > 0


def test_a_forecast_does_not_read_sets_after_its_start():
    columns = history()
    rows = csv_rows(columns)
    full = drift.features(columns["epoch"], rows, 20, 30)
    cut = drift.features(columns["epoch"][:21], rows[:21], 20, 30)
    assert np.allclose(full, cut, equal_nan=True)


def test_a_replayed_set_on_a_whole_second_can_be_read():
    columns = {**history(n=1), "epoch": np.array([27300.0])}
    assert orbits.satrec(replay_check.csv_row(columns, 0, 12345)).satnum == 12345


def test_a_short_history_scales_the_update_count_and_flags_lookbacks():
    columns = history()
    rows = csv_rows(columns)
    x = np.array([drift.features(columns["epoch"][8:], rows[8:], 8, 8)], dtype=np.float32)
    # Eight sets in about eight days reads as about thirty in thirty.
    assert x[0, drift.INDEX["updates_30d"]] == pytest.approx(30, abs=1)
    assert np.isfinite(x[0, drift.INDEX["age_7"]]) and np.isnan(x[0, drift.INDEX["age_15"]])
    assert drift.inputs(x, np.array([3.0], np.float32))[0, -1] == 1.0

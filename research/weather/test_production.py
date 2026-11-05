"""The daily job's numpy port against the training code, on synthetic fields and a saved network."""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from forecast import cloudnet  # noqa: E402

import candidate  # noqa: E402
import fetch_openmeteo  # noqa: E402
from export_weights import export  # noqa: E402
from features import Spec, one_day  # noqa: E402
from nets import UNet  # noqa: E402

WHEN = date(2026, 3, 2)  # the 30 days before it cross a month end
SPEC = Spec(cloudnet.VARIABLES, cloudnet.STEPS_PER_DAY, cloudnet.LAGS)


def scales(rng):
    stats = {}
    for name in cloudnet.VARIABLES:
        stats[f"clim_{name}"] = rng.normal(10, 3, (366, 24, 48)).astype(np.float32)
        stats[f"std_{name}"] = float(rng.uniform(1, 5))
    stats["land_sea_mask"], stats["geopotential_at_surface"] = rng.random((24, 48)), rng.normal(0, 1, (24, 48))
    return stats


@pytest.fixture
def saved(tmp_path):
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    net = UNet(SPEC.channels, 8, 4)
    # The trained output layer starts at zero, so a fresh net needs non-zero weights to show a difference.
    torch.nn.init.normal_(net.out.weight, std=.1)
    torch.nn.init.normal_(net.place, std=.5)
    path = tmp_path / "net.pt"
    torch.save({"kind": "net", "state": net.state_dict(), "stats": scales(rng), "width": 8, "place": 4,
                "spec": {"variables": SPEC.variables, "steps": SPEC.steps, "lags": SPEC.lags}}, path)
    return path


def held(seed=2):
    rng = np.random.default_rng(seed)
    return rng.normal(12, 4, (7, cloudnet.DAYS_NEEDED * cloudnet.STEPS_PER_DAY, 24, 48)).astype(np.float32)


def as_fields(steps):
    return {name: steps[k] for k, name in enumerate(cloudnet.VARIABLES)}


def test_inputs_match_the_training_code(saved, tmp_path):
    out = tmp_path / "w.npz"
    export(out, saved, "float32")
    weights = dict(np.load(out))
    stats = torch.load(saved, weights_only=False)["stats"]
    x, base = one_day(as_fields(held()), WHEN, SPEC, stats)
    ours, our_base = cloudnet.inputs(held(), WHEN, weights)
    assert ours.shape == x[0].shape == (SPEC.channels, 24, 48)
    for channel in range(SPEC.channels):
        np.testing.assert_allclose(ours[channel], x[0, channel].numpy(), atol=2e-5, err_msg=str(channel))
    np.testing.assert_allclose(our_base, base[0].numpy(), atol=1e-3)


def test_forecast_matches_the_torch_network(saved, tmp_path):
    out = tmp_path / "w.npz"
    export(out, saved, "float32")
    found = candidate.load(str(saved)).cloud(as_fields(held()), WHEN)
    np.testing.assert_allclose(cloudnet.predict(held(), WHEN, dict(np.load(out))), found, atol=2e-2)


def test_day_steps_match_the_research_download():
    rng = np.random.default_rng(3)
    length = 24 * 3
    hourly = {name: rng.uniform(0, 50, (length, 24, 48)).astype(np.float32) for name in fetch_openmeteo.HOURLY}
    hourly["direction"] = rng.uniform(0, 360, (length, 24, 48)).astype(np.float32)
    research = fetch_openmeteo.six_hourly(hourly)
    hours = np.datetime64("2026-03-01T00", "h") + np.arange(length)
    found = cloudnet.day_steps(hours, hourly, date(2026, 3, 2))
    for k, name in enumerate(cloudnet.VARIABLES):
        np.testing.assert_allclose(found[k], research[name][:4], atol=1e-3, err_msg=name)


def test_export_keeps_the_trained_layers(saved, tmp_path):
    out = tmp_path / "w.npz"
    export(out, saved, "float16")
    weights = np.load(out)
    state = torch.load(saved, weights_only=False)["state"]
    assert weights["down.0.first.weight"].dtype == np.float16
    assert set(state) <= set(weights.files)
    assert weights["clim_cloud"].dtype == np.float32 and weights["fixed"].shape == (4, 24, 48)


def test_a_day_with_missing_hours_is_refused():
    length = 24 * 3
    hours = np.datetime64("2026-03-01T00", "h") + np.arange(length)
    fields = {name: np.zeros((length, 2, 2), dtype=np.float32) for name in fetch_openmeteo.HOURLY}
    assert cloudnet.day_steps(hours, fields, date(2026, 3, 1)) is None  # no hours of the day before for its rain
    assert cloudnet.day_steps(hours, fields, date(2026, 3, 2)) is not None
    assert cloudnet.day_steps(hours[:24 + 18], fields, date(2026, 3, 2)) is None  # the 18 UTC step is not held
    assert cloudnet.view({(WHEN - timedelta(days=1)).isoformat(): np.zeros((7, 4, 2, 2))}, WHEN) is None


@pytest.mark.parametrize("target", ["temp", "pressure"])
def test_a_target_forecast_matches_the_torch_network(tmp_path, target):
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    spec = Spec(cloudnet.VARIABLES, cloudnet.STEPS_PER_DAY, cloudnet.LAGS, target)
    net = UNet(spec.channels, 8, 4)
    torch.nn.init.normal_(net.out.weight, std=.1)
    torch.nn.init.normal_(net.place, std=.5)
    stats = scales(rng)
    stats[f"clim_{target}"] = stats[f"clim_{target}"] + 1000  # a pressure-sized mean must not swamp the input
    path = tmp_path / "net.pt"
    torch.save({"kind": "net", "state": net.state_dict(), "stats": stats, "width": 8, "place": 4,
                "spec": {"variables": spec.variables, "steps": spec.steps, "lags": spec.lags, "target": target}}, path)
    out = tmp_path / "w.npz"
    export(out, path, "float32")
    weights = dict(np.load(out))
    steps = held()
    torch_x, torch_base = one_day(as_fields(steps), WHEN, spec, stats)
    x, base = cloudnet.inputs(steps, WHEN, weights, target)
    assert np.allclose(x, torch_x[0].numpy(), atol=1e-3)
    assert np.allclose(base, torch_base[0].numpy(), atol=1e-2)
    assert abs(x[-1]).max() < 20
    got = cloudnet.predict(steps, WHEN, weights, target)
    assert np.allclose(got, candidate.load(str(path)).forecast(as_fields(steps), WHEN), atol=2e-2)


def test_a_layers_file_over_the_cloud_scales_gives_the_same_forecast(tmp_path):
    torch.manual_seed(2)
    spec = Spec(cloudnet.VARIABLES, cloudnet.STEPS_PER_DAY, cloudnet.LAGS, "pressure")
    net = UNet(spec.channels, 8, 4)
    torch.nn.init.normal_(net.out.weight, std=.1)
    path, whole, layers = tmp_path / "net.pt", tmp_path / "whole.npz", tmp_path / "layers.npz"
    torch.save({"kind": "net", "state": net.state_dict(), "stats": scales(np.random.default_rng(2)), "width": 8, "place": 4,
                "spec": {"variables": spec.variables, "steps": spec.steps, "lags": spec.lags, "target": "pressure"}}, path)
    export(whole, path, "float32")
    export(layers, path, "float32", layers=True)
    assert not any(name.startswith(("clim_", "std_")) or name == "fixed" for name in cloudnet.load(layers))
    # The cloud file brings the scales, and its own layers must give way to the pressure network's.
    cloud_file = {**cloudnet.load(whole), "out.weight": np.zeros_like(cloudnet.load(whole)["out.weight"])}
    joined = cloudnet.load_others(cloud_file, {"pressure": layers})["pressure"]
    steps = held()
    assert np.array_equal(cloudnet.predict(steps, WHEN, joined, "pressure"), cloudnet.predict(steps, WHEN, cloudnet.load(whole), "pressure"))

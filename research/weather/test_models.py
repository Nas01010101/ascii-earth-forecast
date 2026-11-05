"""Inputs, the network and the linear fit, on synthetic fields."""
from argparse import Namespace
from datetime import date, timedelta

import numpy as np
import pytest
import torch

import candidate
import fit_linear
import nets
import score
import train_net
from common import FIELDS, STEPS_PER_DAY, daily
from features import Frames, Spec, TRAIL_DAYS, one_day

DAYS = 400
START = date(2001, 1, 1)


def stats():
    out = {f"clim_{n}": np.full((366, 24, 48), 50.0, np.float32) for n in FIELDS}
    out.update({f"std_{n}": np.float32(10.0) for n in FIELDS})
    out.update(land_sea_mask=np.zeros((24, 48), np.float32), geopotential_at_surface=np.zeros((24, 48), np.float32))
    return out


def moving(days=DAYS, seed=0, shift=1):
    """Each day is yesterday's anomaly pattern moved east by `shift` cells, with fresh noise mixed in."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((days, 24, 48), np.float32)
    grid[0] = rng.standard_normal((24, 48))
    for d in range(1, days):
        grid[d] = 0.95 * np.roll(grid[d - 1], shift, axis=1) + 0.3 * rng.standard_normal((24, 48))
    field = np.repeat(50 + 10 * grid, STEPS_PER_DAY, axis=0).astype(np.float32)
    return {n: field.copy() for n in FIELDS}


def test_channels_match_the_spec():
    spec = Spec(("cloud", "u"), steps=8, lags=3)
    x, base = Frames(moving(), START, spec, stats()).batch(torch.tensor([50, 60]))
    assert x.shape == (2, spec.channels, 24, 48) and base.shape == (2, 24, 48)


def test_a_batch_does_not_read_the_target_day_or_later():
    fields, spec = moving(), Spec(("cloud", "temp"), steps=8, lags=3)
    first, _ = Frames(fields, START, spec, stats()).batch(torch.tensor([100]))
    for n in fields:
        fields[n][100 * STEPS_PER_DAY:] = 999.0
    second, _ = Frames(fields, START, spec, stats()).batch(torch.tensor([100]))
    assert torch.equal(first, second)


def test_the_step_just_before_the_target_is_read():
    fields, spec = moving(), Spec(("cloud",), steps=1, lags=0)
    fields["cloud"][100 * STEPS_PER_DAY - 1] = 70.0
    x, _ = Frames(fields, START, spec, stats()).batch(torch.tensor([100]))
    assert torch.allclose(x[0, 0], torch.full((24, 48), 2.0))


def test_one_forecast_matches_the_training_batch():
    fields, spec = moving(), Spec(("cloud", "temp"), steps=4, lags=3)
    x, base = Frames(fields, START, spec, stats()).batch(torch.tensor([100]))
    view = {n: f[:100 * STEPS_PER_DAY] for n, f in fields.items()}
    y, other = one_day(view, START + timedelta(days=100), spec, stats())
    assert torch.allclose(x, y, atol=1e-4) and torch.allclose(base, other, atol=1e-3)


def test_the_base_is_the_mean_plus_the_trailing_departure():
    fields = {n: np.full((60 * STEPS_PER_DAY, 24, 48), 60.0, np.float32) for n in FIELDS}
    _, base = Frames(fields, START, Spec(), stats()).batch(torch.tensor([TRAIL_DAYS + 5]))
    assert torch.allclose(base, torch.full((1, 24, 48), 60.0))


def test_truth_is_the_daily_mean_in_percent():
    fields = moving()
    got = Frames(fields, START, Spec(), stats()).truth(torch.tensor([70]))
    assert np.allclose(got[0].numpy(), daily(fields["cloud"])[70], atol=1e-3)


def test_longitude_wraps_and_the_network_moves_with_its_input():
    net = nets.UNet(3, width=4, place=0).eval()
    nn = torch.nn.init
    nn.normal_(net.out.weight, 0, 0.1)
    x = torch.randn(1, 3, 24, 48)
    with torch.no_grad():
        # Three halvings make shifts of 8 cells exact for the pooled grid.
        assert torch.allclose(net(x.roll(8, -1)), net(x).roll(8, -1), atol=1e-4)


def test_a_new_network_predicts_the_base_forecast():
    assert nets.UNet(3, width=4)(torch.randn(2, 3, 24, 48)).abs().max() == 0


def small_args(**kw):
    return Namespace(seed=0, width=8, place=2, epochs=8, batch=16, lr=3e-3, decay=0.0, **kw)


def test_a_network_learns_motion_a_positive_control():
    spec = Spec(("cloud",), 0, 2)
    frames = Frames(moving(DAYS), START, spec, stats())
    val = Frames(moving(120, seed=5), START, spec, stats())
    days = np.arange(val.first_day(), 120)
    untrained = train_net.loss_of(nets.UNet(spec.channels, 8, 2), val, days)
    state, _, best = train_net.train(frames, val, days, spec, small_args(), "cpu")
    assert state is not None and best < 0.5 * untrained


def test_the_linear_fit_recovers_motion_from_a_wider_stencil():
    spec = Spec(("cloud",), 0, 1)
    frames = Frames(moving(), START, spec, stats())
    val = Frames(moving(120, seed=5), START, spec, stats())
    days = np.arange(val.first_day(), 120)
    group = fit_linear.groups("band")
    xx, xy = fit_linear.normal_equations(frames, np.arange(frames.first_day(), DAYS), spec, 1, group)
    weights = fit_linear.solve(xx, xy, 1.0)
    # A one-cell move east reads the neighbour to the west, the fourth of nine stencil cells down the first channel.
    assert weights[1, :9].abs().argmax() == 3
    narrow_xx, narrow_xy = fit_linear.normal_equations(frames, np.arange(frames.first_day(), DAYS), spec, 0, group)
    narrow = fit_linear.validation(val, days, fit_linear.solve(narrow_xx, narrow_xy, 1.0), group, spec, 0)
    wide = fit_linear.validation(val, days, weights, group, spec, 1)
    assert wide < 0.7 * narrow


def test_stencil_shape_and_order_keep_the_centre():
    x = torch.arange(24 * 48, dtype=torch.float32).reshape(1, 1, 24, 48)
    out = fit_linear.stencil(x, 1)
    assert out.shape == (1, 1152, 9)
    assert out[0, 0, 4] == x[0, 0, 0, 0] and out[0, 1, 4] == x[0, 0, 0, 1]


def test_saved_network_runs_through_the_judge(tmp_path):
    spec = Spec(("cloud", "temp"), 0, 3)
    net = nets.UNet(spec.channels, 4, 2)
    path = tmp_path / "n.pt"
    torch.save({"kind": "net", "state": net.state_dict(), "spec": {"variables": spec.variables, "steps": 0, "lags": 3}, "width": 4, "place": 2, "stats": stats()}, path)
    data = {n: f for n, f in moving(150).items()}
    data.update(first_date=np.array("2020-11-02"), first_target=np.array("2021-01-01"))
    found = score.losses(candidate.load(str(path)), data)
    assert sorted(found) == ["2021Q1"] and len(found["2021Q1"]) == 90


def test_saved_linear_runs_through_the_judge(tmp_path):
    spec = Spec(("cloud",), 0, 1)
    group = fit_linear.groups("cell")
    path = tmp_path / "l.npz"
    np.savez(path, kind="linear", weights=np.zeros((1152, spec.dynamic * 9), np.float32), group=group, radius=1, variables=np.array(spec.variables), steps=0, lags=1, penalty=1.0, **stats())
    data = {n: f for n, f in moving(150).items()}
    data.update(first_date=np.array("2020-11-02"), first_target=np.array("2021-01-01"))
    rows = score.losses(candidate.load(str(path)), data)["2021Q1"]
    assert len(rows) == 90 and all(r[0] > 0 for r in rows)


def test_a_climatology_file_runs_through_the_judge_as_the_base_forecast(tmp_path):
    path = tmp_path / "stats.npz"
    np.savez(path, **stats())
    data = {n: f for n, f in moving(150).items()}
    data.update(first_date=np.array("2020-11-02"), first_target=np.array("2021-01-01"))
    model = candidate.load(str(path))
    assert isinstance(model, candidate.Climatology)
    assert len(score.losses(model, data)["2021Q1"]) == 90


if __name__ == "__main__":
    pytest.main([__file__])

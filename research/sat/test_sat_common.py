"""Checks the parser and error geometry against sgp4's own TLE reader."""
import math

import numpy as np
import pytest
from sgp4.api import Satrec

from build_samples import EPOCH_TO_UNIX_DAY, error_of
from sat_common import curvilinear_error, dedupe_sorted, error_between, make_satrec, miss_km, parse_pair, rtn_error
from score import group_scores

# The ISS element set used in the sgp4 package's documentation.
L1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
L2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"


def test_parsed_fields_propagate_like_twoline():
    row = parse_pair(L1, L2)
    assert row[0] == 25544
    ours, theirs = make_satrec(*row), Satrec.twoline2rv(L1, L2)
    for minutes in (0.0, 1440.0, 21600.0):
        _, r0, v0 = ours.sgp4_tsince(minutes)
        _, r1, v1 = theirs.sgp4_tsince(minutes)
        assert np.allclose(r0, r1, atol=1e-6) and np.allclose(v0, v1, atol=1e-9)


def test_negative_exponent_fields():
    row = parse_pair(L1, L2)
    assert row[10] == pytest.approx(0.38792e-4)
    assert row[9] == 0.0


def test_alpha5_catalogue_number():
    line = "1 A0000U" + L1[8:]
    assert parse_pair(line, L2)[0] == 100000


def test_malformed_pair_raises():
    with pytest.raises(ValueError):
        parse_pair(L2, L1)


def test_rtn_axes():
    r, v = (7000.0, 0.0, 0.0), (0.0, 7.5, 0.0)
    assert rtn_error((7001.0, 0.0, 0.0), r, v) == pytest.approx((1.0, 0.0, 0.0))
    assert rtn_error((7000.0, 0.0, 1.0), r, v) == pytest.approx((0.0, 1.0, 0.0))
    assert rtn_error((7000.0, 1.0, 0.0), r, v) == pytest.approx((0.0, 0.0, 1.0))


def test_rtn_preserves_length_on_a_tilted_orbit():
    r, v = (4000.0, 3000.0, 5000.0), (-3.0, 6.0, 0.5)
    e = rtn_error((4001.0, 2998.0, 5003.0), r, v)
    assert math.sqrt(sum(c * c for c in e)) == pytest.approx(math.sqrt(1 + 4 + 9))


def test_error_is_zero_against_itself():
    sat = make_satrec(*parse_pair(L1, L2))
    assert error_between(sat, sat, 0.0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_dedupe_sorts_and_drops_repeats():
    assert dedupe_sorted(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [1, 3, 0]


def test_large_along_error_reads_as_an_arc_with_no_false_height():
    r, v = (7000.0, 0.0, 0.0), (0.0, 7.5, 0.0)
    angle = 1000.0 / 7000.0
    true = (7000.0 * math.cos(angle), 7000.0 * math.sin(angle), 0.0)
    height, along, cross, radius = curvilinear_error(r, v, true)
    assert height == pytest.approx(0.0, abs=1e-9)
    assert along == pytest.approx(1000.0)
    assert cross == pytest.approx(0.0, abs=1e-9)
    assert radius == pytest.approx(7000.0)


def test_miss_rebuilds_the_real_distance_on_a_tilted_orbit():
    r, v = (4000.0, 3000.0, 5000.0), (-3.0, 6.0, 0.5)
    true = (4100.0, 2700.0, 5200.0)
    h, a, c, radius = curvilinear_error(r, v, true)
    raw = miss_km(np.array([radius]), np.array([[h, a, c]]), np.zeros((1, 3)))[0]
    assert raw == pytest.approx(math.dist(r, true), rel=1e-9)


def test_perfect_correction_leaves_no_miss():
    truth = np.array([[3.0, -400.0, 20.0]])
    assert miss_km(np.array([7000.0]), truth, truth)[0] == pytest.approx(0.0, abs=1e-9)


def test_score_rewards_a_correction_and_punishes_a_wrong_one():
    n = 2000
    norad = np.repeat(np.arange(n // 5), 5)
    horizon = np.full(n, 15.0)
    radius = np.full(n, 7000.0)
    truth = np.tile([0.0, 100.0, 0.0], (n, 1))
    # Miss is a chord and the correction an arc, so half the arc is very slightly more than half the miss.
    assert group_scores(norad, horizon, radius, truth, truth * 0.5) == pytest.approx(50.0, abs=1e-2)
    assert group_scores(norad, horizon, radius, truth, truth * 0.0) == pytest.approx(0.0, abs=1e-9)
    assert (group_scores(norad, horizon, radius, truth, truth * -1.0) < 0).all()


def test_score_ignores_other_horizons():
    norad = np.repeat(np.arange(200), 10)
    horizon = np.tile([15.0, 1.0], 1000)
    truth = np.tile([0.0, 100.0, 0.0], (2000, 1))
    correction = truth.copy()
    correction[horizon == 1.0] = 0.0
    assert group_scores(norad, horizon, np.full(2000, 7000.0), truth, correction) == pytest.approx(100.0, abs=1e-6)


def test_epoch_offset_matches_the_unix_day():
    row = parse_pair(L1, L2)
    # Epoch 2019 day 343.69 is 2019-12-09.
    assert int(np.floor(row[1])) + EPOCH_TO_UNIX_DAY == int(np.datetime64("2019-12-09").astype(int))


def test_samples_use_the_same_error_as_the_direct_comparison():
    sat = make_satrec(*parse_pair(L1, L2))
    h, a, c, _ = error_of(sat, sat, 0.0)
    assert (h, a, c) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

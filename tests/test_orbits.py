import csv
import io
import struct
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from forecast import orbits

SAMPLE = (Path(__file__).parent / "active-sample.csv").read_text()


def aged(rows, days):
    """The same satellites with a later epoch."""
    out = {}
    for key, row in rows.items():
        shifted = dict(row)
        year_part, rest = row["EPOCH"].split("T")
        shifted["EPOCH"] = f"{(date.fromisoformat(year_part) + timedelta(days=days)).isoformat()}T{rest}"
        out[key] = shifted
    return out


def test_identical_elements_are_dropped():
    rows = orbits.parse(SAMPLE)
    assert orbits.compare(rows, rows) is None


def test_error_components_rebuild_the_total():
    rows = orbits.parse(SAMPLE)
    pair = orbits.compare(rows, aged(rows, 1))
    assert pair["id"].size > 0
    assert np.isfinite(pair["km"]).all()
    assert pair["age"] == pytest.approx(1, abs=1e-6)
    # Radial, cross and along are perpendicular, so together they must rebuild the full error.
    decomposed = np.sqrt(pair["radial"] ** 2 + pair["cross"] ** 2 + pair["along"] ** 2)
    assert decomposed == pytest.approx(pair["km"], rel=1e-6)


def test_drift_fit_needs_enough_low_orbits():
    pair = {k: np.zeros(5) for k in ("rev", "along", "age", "ndot", "bstar", "radial", "cross")}
    assert orbits.fit_drift(pair) is None


def test_drift_fit_removes_a_known_quadratic():
    rng = np.random.default_rng(1)
    n = 2000
    age = rng.uniform(1, 3, n)
    pair = {"rev": np.full(n, 15.0), "age": age, "ndot": rng.uniform(0, 1e-3, n), "bstar": rng.uniform(0, 1e-3, n),
            "radial": np.zeros(n), "cross": np.zeros(n), "along": 4 * age**2}
    weights = orbits.fit_drift(pair)
    assert np.median(orbits.corrected_km(pair, weights)) < 0.01 * np.median(np.abs(pair["along"]))


def test_regimes_split_by_mean_motion():
    pair = {"rev": np.array([15.0, 2.0, 1.0]), "km": np.array([1.0, 2.0, 3.0])}
    split = orbits.regimes(pair)
    assert [split[k]["median_km"] for k in ("low", "medium", "geostationary")] == [1.0, 2.0, 3.0]


def test_pack_matches_the_site_format():
    rows = orbits.parse(SAMPLE)
    data = orbits.pack(SAMPLE)
    assert (len(data) - 8) % 32 == 0
    assert (len(data) - 8) // 32 == len(rows)
    assert struct.unpack("<d", data[:8])[0] > 1e12


def row_text(rows):
    """A CSV body from DictReader-style rows."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


@pytest.mark.parametrize("field,value", [
    ("BSTAR", ""),
    ("MEAN_MOTION", ""),
    ("MEAN_MOTION", "nan"),
    ("NORAD_CAT_ID", "340000"),
    ("EPOCH", "garbage"),
])
def test_clean_drops_only_the_unusable_row(field, value):
    rows = list(csv.DictReader(io.StringIO(SAMPLE)))
    assert len(rows) > 25
    rows[0][field] = value
    kept = orbits.clean(row_text(rows))
    assert len(list(csv.DictReader(io.StringIO(kept)))) == len(rows) - 1
    orbits.compare(orbits.parse(kept), aged(orbits.parse(kept), 1))
    orbits.pack(kept)


def test_clean_refuses_a_list_that_is_mostly_unusable():
    rows = list(csv.DictReader(io.StringIO(SAMPLE)))
    for row in rows[: len(rows) // 2]:
        row["BSTAR"] = ""
    with pytest.raises(ValueError):
        orbits.clean(row_text(rows))

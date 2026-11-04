import csv
import io
import struct
from pathlib import Path

import pytest

from forecast import orbits

SAMPLE = (Path(__file__).parent / "active-sample.csv").read_text()


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
    orbits.pack(kept)


def test_clean_refuses_a_list_that_is_mostly_unusable():
    rows = list(csv.DictReader(io.StringIO(SAMPLE)))
    for row in rows[: len(rows) // 2]:
        row["BSTAR"] = ""
    with pytest.raises(ValueError):
        orbits.clean(row_text(rows))

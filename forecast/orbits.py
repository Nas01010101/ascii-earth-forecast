"""Reads CelesTrak elements, drops the rows SGP4 or the site cannot read, and packs them for the browser."""
import csv
import io
import math
import struct
from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec
from sgp4 import omm

MAX_DROPPED = 0.05
PACKED = ("MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
          "MEAN_MOTION_DOT")
NUMERIC = PACKED + ("BSTAR",)


def parse(text):
    """Returns {norad id: csv row}. The last row wins if a satellite appears twice."""
    return {int(row["NORAD_CAT_ID"]): row for row in csv.DictReader(io.StringIO(text))}


def satrec(row):
    sat = Satrec()
    omm.initialize(sat, row)
    return sat


def _usable(row):
    try:
        satrec(row)
        int(row["NORAD_CAT_ID"])
        datetime.fromisoformat(row["EPOCH"])
        return all(math.isfinite(float(row[k])) for k in NUMERIC)
    except (KeyError, TypeError, ValueError):
        return False


def clean(text):
    """Returns the CSV without rows SGP4 or the site cannot read, so stored snapshots stay readable."""
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    kept = [r for r in rows if _usable(r)]
    if len(kept) < (1 - MAX_DROPPED) * len(rows):
        raise ValueError(f"elements: only {len(kept)} of {len(rows)} rows are usable")
    out = io.StringIO()
    writer = csv.DictWriter(out, reader.fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(kept)
    return out.getvalue()


def pack(text):
    """Packs CSV elements into the site's sats.bin: a float64 base epoch, then eight float32s per satellite."""
    rows = list(csv.DictReader(io.StringIO(text)))
    epoch_ms = np.array([datetime.fromisoformat(r["EPOCH"]).replace(tzinfo=timezone.utc).timestamp() * 1000 for r in rows])
    base = epoch_ms.max()
    table = np.column_stack([(epoch_ms - base) / 1000, [[float(r[k]) for k in PACKED] for r in rows]])
    return struct.pack("<d", base) + table.astype("<f4").tobytes()

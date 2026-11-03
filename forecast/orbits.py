"""Compares each satellite's position predicted from older elements with its newest elements.

There is no public ground truth for the whole fleet, so the newest elements stand in for
"actual". Errors are measured at the epoch of the newest elements, where SGP4 is most accurate.
"""
import csv
import io
import math
import struct
from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec
from sgp4 import omm

ISS = 25544
LEO_REV_PER_DAY = 11.25
GEO_REV_PER_DAY = 1.2
MAX_ALONG_KM = 5000  # Past this the object almost certainly manoeuvred; drag has no say in it.
MAX_DROPPED = 0.05
NUMERIC = ("MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
           "MEAN_MOTION_DOT", "BSTAR")


def parse(text):
    """Returns {norad id: csv row}. The last row wins if a satellite appears twice."""
    return {int(row["NORAD_CAT_ID"]): row for row in csv.DictReader(io.StringIO(text))}


def _satrec(row):
    sat = Satrec()
    omm.initialize(sat, row)
    return sat


def _usable(row):
    try:
        _satrec(row)
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


def _epoch_jd(sat):
    return sat.jdsatepoch + sat.jdsatepochF


def compare(old, new):
    """Returns aligned arrays for satellites in both lists whose elements moved forward in time."""
    ids = sorted(old.keys() & new.keys())
    was, now = [_satrec(old[i]) for i in ids], [_satrec(new[i]) for i in ids]
    jd = np.array([_epoch_jd(s) for s in now])
    keep = jd > np.array([_epoch_jd(s) for s in was])
    ids = np.array(ids, dtype=np.int64)[keep]
    was, now = [s for s, k in zip(was, keep) if k], [s for s, k in zip(now, keep) if k]
    if not ids.size:
        return None
    jd = jd[keep]
    # Both element sets are evaluated at the same instant, one per satellite.
    whole, frac = np.floor(jd), jd - np.floor(jd)
    err_a, r_a, _ = _propagate(was, whole, frac)
    err_b, r_b, v_b = _propagate(now, whole, frac)
    good = (err_a == 0) & (err_b == 0)
    delta = (r_a - r_b)[good]
    r, v = r_b[good], v_b[good]
    radial = r / np.linalg.norm(r, axis=1, keepdims=True)
    cross = np.cross(radial, v)
    cross /= np.linalg.norm(cross, axis=1, keepdims=True)
    along = np.cross(cross, radial)
    ids = ids[good]
    rows = [old[int(i)] for i in ids]
    return {
        "id": ids,
        "km": np.linalg.norm(delta, axis=1),
        "radial": (delta * radial).sum(1),
        "cross": (delta * cross).sum(1),
        "along": (delta * along).sum(1),
        "age": jd[good] - np.array([_epoch_jd(s) for s, g in zip(was, good) if g]),
        "rev": np.array([float(r["MEAN_MOTION"]) for r in rows]),
        "ndot": np.array([float(r["MEAN_MOTION_DOT"]) for r in rows]),
        "bstar": np.array([float(r["BSTAR"]) for r in rows]),
    }


def _propagate(sats, whole, frac):
    """Each satellite at its own instant; SatrecArray would evaluate every time for every satellite."""
    err = np.empty(len(sats), dtype=np.int8)
    r = np.empty((len(sats), 3))
    v = np.empty((len(sats), 3))
    for k, (sat, a, b) in enumerate(zip(sats, whole, frac)):
        err[k], r[k], v[k] = sat.sgp4(a, b)
    return err, r, v


def _drift_inputs(pair):
    """Along-track error grows with the square of age and with drag, so these are the terms the model sees."""
    age = pair["age"]
    return np.column_stack([age, age**2, pair["ndot"] * age**2, pair["bstar"] * age**2, np.ones(age.size)])


def fit_drift(pair, ridge=1e-3):
    """Fits along-track error on low orbits, skipping likely manoeuvres."""
    use = (pair["rev"] >= LEO_REV_PER_DAY) & (np.abs(pair["along"]) < MAX_ALONG_KM)
    if use.sum() < 100:
        return None
    x, y = _drift_inputs(pair)[use], pair["along"][use]
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)


def corrected_km(pair, weights):
    """Position error on low orbits after removing the predicted along-track error."""
    leo = pair["rev"] >= LEO_REV_PER_DAY
    along = pair["along"][leo] - _drift_inputs(pair)[leo] @ weights
    return np.sqrt(pair["radial"][leo] ** 2 + pair["cross"][leo] ** 2 + along**2)


def summary(km):
    return {"n": int(km.size), "median_km": round(float(np.median(km)), 2)}


def regimes(pair):
    """Splits a comparison into low, geostationary and everything in between."""
    leo = pair["rev"] >= LEO_REV_PER_DAY
    geo = pair["rev"] <= GEO_REV_PER_DAY
    out = {}
    for name, mask in (("low", leo), ("medium", ~leo & ~geo), ("geostationary", geo)):
        if mask.any():
            out[name] = summary(pair["km"][mask])
    return out


def pack(text):
    """Packs CSV elements into the site's sats.bin: a float64 base epoch, then eight float32s per satellite."""
    rows = list(csv.DictReader(io.StringIO(text)))
    def epoch_ms(row):
        return datetime.fromisoformat(row["EPOCH"]).replace(tzinfo=timezone.utc).timestamp() * 1000
    base = max(epoch_ms(r) for r in rows)
    out = bytearray(struct.pack("<d", base))
    for r in rows:
        out += struct.pack("<8f", (epoch_ms(r) - base) / 1000, float(r["MEAN_MOTION"]), float(r["ECCENTRICITY"]),
                           float(r["INCLINATION"]), float(r["RA_OF_ASC_NODE"]), float(r["ARG_OF_PERICENTER"]),
                           float(r["MEAN_ANOMALY"]), float(r["MEAN_MOTION_DOT"]))
    return bytes(out)

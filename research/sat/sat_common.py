"""TLE parsing and SGP4 error geometry shared by the satellite trial scripts."""
from __future__ import annotations

import math

import numpy as np
from sgp4.api import WGS72, Satrec, jday

# sgp4init counts epoch days from this Julian date (1949-12-31 00:00 UT).
SGP4_EPOCH_JD = 2433281.5
XPDOTP = 1440.0 / (2.0 * math.pi)  # rev/day to rad/min
DEG = math.pi / 180.0
COLUMNS = ("norad", "epoch", "inclo", "nodeo", "ecco", "argpo", "mo", "no_rev", "ndot", "nddot", "bstar")


def _satnum(field: str) -> int:
    # Alpha-5 catalogue numbers use a letter for the leading digits; analyst objects are dropped later.
    field = field.strip()
    if field[0].isalpha():
        letter = field[0].upper()
        offset = ord(letter) - 55 - (letter > "I") - (letter > "O")
        return offset * 10000 + int(field[1:])
    return int(field)


def _exp(field: str) -> float:
    # TLE assumed-decimal form: " 12345-3" is 0.12345e-3.
    field = field.strip().replace(" ", "")
    if not field or field in ("00000-0", "00000+0"):
        return 0.0
    sign = -1.0 if field[0] == "-" else 1.0
    body = field.lstrip("+-")
    return sign * float(f"0.{body[:-2]}e{body[-2:]}")


def parse_pair(line1: str, line2: str) -> tuple:
    """Returns the COLUMNS values of one element set. Raises ValueError on a malformed pair."""
    if line1[0] != "1" or line2[0] != "2" or len(line1) < 62 or len(line2) < 63:
        raise ValueError("not a TLE pair")
    yy = int(line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    start, _ = jday(year, 1, 1, 0, 0, 0.0)
    epoch = start + float(line1[20:32]) - 1.0 - SGP4_EPOCH_JD
    return (
        _satnum(line1[2:7]),
        epoch,
        float(line2[8:16]),
        float(line2[17:25]),
        float("0." + line2[26:33].strip()),
        float(line2[34:42]),
        float(line2[43:51]),
        float(line2[52:63]),
        float(line1[33:43]),
        _exp(line1[44:52]),
        _exp(line1[53:61]),
    )


def make_satrec(norad, epoch, inclo, nodeo, ecco, argpo, mo, no_rev, ndot, nddot, bstar) -> Satrec:
    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", int(norad) if int(norad) < 340000 else 0, epoch, bstar,
        ndot / (XPDOTP * 1440.0), nddot / (XPDOTP * 1440.0 * 1440.0),
        ecco, argpo * DEG, inclo * DEG, mo * DEG, no_rev / XPDOTP, nodeo * DEG,
    )
    return sat


def rtn_error(pred_r, true_r, true_v) -> tuple[float, float, float]:
    """Predicted minus true position in the true orbit's radial, cross-track, along-track frame, km."""
    dx = pred_r[0] - true_r[0]; dy = pred_r[1] - true_r[1]; dz = pred_r[2] - true_r[2]
    rn = math.sqrt(true_r[0] ** 2 + true_r[1] ** 2 + true_r[2] ** 2)
    rx, ry, rz = true_r[0] / rn, true_r[1] / rn, true_r[2] / rn
    wx = true_r[1] * true_v[2] - true_r[2] * true_v[1]
    wy = true_r[2] * true_v[0] - true_r[0] * true_v[2]
    wz = true_r[0] * true_v[1] - true_r[1] * true_v[0]
    wn = math.sqrt(wx * wx + wy * wy + wz * wz)
    wx /= wn; wy /= wn; wz /= wn
    sx, sy, sz = wy * rz - wz * ry, wz * rx - wx * rz, wx * ry - wy * rx
    return (dx * rx + dy * ry + dz * rz, dx * wx + dy * wy + dz * wz, dx * sx + dy * sy + dz * sz)


def error_between(older: Satrec, newer: Satrec, dt_days: float):
    """Propagates older by dt_days to newer's epoch and returns the RTN error, or None on an SGP4 failure."""
    e0, r0, _ = older.sgp4_tsince(dt_days * 1440.0)
    e1, r1, v1 = newer.sgp4_tsince(0.0)
    if e0 or e1:
        return None
    return rtn_error(r0, r1, v1)


def curvilinear_error(pred_r, pred_v, true_r) -> tuple[float, float, float, float]:
    """True minus predicted as height, along-track arc and cross-track arc, km, plus the predicted radius.

    A flat frame turns a large along-track error into a false radial one (1000 km along at 7000 km
    radius reads as 71 km radial), so the arcs are measured on the predicted sphere. Only the
    predicted state defines the frame, so a correction can be applied without knowing the truth.
    """
    rn = math.sqrt(pred_r[0] ** 2 + pred_r[1] ** 2 + pred_r[2] ** 2)
    rx, ry, rz = pred_r[0] / rn, pred_r[1] / rn, pred_r[2] / rn
    wx = pred_r[1] * pred_v[2] - pred_r[2] * pred_v[1]
    wy = pred_r[2] * pred_v[0] - pred_r[0] * pred_v[2]
    wz = pred_r[0] * pred_v[1] - pred_r[1] * pred_v[0]
    wn = math.sqrt(wx * wx + wy * wy + wz * wz)
    wx /= wn; wy /= wn; wz /= wn
    sx, sy, sz = wy * rz - wz * ry, wz * rx - wx * rz, wx * ry - wy * rx
    a = true_r[0] * rx + true_r[1] * ry + true_r[2] * rz
    b = true_r[0] * sx + true_r[1] * sy + true_r[2] * sz
    c = true_r[0] * wx + true_r[1] * wy + true_r[2] * wz
    tn = math.sqrt(a * a + b * b + c * c)
    return (tn - rn, math.atan2(b, a) * rn, math.asin(max(-1.0, min(1.0, c / tn))) * rn, rn)


def miss_km(radius: np.ndarray, truth: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """3D distance between the true point and the corrected point. Both are (height, along, cross) in km."""
    def point(d):
        rho = radius + d[:, 0]
        u, w = d[:, 1] / radius, d[:, 2] / radius
        return np.stack([rho * np.cos(w) * np.cos(u), rho * np.cos(w) * np.sin(u), rho * np.sin(w)], axis=1)
    return np.linalg.norm(point(truth) - point(correction), axis=1)


def dedupe_sorted(epoch: np.ndarray) -> np.ndarray:
    """Indices of a time-ordered, strictly increasing subset of epochs."""
    order = np.argsort(epoch, kind="stable")
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.diff(epoch[order]) > 1e-6
    return order[keep]

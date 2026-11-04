"""Loads samples into arrays and defines the transforms and the training weights shared by every model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sgp4.api import jday

from build_samples import FEATURES, HISTORY_FEATURES, LOOKBACKS, TARGETS

SGP4_EPOCH_JD = 2433281.5


def day(year: int, month: int = 1, date: int = 1) -> float:
    return jday(year, month, date, 0, 0, 0)[0] - SGP4_EPOCH_JD


def load(path: str, start: float, end: float, shards: set | None = None, rate: float = 1.0, labelled: bool = True, seed: int = 0) -> dict:
    """Rows with t0 in [start, end), truth before `end` when labelled, limited to catalogue shards and thinned by row."""
    ids = ["norad", "epoch", "horizon", "dt", "radius"]
    columns = ids + (TARGETS if labelled else []) + FEATURES
    keep = []
    rng = np.random.default_rng(seed)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500000, columns=columns):
        t = pa.Table.from_batches([batch])
        mask = pc.and_(pc.greater_equal(t["epoch"], start), pc.less(t["epoch"], end))
        if labelled:
            mask = pc.and_(mask, pc.less(pc.add(t["epoch"], pc.cast(t["dt"], pa.float64())), end))
        t = t.filter(mask)
        if shards is not None:
            t = t.filter(pa.array(np.isin(t["norad"].to_numpy() % 64, list(shards))))
        if rate < 1.0 and t.num_rows:
            t = t.filter(pa.array(rng.random(t.num_rows) < rate))
        if t.num_rows:  # converted per batch: holding the arrow tables doubled the peak on large loads
            part = {c: t[c].to_numpy() for c in ids}
            part["x"] = np.stack([t[c].to_numpy() for c in FEATURES], axis=1).astype(np.float32)
            if labelled:
                part["y"] = np.stack([t[c].to_numpy() for c in TARGETS], axis=1).astype(np.float32)
            keep.append(part)
    return {k: np.concatenate([part[k] for part in keep]) for k in keep[0]}


def load_folder(folder: str, start: float, end: float, shards: set | None = None) -> dict:
    """load() over a folder of shard_NN.parquet files, where NN is norad % 64."""
    parts = []
    for path in sorted(Path(folder).glob("shard_*.parquet")):
        if pq.ParquetFile(path).metadata.num_rows and (shards is None or int(path.stem.split("_")[1]) in shards):
            parts.append(load(str(path), start, end))
    # popped key by key: holding every part through the concatenation doubles the peak
    return {k: np.concatenate([part.pop(k) for part in parts]) for k in list(parts[0])}


def col(d: dict, name: str) -> np.ndarray:
    return d["x"][:, FEATURES.index(name)]


def own_drift(d: dict) -> np.ndarray:
    """Along-track error this object's own history implies at each horizon.

    Fits along = c * age^2 through the origin on the hindcasts at t0, then evaluates at dt. A missing
    hindcast adds nothing, and an object with none gets zero.
    """
    top = np.zeros(len(d["dt"]), dtype=np.float64)
    bottom = np.zeros(len(d["dt"]), dtype=np.float64)
    for a in LOOKBACKS:
        age = col(d, f"age_{int(a)}").astype(np.float64)
        along = col(d, f"along_{int(a)}").astype(np.float64)
        ok = np.isfinite(age) & np.isfinite(along)
        top[ok] += along[ok] * age[ok] ** 2
        bottom[ok] += age[ok] ** 4
    c = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
    return (c * d["dt"].astype(np.float64) ** 2).astype(np.float32)


def slog(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.log1p(np.abs(v))


LOG_FEATURES = [f for f in HISTORY_FEATURES if f.split("_")[0] in ("height", "along", "cross")] + ["perigee_km", "apogee_km", "updates_30d", "f107", "f107_81", "ap"]
RATE_FEATURES = ["bstar", "ndot", "nddot"] + [f for f in HISTORY_FEATURES if f.split("_")[0] in ("dno", "dbstar")]


# inputs() ends with the drift estimate and one missing flag per lookback; these columns depend on history.
HISTORY_TAIL = 1 + len(LOOKBACKS)
CHUNK = 2_000_000


def inputs(d: dict) -> np.ndarray:
    """Network inputs: horizon, compressed features, missing-value flags, and the drift estimate."""
    n = len(d["dt"])
    if n > CHUNK:  # the transforms run in float64, too large to hold at once for the whole catalogue
        return np.concatenate([inputs({k: v[i:i + CHUNK] for k, v in d.items()}) for i in range(0, n, CHUNK)])
    x = d["x"].astype(np.float64)
    missing = ~np.isfinite(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    for name in LOG_FEATURES:
        i = FEATURES.index(name)
        x[:, i] = slog(x[:, i])
    for name in RATE_FEATURES:
        i = FEATURES.index(name)
        x[:, i] = slog(x[:, i] * 1e4)
    x[:, FEATURES.index("no_rev")] /= 16.0
    x[:, FEATURES.index("inclo")] /= 180.0
    for name in [f for f in HISTORY_FEATURES if f.startswith("age_")]:
        x[:, FEATURES.index(name)] /= 30.0
    drift = slog(own_drift(d).astype(np.float64))
    extra = np.stack([np.log(d["dt"].astype(np.float64) + 0.1), d["dt"].astype(np.float64) / 30.0, drift], axis=1)
    flags = missing[:, [FEATURES.index(f"age_{int(a)}") for a in LOOKBACKS]].astype(np.float64)
    return np.concatenate([x, extra, flags], axis=1).astype(np.float32)


def metric_weights(d: dict) -> np.ndarray:
    """Weights that turn a sum of misses into the score: each object and horizon counts once, relative to its raw error."""
    raw = np.linalg.norm(d["y"], axis=1).astype(np.float64)
    key = d["norad"].astype(np.int64) * 100 + d["horizon"].astype(np.int64)
    _, inverse = np.unique(key, return_inverse=True)
    total = np.bincount(inverse, raw)
    w = 1.0 / np.maximum(total[inverse], 1e-3)
    return (w / w.mean()).astype(np.float32)

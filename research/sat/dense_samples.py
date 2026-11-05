"""Builds samples at any time gap from 3 hours to 30 days, for a network told the real gap.

usage: dense_samples.py SHARD_DIR SPACE_WEATHER_CSV OUT_PARQUET

Each t0 asks for one random gap per bucket, and the truth is the first later set, at its own epoch, so no truth is
propagated. A set that arrives long after the gap asked for is refused: the gap then says little about when sets
arrive, which is something a query for a future time cannot know.
"""
from __future__ import annotations

import sys

import numpy as np

from build_samples import main

EDGES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_LOG_EDGES = np.log(EDGES)


def tolerance_of(gap: float) -> float:
    """How far past the gap the truth set may be, in days: a quarter of the gap, within an eighth of a day and a day."""
    return min(1.0, max(0.125, 0.25 * gap))


def dense_targets(norad: int, t: float) -> list:
    """Repeatable for one object and t0, so a rebuild writes the same rows."""
    rng = np.random.default_rng([norad, int(t * 1000)])
    gaps = np.exp(rng.uniform(_LOG_EDGES[:-1], _LOG_EDGES[1:]))
    return [(EDGES[k], float(gap), tolerance_of(float(gap))) for k, gap in enumerate(gaps)]


if __name__ == "__main__":
    main(*sys.argv[1:4], targets=dense_targets)

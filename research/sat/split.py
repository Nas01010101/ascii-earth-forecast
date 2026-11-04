"""Splits samples by time: fit before 2025, holdout t0 in 2025.

usage: split.py SAMPLES_DIR OUT_DIR
Writes fit.parquet (t0 and truth both before the gap), holdout_x.parquet (features) and
holdout_y.npz (labels). A fit sample's truth must end before CUT - GAP_DAYS, so no element set
used to score the holdout was ever a training target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sgp4.api import jday

from build_samples import TARGETS

CUT = jday(2025, 1, 1, 0, 0, 0)[0] - 2433281.5
GAP_DAYS = 1.0


def main(samples: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_writer = holdout_writer = None
    labels = {k: [] for k in ["norad", "horizon", "radius", *TARGETS]}
    n_fit = n_hold = 0
    for path in sorted(Path(samples).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500000):
            table = pa.Table.from_batches([batch])
            end = pc.add(table["epoch"], pa.compute.cast(table["dt"], pa.float64()))
            fit = table.filter(pc.less(end, CUT - GAP_DAYS))
            hold = table.filter(pc.greater_equal(table["epoch"], CUT))
            if fit.num_rows:
                fit_writer = fit_writer or pq.ParquetWriter(out_dir / "fit.parquet", table.schema, compression="zstd")
                fit_writer.write_table(fit)
                n_fit += fit.num_rows
            if hold.num_rows:
                x = hold.drop_columns(TARGETS)
                holdout_writer = holdout_writer or pq.ParquetWriter(out_dir / "holdout_x.parquet", x.schema, compression="zstd")
                holdout_writer.write_table(x)
                for k in labels:
                    labels[k].append(hold[k].to_numpy())
                n_hold += hold.num_rows
    fit_writer.close()
    holdout_writer.close()
    np.savez_compressed(out_dir / "holdout_y.npz", **{k: np.concatenate(v) for k, v in labels.items()})
    print(f"fit {n_fit} rows, holdout {n_hold} rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])

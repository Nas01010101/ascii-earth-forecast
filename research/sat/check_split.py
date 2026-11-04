"""Leakage and value checks on the split, reading the sealed labels not at all.

usage: check_split.py FIT_PARQUET HOLDOUT_X OUT_JSON
Exit 1 on a blocker: label columns in the holdout features, truth in the fit set reaching the holdout
period, a holdout row before the cut, an infinite value, or a key present in both files.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq

from build_samples import FEATURES, TARGETS
from data import day

CUT = day(2025)
KEYS = ["norad", "epoch", "horizon"]


def scan(path: str, extra: list) -> dict:
    file = pq.ParquetFile(path)
    names = set(file.schema_arrow.names)
    numeric = FEATURES + [c for c in extra if c in names]
    out = {"rows": file.metadata.num_rows, "inf": {}, "nan": {}, "epoch_min": np.inf, "epoch_max": -np.inf, "truth_max": -np.inf, "keys": []}
    for batch in file.iter_batches(batch_size=500000, columns=KEYS + numeric + ["dt"]):
        epoch = batch["epoch"].to_numpy()
        out["epoch_min"] = min(out["epoch_min"], float(epoch.min()))
        out["epoch_max"] = max(out["epoch_max"], float(epoch.max()))
        out["truth_max"] = max(out["truth_max"], float((epoch + batch["dt"].to_numpy()).max()))
        for c in numeric:
            v = batch[c].to_numpy(zero_copy_only=False)
            out["inf"][c] = out["inf"].get(c, 0) + int(np.isinf(v).sum())
            out["nan"][c] = out["nan"].get(c, 0) + int(np.isnan(v).sum())
        # One integer per row: exact epoch to a microday, horizon, object.
        out["keys"].append((batch["norad"].to_numpy().astype(np.int64) * 1000 + batch["horizon"].to_numpy().astype(np.int64)) * 10 ** 11 + np.round(epoch * 1e6).astype(np.int64))
    out["keys"] = np.concatenate(out["keys"])
    return out


def main(fit: str, holdout: str, out: str) -> None:
    held_names = set(pq.ParquetFile(holdout).schema_arrow.names)
    f = scan(fit, TARGETS)
    h = scan(holdout, [])
    report = {
        "fit_rows": f["rows"], "holdout_rows": h["rows"],
        "label_columns_in_holdout": sorted(held_names & set(TARGETS)),
        "fit_truth_max_before_cut_days": CUT - f["truth_max"],
        "holdout_epoch_min_after_cut_days": h["epoch_min"] - CUT,
        "fit_duplicate_keys": int(len(f["keys"]) - len(np.unique(f["keys"]))),
        "holdout_duplicate_keys": int(len(h["keys"]) - len(np.unique(h["keys"]))),
        "keys_in_both": int(np.intersect1d(f["keys"], h["keys"]).size),
        "fit_inf": {k: v for k, v in f["inf"].items() if v},
        "holdout_inf": {k: v for k, v in h["inf"].items() if v},
        "fit_nan_frac": {k: round(v / f["rows"], 4) for k, v in f["nan"].items() if v},
        "holdout_nan_frac": {k: round(v / h["rows"], 4) for k, v in h["nan"].items() if v},
    }
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    blockers = [
        report["label_columns_in_holdout"], report["fit_truth_max_before_cut_days"] <= 0,
        report["holdout_epoch_min_after_cut_days"] < 0, report["fit_inf"], report["holdout_inf"], report["keys_in_both"],
    ]
    sys.exit(1 if any(blockers) else 0)


if __name__ == "__main__":
    main(*sys.argv[1:4])

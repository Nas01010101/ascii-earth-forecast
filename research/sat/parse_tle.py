"""Streams one yearly TLE text or zip file into parquet files split into 64 shards by catalogue number.

usage: parse_tle.py INPUT YEAR OUT_DIR
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sat_common import COLUMNS, parse_pair

SHARDS = 64
BATCH = 20000
SCHEMA = pa.schema([("norad", pa.int32())] + [(name, pa.float64()) for name in COLUMNS[1:]])


def open_text(path: Path):
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        return io.TextIOWrapper(z.open(z.namelist()[0]), encoding="ascii", errors="replace")
    return open(path, encoding="ascii", errors="replace")


def main(path: str, year: str, out: str) -> None:
    out_dir = Path(out)
    writers, buffers = {}, {s: [] for s in range(SHARDS)}
    seen = bad = 0

    def flush(s):
        if not buffers[s]:
            return
        if s not in writers:
            (out_dir / f"shard={s:02d}").mkdir(parents=True, exist_ok=True)
            writers[s] = pq.ParquetWriter(out_dir / f"shard={s:02d}" / f"tle_{year}.parquet", SCHEMA, compression="zstd")
        columns = list(zip(*buffers[s]))
        writers[s].write_table(pa.table({n: pa.array(c, type=SCHEMA.field(n).type) for n, c in zip(COLUMNS, columns)}))
        buffers[s] = []

    with open_text(Path(path)) as fh:
        first = None
        for line in fh:
            if line.startswith("1 "):
                first = line
            elif line.startswith("2 ") and first is not None:
                seen += 1
                try:
                    row = parse_pair(first, line)
                except (ValueError, IndexError):
                    bad += 1
                    first = None
                    continue
                first = None
                s = row[0] % SHARDS
                buffers[s].append(row)
                if len(buffers[s]) >= BATCH:
                    flush(s)
    for s in range(SHARDS):
        flush(s)
    for w in writers.values():
        w.close()
    print(f"{year}: {seen} pairs, {bad} malformed")


if __name__ == "__main__":
    main(*sys.argv[1:4])

#!/bin/bash
# Builds dense samples for shards 00-31, the training and validation objects, then writes a row count manifest.
# usage: prep_dense.sh DATA_DIR PYTHON   (3 builders at once: the box runs other jobs)
set -euo pipefail
data=$1 py=$2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$data/dense"

seq -w 0 31 | xargs -P 3 -I{} bash -c 'OPENBLAS_NUM_THREADS=1 PYTHONPATH='"$here"' '"$py"' '"$here"'/dense_samples.py '"$data"'/shards/shard={} '"$data"'/SW-All.csv '"$data"'/dense/shard_{}.parquet'

PYTHONPATH="$here" "$py" - "$data/dense" > "$data/dense/manifest.txt" <<'EOF'
import sys
from pathlib import Path
import pyarrow.parquet as pq
total = 0
for p in sorted(Path(sys.argv[1]).glob("*.parquet")):
    n = pq.ParquetFile(p).metadata.num_rows
    total += n
    print(p.name, n)
print("total", total)
EOF
tail -1 "$data/dense/manifest.txt"

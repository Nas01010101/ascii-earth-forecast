#!/bin/bash
# Builds dense samples for a range of shards, then writes a row count manifest for the folder.
# usage: prep_dense.sh DATA_DIR PYTHON FIRST LAST   (3 builders at once: the box runs other jobs)
set -euo pipefail
data=$1 py=$2 first=$3 last=$4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$data/dense"
manifest="$data/dense/manifest_${first}_${last}.txt"

seq -w "$first" "$last" | xargs -P 3 -I{} bash -c 'OPENBLAS_NUM_THREADS=1 PYTHONPATH='"$here"' '"$py"' '"$here"'/dense_samples.py '"$data"'/shards/shard={} '"$data"'/SW-All.csv '"$data"'/dense/shard_{}.parquet'

PYTHONPATH="$here" "$py" - "$data/dense" > "$manifest" <<'EOF'
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
tail -1 "$manifest"

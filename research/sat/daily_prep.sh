#!/bin/bash
# Builds the once-a-day 2025 samples from every parsed shard and writes a row count manifest.
# usage: daily_prep.sh DATA_DIR PYTHON MANIFEST   (3 builders at once: the box runs other jobs)
set -euo pipefail
data=$1 py=$2 manifest=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$data/daily"

ls -d "$data"/shards/shard=* | while read -r s; do
  echo "$s ${s##*=}"
done | xargs -P 3 -L 1 bash -c 'PYTHONPATH='"$here"' '"$py"' '"$here"'/daily_samples.py "$0" '"$data"'/SW-All.csv '"$data"'/daily/shard_$1.parquet'

PYTHONPATH="$here" "$py" - "$data/daily" > "$manifest" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq
total = 0
for p in sorted(Path(sys.argv[1]).glob("*.parquet")):
    n = pq.ParquetFile(p).metadata.num_rows
    total += n
    print(p.name, n)
print("total", total)
PY
tail -1 "$manifest"

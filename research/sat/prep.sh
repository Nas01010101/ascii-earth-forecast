#!/bin/bash
# Parses every yearly element file into shards, builds samples per shard, then writes a row count manifest.
# usage: prep.sh DATA_DIR PYTHON   (3 parsers and 4 builders at once: the box runs other jobs)
set -euo pipefail
data=$1 py=$2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$data/shards" "$data/samples"

ls "$data"/tle/tle20*.txt | while read -r f; do
  y=$(basename "$f" .txt); echo "$f ${y#tle}"
done | xargs -P 3 -L 1 bash -c 'PYTHONPATH='"$here"' '"$py"' '"$here"'/parse_tle.py "$0" "$1" '"$data"'/shards'

ls -d "$data"/shards/shard=* | while read -r s; do
  echo "$s ${s##*=}"
done | xargs -P 4 -L 1 bash -c 'PYTHONPATH='"$here"' '"$py"' '"$here"'/build_samples.py "$0" '"$data"'/SW-All.csv '"$data"'/samples/shard_$1.parquet'

PYTHONPATH="$here" "$py" - "$data/samples" > "$data/samples/manifest.txt" <<'EOF'
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
tail -1 "$data/samples/manifest.txt"

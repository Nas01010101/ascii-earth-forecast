#!/usr/bin/env bash
# Moves the data directory to and from the data branch, which holds one commit so daily runs add no history.
# Usage: data_branch.sh restore|publish <remote> <directory>
set -euo pipefail
action=$1 remote=$2 dir=$3

case $action in
  restore)
    # Exit 2 means the branch does not exist yet. Any other failure stops the run, because starting
    # empty after a network error would overwrite the kept history on publish.
    status=0
    git ls-remote --exit-code --heads "$remote" data >/dev/null || status=$?
    if [ "$status" -eq 2 ]; then
      mkdir -p "$dir"
    elif [ "$status" -eq 0 ]; then
      git clone --quiet --depth 1 --branch data "$remote" "$dir"
      rm -rf "$dir/.git"
    else
      exit "$status"
    fi
    ;;
  publish)
    if [ ! -f "$dir/forecast.json" ]; then
      echo "no forecast.json in $dir, nothing published" >&2
      exit 1
    fi
    cd "$dir"
    rm -rf .git
    git init --quiet --initial-branch data
    git add --all
    git -c user.name="github-actions[bot]" \
        -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
        commit --quiet --message "data for $(date -u +%F)"
    git push --quiet --force "$remote" data
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac

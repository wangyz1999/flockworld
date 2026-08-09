#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_CONFIG="${FLOCKWORLD_PATH_CONFIG:-$PROJECT_SRC/config/slurm_paths.conf}"
PATH_CONFIG="$(cd "$(dirname "$PATH_CONFIG")" && pwd)/$(basename "$PATH_CONFIG")"
source "$PATH_CONFIG"

cd "$FLOCKWORLD_CODE_DIR"

# Submits both by default. Pass "color-bg" or "floor" to submit just one,
# e.g. ./submit_all.sh color-bg
JOB="${1:-all}"

case "$JOB" in
  color-bg|all)
    sbatch \
      --export=ALL,FLOCKWORLD_PATH_CONFIG="$PATH_CONFIG" \
      jobs/color_bg_and_floor_20260805/color_bg_multistream.job
    ;;
esac

case "$JOB" in
  floor|all)
    sbatch \
      --export=ALL,FLOCKWORLD_PATH_CONFIG="$PATH_CONFIG" \
      jobs/color_bg_and_floor_20260805/single_stream_floor.job
    ;;
esac

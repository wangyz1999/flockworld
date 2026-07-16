#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_CONFIG="${FLOCKWORLD_PATH_CONFIG:-$PROJECT_SRC/config/slurm_paths.conf}"
PATH_CONFIG="$(cd "$(dirname "$PATH_CONFIG")" && pwd)/$(basename "$PATH_CONFIG")"
source "$PATH_CONFIG"

cd "$FLOCKWORLD_CODE_DIR"

sbatch \
  --export=ALL,FLOCKWORLD_PATH_CONFIG="$PATH_CONFIG" \
  jobs/flockdit_streaming_20260715/flockdit_streaming.job

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_CONFIG="${FLOCKWORLD_PATH_CONFIG:-$PROJECT_SRC/config/slurm_paths.conf}"
PATH_CONFIG="$(cd "$(dirname "$PATH_CONFIG")" && pwd)/$(basename "$PATH_CONFIG")"
source "$PATH_CONFIG"

cd "$FLOCKWORLD_CODE_DIR"

# Array 0-5 -> num_agents 2, 5, 8, 10, 20, 30. Pass a subset with
# ARRAY_SPEC, e.g. ARRAY_SPEC=0,2 ./submit_all.sh for just N=2 and N=8.
ARRAY_SPEC="${ARRAY_SPEC:-}"

sbatch \
  ${ARRAY_SPEC:+--array="$ARRAY_SPEC"} \
  --export=ALL,FLOCKWORLD_PATH_CONFIG="$PATH_CONFIG" \
  jobs/agentcount_20260804/agentcount.job

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$PROJECT_SRC/logs/flockdit_streaming_20260715"
cd "$PROJECT_SRC"

sbatch jobs/flockdit_streaming_20260715/flockdit_streaming.job

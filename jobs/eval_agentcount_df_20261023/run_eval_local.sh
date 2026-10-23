#!/bin/bash
# Run the agent-count DF eval on a plain multi-GPU machine (no SLURM).
# Group g runs on GPU (i mod NUM_GPUS), so 3 GPUs = all three groups in parallel.
# Setup and the checkpoint bundle: docs/handoff_eval_agentcount_df.md
#
#   FLOCKWORLD_PATH_CONFIG=config/local_paths.conf \
#     bash jobs/eval_agentcount_df_20261023/run_eval_local.sh
#
# The knobs of eval_agentcount_df.job pass through: MODES, AGENT_COUNTS, EPISODES,
# SECONDS_, STEPS, EVAL_ROOT. EVAL_GROUPS="color_bg" runs a subset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
export FLOCKWORLD_PATH_CONFIG="$(realpath "${FLOCKWORLD_PATH_CONFIG:-$REPO/config/local_paths.conf}")"
[[ -f "$FLOCKWORLD_PATH_CONFIG" ]] || { echo "paths file not found: $FLOCKWORLD_PATH_CONFIG (copy config/local_paths.conf.example)" >&2; exit 1; }

EVAL_GROUPS="${EVAL_GROUPS:-color_agent color_bg random_hue}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
(( NUM_GPUS > 0 )) || { echo "no GPUs visible (nvidia-smi -L); set NUM_GPUS to override" >&2; exit 1; }

i=0; pids=()
for g in $EVAL_GROUPS; do
  gpu=$(( i % NUM_GPUS ))
  echo "[$(date --iso-8601=seconds)] launching $g on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu GROUP=$g bash "$SCRIPT_DIR/eval_agentcount_df.job" &
  pids+=($!); i=$((i + 1))
  # Groups sharing a GPU would just contend; with fewer GPUs than groups, wait per wave.
  if (( i % NUM_GPUS == 0 )); then wait "${pids[@]}" || true; pids=(); fi
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "[$(date --iso-8601=seconds)] all groups finished (status $status)"
exit $status

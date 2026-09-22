#!/bin/bash
# Runs the full evaluation sweep directly inside an already-allocated interactive
# SLURM session (no sbatch) -- see docs/experiment_log/*eval_sweep*.md for context.
set -uo pipefail

PATH_CONFIG="${FLOCKWORLD_PATH_CONFIG:-$(pwd)/config/slurm_paths.conf}"
source "$PATH_CONFIG"
cd "$FLOCKWORLD_CODE_DIR"
export PYTHONUNBUFFERED=1

SWEEP_ID="${SWEEP_ID:?set SWEEP_ID}"
RESULTS_ROOT="$FLOCKWORLD_CODE_DIR/eval_results/$SWEEP_ID/results"
LOG_DIR="$FLOCKWORLD_CODE_DIR/eval_results/$SWEEP_ID/logs"
mkdir -p "$RESULTS_ROOT/streaming_arch" "$RESULTS_ROOT/agentcount" "$RESULTS_ROOT/color_bg" "$LOG_DIR"

EPISODES="${EPISODES:-6}"
SECONDS_="${SECONDS_:-10}"
STEPS="${STEPS:-50}"
export SWEEP_ID EPISODES SECONDS_ STEPS SKIP_LARGE_BASELINE ARCH_BASELINE FLOCKWORLD_PATH_CONFIG="$PATH_CONFIG"

STREAM_ROOT="$FLOCKWORLD_STORAGE_DIR/output/flockdit_streaming_20260715"
FLOOR_CONFIG=config/train_flockdit_latent_single_stream.yaml
FLOOR_DIR="$FLOCKWORLD_STORAGE_DIR/output/flock_dit_single_stream_floor"
VAE_CKPT="$FLOCKWORLD_VAE_CHECKPOINT"

run_eval() {
  local name="$1"; local group="$2"; shift 2
  local logf="$LOG_DIR/${group}_${name}.log"
  echo "[$(date --iso-8601=seconds)] === $group/$name === (log: $logf)"
  local t0 t1
  t0=$(date +%s)
  if uv run python -m modeling.cli.eval_flock_multi --mode metrics --episodes "$EPISODES" --seconds "$SECONDS_" --steps "$STEPS" \
      --tag "$name" --save-json "$RESULTS_ROOT/$group/$name.json" "$@" > "$logf" 2>&1; then
    t1=$(date +%s)
    echo "[$(date --iso-8601=seconds)] OK: $group/$name ($((t1 - t0))s)"
  else
    t1=$(date +%s)
    echo "[$(date --iso-8601=seconds)] FAILED: $group/$name ($((t1 - t0))s) -- see $logf" >&2
  fi
}

run_arch() {
  local name="$1"; local cfg="$2"; local dir="$3"
  if [[ "${ARCH_BASELINE:-single}" == "none" ]]; then
    run_eval "$name" streaming_arch --config "$cfg" \
      "vae.checkpoint_path=$VAE_CKPT" "output_dir=$STREAM_ROOT/$dir" --baseline none
  else
    run_eval "$name" streaming_arch --config "$cfg" \
      "vae.checkpoint_path=$VAE_CKPT" "output_dir=$STREAM_ROOT/$dir" \
      --baseline single --baseline-config "$FLOOR_CONFIG" --baseline-output-dir "$FLOOR_DIR"
  fi
}

run_agentcount() {
  local n="$1"; local dir="$2"
  local no_baseline="${3:-}"
  local extra=(--baseline single --baseline-config "$FLOOR_CONFIG" --baseline-output-dir "$FLOOR_DIR")
  [[ -n "$no_baseline" ]] && extra=(--baseline none)
  run_eval "agents$(printf '%02d' "$n")" agentcount \
    --config config/train_flockdit_latent_multi_stream.yaml \
    "vae.checkpoint_path=$VAE_CKPT" \
    "data.num_agents=$n" \
    "data.streaming.sim_overrides=[rendering.color_mode=agent_id,rendering.background_gradient=false,collection.partial_agents=$n]" \
    "output_dir=$FLOCKWORLD_STORAGE_DIR/output/agentcount_20260804/$dir" \
    "${extra[@]}"
}

case "${1:-all}" in
  calibrate)
    run_agentcount 2 agents02-10826537_0
    ;;
  streaming_arch)
    run_arch exp01_baseline config/train_flockdit_latent_multi_stream.yaml exp01_baseline-10290187_0
    run_arch exp02_tiled config/train_flockdit_latent_multi_stream_tiled.yaml exp02_tiled-10290187_1
    run_arch exp03_diffusion_forcing config/train_flockdit_latent_multi_stream_df.yaml exp03_diffusion_forcing-10290187_2
    run_arch exp04_two_stage config/train_flockdit_latent_multi_stream_twostage.yaml exp04_two_stage-10621962_3
    run_arch exp06_tiled_df config/train_flockdit_latent_multi_stream_tiled_df.yaml exp06_tiled_df-10290187_4
    run_arch exp07_tiled_df_two_stage config/train_flockdit_latent_multi_stream_triple.yaml exp07_tiled_df_two_stage-10621962_5
    run_arch exp08_tiled_two_stage config/train_flockdit_latent_multi_stream_twostage_tiled.yaml exp08_tiled_two_stage-10621962_6
    ;;
  agentcount)
    run_agentcount 2 agents02-10826537_0
    run_agentcount 5 agents05-10826537_1
    run_agentcount 8 agents08-10826537_2
    run_agentcount 10 agents10-10826537_3
    run_agentcount 20 agents20-10826537_4 "${SKIP_LARGE_BASELINE:-}"
    run_agentcount 30 agents30-10826537_5 "${SKIP_LARGE_BASELINE:-}"
    ;;
  color_bg)
    run_eval color_bg_multi color_bg \
      --config config/train_flockdit_latent_multi_stream.yaml \
      "vae.checkpoint_path=$FLOCKWORLD_STORAGE_DIR/pretrained/color_bg_stream/vae-6928-0.0034.ckpt" \
      "data.streaming.sim_overrides=[rendering.color_mode=agent_id,rendering.background_gradient=true]" \
      "output_dir=$FLOCKWORLD_STORAGE_DIR/output/flock_dit_multi_stream_color_bg" \
      --baseline none
    ;;
  all)
    "$0" streaming_arch
    "$0" agentcount
    "$0" color_bg
    ;;
esac
echo "[$(date --iso-8601=seconds)] $1 done"

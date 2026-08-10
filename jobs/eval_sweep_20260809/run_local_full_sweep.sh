#!/bin/bash
# Full-fidelity (6 episodes / 10s / 50 Euler steps -- eval_flock_multi.py's own defaults for
# episodes/seconds, explicit --steps here for clarity) re-run of the reduced
# eval_results/20260809_full_sweep pass, run locally (no SLURM) on whatever GPU is on this
# machine. Covers all 3 groups: streaming_arch (+ floor + the new single_stream_floor
# checkpoint as an informational extra column), agentcount, and color_bg (the new
# multi_stream_color_bg checkpoint). Writes per-episode CSV + aggregated JSON for every run
# (see compare_experiments.py / eval_flock_multi.py --save-csv), so future plotting never
# needs a rerun.
#
# Run from repo root: bash jobs/eval_sweep_20260809/run_local_full_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

ROOT=output/flockdit_streaming_20260809
RESULTS=eval_results/20260809_full_fidelity/results
LOGS="$ROOT/eval_logs"
mkdir -p "$RESULTS/streaming_arch" "$RESULTS/agentcount" "$RESULTS/color_bg" "$LOGS"

EPISODES=6
SECONDS_=10
STEPS=50

VAE_MAIN=pretrained/color_agent/checkpoint2/vae-081-0.0042.ckpt
VAE_COLORBG=pretrained/color_bg_stream/vae-6928-0.0034.ckpt
SINGLE_FLOOR_DIR="$ROOT/single_stream_floor"
SINGLE_FLOOR_CFG=config/train_flockdit_latent_single_stream.yaml
COLORBG_DIR="$ROOT/multi_stream_color_bg"

run() {
  local name="$1"; local logf="$2"; shift 2
  echo "[$(date --iso-8601=seconds)] === $name === (log: $logf)"
  local t0 t1
  t0=$(date +%s)
  if "$@" > "$logf" 2>&1; then
    t1=$(date +%s)
    echo "[$(date --iso-8601=seconds)] OK: $name ($((t1 - t0))s)"
  else
    t1=$(date +%s)
    echo "[$(date --iso-8601=seconds)] FAILED: $name ($((t1 - t0))s) -- see $logf" >&2
  fi
}

# --- Group 1: streaming architecture ablation, all 7 experiments + floor in ONE
# compare_experiments.py call (shares the held-out dataset build across columns), plus
# single_stream_floor as an extra INFORMATIONAL column (not wired in as the floor itself --
# floor stays the pre-existing exp04-stage1 baseline for comparability with older tables).
run streaming_arch_full "$LOGS/streaming_arch_full.log" \
  uv run python compare_experiments.py \
    --include exp01_baseline exp02_tiled exp03_diffusion_forcing exp04_two_stage \
               exp06_tiled_df exp07_tiled_df_two_stage exp08_tiled_two_stage \
    --episodes "$EPISODES" --seconds "$SECONDS_" --steps "$STEPS" \
    --extra-single-agent-name single_stream_floor_new \
    --extra-single-agent-config "$SINGLE_FLOOR_CFG" \
    --extra-single-agent-dir "$SINGLE_FLOOR_DIR" \
    --video-dir "$ROOT/eval_videos/streaming_arch_full" \
    --save-json "$RESULTS/streaming_arch/comparison.json" \
    --save-csv "$RESULTS/streaming_arch/comparison.csv"

# --- Group 2: agent-count scaling sweep. Baseline = the dedicated single-agent floor model
# (pre-existing convention in this job's --baseline single default), included for every N
# now that we're not on a time-boxed interactive allocation.
run_agentcount() {
  local n="$1"; local dir="$2"
  local tag; tag="agents$(printf '%02d' "$n")"
  run "agentcount_$tag" "$LOGS/agentcount_$tag.log" \
    uv run python eval_flock_multi.py --mode metrics \
      --episodes "$EPISODES" --seconds "$SECONDS_" --steps "$STEPS" \
      --config config/train_flockdit_latent_multi_stream.yaml \
      "vae.checkpoint_path=$VAE_MAIN" \
      "data.num_agents=$n" \
      "data.streaming.sim_overrides=[rendering.color_mode=agent_id,rendering.background_gradient=false,collection.partial_agents=$n]" \
      "output_dir=output/agentcount_20260804/$dir" \
      --baseline single --baseline-config "$SINGLE_FLOOR_CFG" \
      --baseline-output-dir "$SINGLE_FLOOR_DIR" \
      --tag "$tag" \
      --save-json "$RESULTS/agentcount/$tag.json" \
      --save-csv "$RESULTS/agentcount/$tag.csv"
}
run_agentcount 2  agents02-10826537_0
run_agentcount 5  agents05-10826537_1
run_agentcount 8  agents08-10826537_2
run_agentcount 10 agents10-10826537_3
run_agentcount 20 agents20-10826537_4
run_agentcount 30 agents30-10826537_5

# --- Group 3: color+gradient-background run (new checkpoint, different VAE + rendering --
# can't share compare_experiments.py's single shared-VAE table, same as before).
run color_bg_multi "$LOGS/color_bg_multi.log" \
  uv run python eval_flock_multi.py --mode metrics \
    --episodes "$EPISODES" --seconds "$SECONDS_" --steps "$STEPS" \
    --config config/train_flockdit_latent_multi_stream.yaml \
    "vae.checkpoint_path=$VAE_COLORBG" \
    "data.streaming.sim_overrides=[rendering.color_mode=agent_id,rendering.background_gradient=true]" \
    "output_dir=$COLORBG_DIR" \
    --baseline none \
    --tag color_bg_multi \
    --save-json "$RESULTS/color_bg/color_bg_multi.json" \
    --save-csv "$RESULTS/color_bg/color_bg_multi.csv"

echo "[$(date --iso-8601=seconds)] full sweep done"

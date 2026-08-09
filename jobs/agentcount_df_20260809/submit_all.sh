#!/bin/bash
# Submit all 21 agent-count + floor runs (diffusion forcing), in order.
#
#   group 1  color_agent_stream   N=2,5,8,10,20,30 + floor   7 jobs
#   group 2  color_bg_stream      N=2,5,8,10,20,30 + floor   7 jobs   (gradient bg)
#   group 3  random_hue_stream    N=2,5,8,10,20,30 + floor   7 jobs
#                                                           ---------
#                                                            21 jobs  (~42 GPU-days)
#
# Each run is submitted as its own single-index array task with a sleep in
# between, so submit times are strictly increasing and the queue sees them in
# exactly this order.
#
# Usage:
#   bash jobs/agentcount_df_20260809/submit_all.sh            # submit all 21
#   DRY_RUN=1 bash .../submit_all.sh                          # print, submit nothing
#   SLEEP_SECONDS=30 bash .../submit_all.sh                   # tighter spacing
#   SWEEP_GROUPS="color_agent color_bg" bash .../submit_all.sh      # a subset of groups
#   TASKS="0 1 2" bash .../submit_all.sh                      # a subset of tasks
#   DEPEND=1 bash .../submit_all.sh                           # chain: each run
#                                                             # starts only after
#                                                             # the previous ends
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
PATH_CONFIG="${FLOCKWORLD_PATH_CONFIG:-$PROJECT_SRC/config/slurm_paths.conf}"
PATH_CONFIG="$(cd "$(dirname "$PATH_CONFIG")" && pwd)/$(basename "$PATH_CONFIG")"
source "$PATH_CONFIG"

cd "$FLOCKWORLD_CODE_DIR"

JOB_FILE="jobs/agentcount_df_20260809/agentcount_df.job"
SWEEP_GROUPS="${SWEEP_GROUPS:-color_agent color_bg random_hue}"
TASKS="${TASKS:-0 1 2 3 4 5 6}"          # 0-5 = N 2,5,8,10,20,30 ; 6 = single-agent floor
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-}"
DEPEND="${DEPEND:-}"

declare -A TASK_LABEL=(
  [0]="agents02" [1]="agents05" [2]="agents08"
  [3]="agents10" [4]="agents20" [5]="agents30" [6]="floor"
)

# Fail before submitting anything if an encoder is missing -- 21 jobs is too many
# to discover a bad path on task 15.
declare -A GROUP_VAE=(
  [color_agent]="$FLOCKWORLD_STORAGE_DIR/pretrained/color_agent_stream/vae-1976-0.0018.ckpt"
  [color_bg]="$FLOCKWORLD_STORAGE_DIR/pretrained/color_bg_stream/vae-6928-0.0034.ckpt"
  [random_hue]="$FLOCKWORLD_STORAGE_DIR/pretrained/random_hue_stream/vae-2433-0.0015.ckpt"
)
for g in $SWEEP_GROUPS; do
  vae="${GROUP_VAE[$g]:-}"
  [[ -n "$vae" ]] || { echo "Unknown group: $g" >&2; exit 2; }
  [[ -f "$vae" ]] || { echo "Encoder missing for group $g: $vae" >&2; exit 1; }
done
[[ -f "$JOB_FILE" ]] || { echo "Job file not found: $JOB_FILE" >&2; exit 1; }

n_total=0
for g in $SWEEP_GROUPS; do for t in $TASKS; do n_total=$((n_total + 1)); done; done

LOG="$SCRIPT_DIR/submitted_$(date +%Y%m%d_%H%M%S).tsv"
echo "Submitting $n_total job(s), ${SLEEP_SECONDS}s apart${DRY_RUN:+ (DRY RUN)}"
echo "Job file: $JOB_FILE"
echo "Storage:  $FLOCKWORLD_STORAGE_DIR/output/agentcount_df_20260809"
[[ -n "$DEPEND" ]] && echo "Chaining: each run waits for the previous to finish"
echo

[[ -z "$DRY_RUN" ]] && printf 'submitted_at\tgroup\ttask\tlabel\tjobid\n' > "$LOG"

i=0
prev_jobid=""
for g in $SWEEP_GROUPS; do
  for t in $TASKS; do
    i=$((i + 1))
    label="${TASK_LABEL[$t]}"
    dep=()
    [[ -n "$DEPEND" && -n "$prev_jobid" ]] && dep=(--dependency="afterany:$prev_jobid")

    if [[ -n "$DRY_RUN" ]]; then
      echo "[$i/$n_total] would submit: $g/$label (array=$t) ${dep[*]:-}"
      prev_jobid="<dry>"
    else
      jobid="$(sbatch --parsable \
        --array="$t" \
        "${dep[@]}" \
        --export=ALL,FLOCKWORLD_PATH_CONFIG="$PATH_CONFIG",AGENTCOUNT_GROUP="$g" \
        "$JOB_FILE")"
      # --parsable on an array submission returns the array job id
      jobid="${jobid%%;*}"
      ts="$(date --iso-8601=seconds)"
      echo "[$i/$n_total] $ts  $g/$label  ->  $jobid"
      printf '%s\t%s\t%s\t%s\t%s\n' "$ts" "$g" "$t" "$label" "$jobid" >> "$LOG"
      prev_jobid="$jobid"
    fi

    # Space out submissions so submit order is unambiguous. No sleep after the last.
    if (( i < n_total )); then
      sleep "$SLEEP_SECONDS"
    fi
  done
done

echo
if [[ -n "$DRY_RUN" ]]; then
  echo "Dry run complete -- nothing submitted."
else
  echo "Submitted $n_total job(s). Record: $LOG"
  echo "Watch with: squeue -u \$USER -o '%.10i %.9P %.30j %.8T %.10M %R'"
fi

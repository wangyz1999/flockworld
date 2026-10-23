#!/usr/bin/env bash
# Run the agent-count DF eval on one Greenland node (8x H200) via greenlandtest/gpujob.
#
#   jobs/eval_agentcount_df_20261023/submit_greenland.sh            # full sweep (36 evals)
#   DRY_RUN=1 jobs/eval_agentcount_df_20261023/submit_greenland.sh  # render the manifest only
#
# Ships this commit (git archive HEAD, so commit first) as a tarball to S3, and expects the
# checkpoint bundle from docs/handoff_eval_agentcount_df.md already uploaded to $BUNDLE_S3.
# Pass-through knobs: MODES, EVAL_GROUPS, AGENT_COUNTS, EPISODES, SECONDS_, STEPS, SKIP_SMOKE.
# Results land in s3://.../output/<alias>/<run-name>/eval_results/; fetch with gpujob/fetch.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPUJOB="${GPUJOB:-/workplace/yunzhewa/greenlandtest/gpujob}"
ALIAS="${ALIAS:-$(whoami)}"
BUCKET="s3://qs-greenland-data-365184689774-sa-east-1"
BUNDLE_S3="${BUNDLE_S3:-$BUCKET/data/$ALIAS/flockworld_eval_bundle/}"
SNAP="$(date +%Y%m%d-%H%M%S)-$(git -C "$REPO" rev-parse --short HEAD)"
CODE_S3="$BUCKET/code/$ALIAS/flockworld/$SNAP.tgz"

git -C "$REPO" status --porcelain | grep -q . && echo "[submit] warning: uncommitted changes are NOT shipped" >&2
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git -C "$REPO" archive --format=tar.gz -o "$TMP/code.tgz" HEAD
[[ "${DRY_RUN:-0}" == 1 ]] || aws s3 cp "$TMP/code.tgz" "$CODE_S3" --region sa-east-1 --only-show-errors
echo "[submit] code $CODE_S3  bundle $BUNDLE_S3"

ENVS=(--env "JOB_CODE_S3=$CODE_S3" --env "JOB_BUNDLE_S3=$BUNDLE_S3")
for k in MODES EVAL_GROUPS AGENT_COUNTS EPISODES SECONDS_ STEPS SKIP_SMOKE; do
  [[ -n "${!k:-}" ]] && ENVS+=(--env "$k=${!k}")
done
cd "$(dirname "$GPUJOB")"
DRY_RUN="${DRY_RUN:-0}" ./gpujob/submit.sh --script "$REPO/jobs/eval_agentcount_df_20261023/greenland_entry.py" \
  --gpus 8 --name evalac --shm 64Gi "${ENVS[@]}"

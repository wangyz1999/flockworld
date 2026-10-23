# Handoff: agent-count diffusion-forcing eval on a fresh GPU machine

Status as of 2026-10-23. Branch `agentcount-df-jobs`.

## What this is

The Aug 9 sweep `jobs/agentcount_df_20260809/` trained 21 FlockDiT runs with
diffusion forcing on CARC: 3 encoder groups × {N = 2, 5, 8, 10, 20, 30 camera
agents, plus a single-agent floor}. **Training is finished. Nothing needs to be
retrained to run this eval.** This note covers running the metrics eval for those
runs on any CUDA machine.

| group         | VAE encoder                               | background        |
|---------------|-------------------------------------------|-------------------|
| `color_agent` | `color_agent_stream/vae-1976-0.0018.ckpt` | plain             |
| `color_bg`    | `color_bg_stream/vae-6928-0.0034.ckpt`    | gradient          |
| `random_hue`  | `random_hue_stream/vae-2433-0.0015.ckpt`  | plain (hue-invariant encoder) |

The eval (`jobs/eval_agentcount_df_20261023/eval_agentcount_df.job`) runs each
N against **its own group's floor** (`--baseline single`), at two checkpoint
choices:

- `best`: each run's lowest-val epoch checkpoint (model and floor).
- `step75k`: `step_00075000.pt` for model and floor. Every run reached this step.
  This comparison matters because training was capped by wall-clock time:
  N=30 got 18 epochs while N=2 got about 345, since attention cost is quadratic
  in N. **N=20 and N=30 were still improving when they stopped** (their best
  epoch is the last one), so `best` is biased against large N. Read `step75k`
  for "how does N scale at equal steps".

Settings: 3 episodes × 6 s, 50 Euler steps. These match
`eval_results/20260809_full_sweep` (the older teacher-forcing sweep), with one
caveat: hue-identity smoothing is on here (the current default), and that
sweep predates it.

That is 36 evals in total (3 groups × 6 N × 2 modes), each 1–5 min on an A40.

## Fresh-machine setup

Requires an NVIDIA driver new enough for CUDA 13 wheels (torch cu130,
`jax[cuda13]`; driver ≥ 580) and Python 3.13 (uv fetches it).

```bash
# 1. code
git clone -b agentcount-df-jobs https://github.com/wangyz1999/flockworld
cd flockworld
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
uv sync                                             # env from uv.lock

# 2. checkpoint bundle (private HF repo, ~5 GB; needs `hf auth login`
#    as wangyz1999 or a token with read access)
uvx --from huggingface_hub hf download wangyz1999/flockworld-eval-agentcount-df \
    --local-dir ~/flockworld_eval_bundle
cd ~/flockworld_eval_bundle && sha256sum -c <(tail -n +2 MANIFEST.tsv | awk -F'\t' '{print $3"  "$1}') && cd -

# 3. paths
cp config/local_paths.conf.example config/local_paths.conf
$EDITOR config/local_paths.conf     # FLOCKWORLD_CODE_DIR = this clone,
                                    # FLOCKWORLD_STORAGE_DIR = the bundle
```

The bundle mirrors the cluster's `$FLOCKWORLD_STORAGE_DIR`
(`/project2/ustun_1726/flockworld`), so the path file is the only change needed:

```
pretrained/<group>_stream/vae-*.ckpt
output/agentcount_df_20260809/<group>/<run>/checkpoints/{epoch_XXXX.pt (best-val), step_00075000.pt}
MANIFEST.tsv     path, bytes, sha256, global_step, val_loss
```

The bundle's checkpoints have optimizer state stripped, which is everything
eval needs. They cannot be used to resume training. The full checkpoints
(736 GB) stay on CARC under
`/project2/ustun_1726/flockworld/output/agentcount_df_20260809/`. The bundle
was built with `tools/make_eval_bundle.py`.

## Run

```bash
# smoke test (~1 min): one eval, results go to a throwaway dir
CUDA_VISIBLE_DEVICES=0 GROUP=color_agent MODES=step75k AGENT_COUNTS=2 EPISODES=1 \
  EVAL_ROOT=/tmp/eval_smoke FLOCKWORLD_PATH_CONFIG=config/local_paths.conf \
  bash jobs/eval_agentcount_df_20261023/eval_agentcount_df.job
tail -30 /tmp/eval_smoke/logs/step75k_color_agent_agents02.log

# full eval: one group per GPU, all three groups in parallel on 3+ GPUs
FLOCKWORLD_PATH_CONFIG=config/local_paths.conf \
  bash jobs/eval_agentcount_df_20261023/run_eval_local.sh
```

Optional knobs (environment variables):

- `EVAL_GROUPS="color_bg random_hue"` runs a subset of groups.
- `MODES=step75k` runs one checkpoint choice.
- `AGENT_COUNTS="20 30"` runs a subset of N.
- `EPISODES`, `SECONDS_`, `STEPS` set the eval size.
- `EVAL_ROOT` sets the output dir.
- `NUM_GPUS` overrides GPU detection.

With a faster machine, consider raising `EPISODES` (6 was the original
default). The 3-episode setting was a time-box on CARC. If you change it,
results no longer compare one-to-one with `20260809_full_sweep`.

Outputs (`eval_results/` JSON is tracked in git; `*.log` is gitignored):

```
eval_results/agentcount_df_20260809/{best,step75k}/<group>/agentsNN.json
eval_results/agentcount_df_20260809/logs/{<mode>_<group>_agentsNN,<group>_driver}.log
```

Each JSON records the exact checkpoint path, `global_step` and `val_loss` of
both model and floor (`model_checkpoint`, `baseline_checkpoint`).

## Plots / tables

`plot_eval_sweep.py` expects `<results-dir>/agentcount/*.json`, so point it at
one mode × group at a time through a symlink:

```bash
for m in best step75k; do for g in color_agent color_bg random_hue; do
  v=eval_results/agentcount_df_20260809/_view/$m/$g; mkdir -p $v
  ln -sfn "$(realpath eval_results/agentcount_df_20260809/$m/$g)" $v/agentcount
  uv run python plot_eval_sweep.py --results-dir $v \
    --plots-dir eval_results/agentcount_df_20260809/plots/$m/$g \
    --tables-dir eval_results/agentcount_df_20260809/tables/$m/$g
done; done
```

## Also running on CARC

The same eval was submitted on CARC as SLURM array `13327238` (tasks 0–2 =
color_agent, color_bg, random_hue) on 2026-10-23. Its results land in the
cluster clone at the same `eval_results/agentcount_df_20260809/` paths.
Check whether it finished before re-running everything:
`sacct -X -j 13327238 --format=JobID%14,State,Elapsed`.

## Changes on this branch you should know about

- `modeling/eval/pair_consistency.py`: **committed merge-conflict markers**
  (from merge `ddb03ee`) made every eval fail on import. Resolved here by
  keeping the HEAD docstring; this is a docstring-only change. **`origin/main`
  still has the broken file** and needs the same fix.
- `eval_flock_multi.py`: new `--checkpoint` / `--baseline-checkpoint` (path or
  file name under `<output_dir>/checkpoints`). The default is unchanged
  (best-val epoch).
- `modeling/training/flow_trainer.py`: graceful-stop fix. `timeout` sends
  SIGTERM to the whole process group, which killed the DataLoader workers, and
  PyTorch's SIGCHLD watchdog then crashed the run mid-step before the final
  save. That is why 12/21 runs are marked FAILED (they still have all periodic
  checkpoints; they lost ≤ ~2 h at the end). The trainer now disarms the
  watchdog on stop. Verified with a standalone repro, but not yet in a real
  training job.
- `jobs/eval_agentcount_df_20261023/`: the eval job (SLURM array or plain
  `bash` with `GROUP=`), plus `run_eval_local.sh`.

## Known caveats when reading results

- `best` vs `step75k`: see above. N=20/30 are undertrained under `best`.
- The floor baseline is fed frames normalized with the multi-agent config's
  latent stats, not its own. This is a documented approximation (the eval
  prints a `[warn]`); `compare_experiments.py` does a per-model-exact version.
- Rates at near-zero volume are noisy (small N rarely has agents in each
  other's view). Read the `volume` block next to each rate.
- The floor's training palette is the 10-hue one, so its eval at N ≠ 10 is
  strictly valid only for `random_hue`. See the comment in
  `jobs/agentcount_df_20260809/agentcount_df.job`.
- Open question: whether to train N=20/30 longer. The trainer has
  `train.warm_start` (weights only, fresh optimizer) but no true resume.

# Repository guide

Run the commands below from the repository root after installing the project
dependencies (`uv sync`). Use `uv run python -m ...` to select the project
environment. Configuration and output paths remain relative to the repository
root unless a command accepts an explicit path.

## Where code belongs

| Directory | Purpose |
| --- | --- |
| `flockworld/` | Reusable JAX simulator, rendering, policies, environment, and recording utilities |
| `flockworld/cli/` | Simulation recording command |
| `modeling/models/`, `modeling/data/`, `modeling/training/` | Model definitions, datasets, and training loops |
| `modeling/eval/` | Reusable evaluation, metrics, checkpoint helpers, and experiment registry |
| `modeling/cli/` | Training, latent caching, and evaluation commands |
| `modeling/legacy/` | Earlier video-prediction baseline retained for reference |
| `scripts/analysis/` | Attention, action-conditioning, and wall-event research probes |
| `scripts/diagnostics/` | Manual checks, often requiring local recorded data and checkpoints |
| `scripts/figures/` | Paper figures, sweep plots, and rollout videos |
| `scripts/benchmarks/` | Simulator performance benchmarks |
| `tests/` | Automated checks that do not require the paper's datasets or checkpoints |
| `config/` | Experiment YAMLs and cluster path settings |
| `jobs/` | Training and evaluation shell scripts, grouped by campaign |
| `docs/`, `eval_results/` | Documentation, presentation media, and published evaluation artifacts |

Put reusable functions in `flockworld/` or `modeling/`; command modules and
research scripts should import those functions. Package initializers in the
command and script directories are intentionally empty apart from docstrings.

Keep generated recordings and checkpoints under `output/`, `outputs/`, `data/`,
or `pretrained/`. These locations are ignored by Git. The files in
`eval_results/` are retained research artifacts; local reruns should use a new
output directory instead of overwriting the published measurements.

## Main commands

| Task | Command |
| --- | --- |
| Record the simulator | `uv run python -m flockworld.cli.data_recording collection.enabled=false generation.num_envs=1 video.duration=5` |
| Train the VAE | `uv run python -m modeling.cli.train_vae --config config/train_vae_color_stream.yaml` |
| Train FlockDiT | `uv run python -m modeling.cli.train_flock_dit --config config/train_flockdit_latent_multi_stream.yaml` |
| Cache latents | `uv run python -m modeling.cli.precompute_latents --help` |
| Evaluate single-agent rollouts | `uv run python -m modeling.cli.eval_flock_dit --help` |
| Evaluate multi-agent rollouts | `uv run python -m modeling.cli.eval_flock_multi --help` |
| Compare paper experiments | `uv run python -m modeling.cli.compare_experiments --help` |
| Evaluate heading-based identity | `uv run python -m modeling.cli.eval_heading_identity --help` |
| Plot an evaluation sweep | `uv run python -m scripts.figures.plot_eval_sweep --help` |
| Benchmark rendering | `uv run python -m scripts.benchmarks.benchmark_render_speed --help` |
| Train the earlier baseline | `uv run python -m modeling.legacy.train_world_model --config config/train_wm.yaml` |

Training and rollout commands need the data/checkpoints described in the
[README](../README.md). Some exploratory scripts keep the original campaign's
paths and CUDA settings near the top of the file; adjust those for your own run.
They are not part of the automated test suite, and many execute immediately
when run, without an argument parser or `--help` mode.

The `flockdit_streaming_20260715` root entry is a Git symlink to the original
checkpoint location under `output/`. On Windows, a checkout with symlinks
disabled may materialize it as a text file. In that case, point `RUN_ROOT` in
`modeling/eval/experiments.py` at your checkpoint directory before comparing
experiments.

## Migration from root-level scripts

Root-level commands have moved. Replace `python name.py` with the corresponding
`python -m package.name` command; CLI options and experiment YAML paths are
unchanged. The redundant `train_world_model.py` wrapper and `modeling/train.py`
have been consolidated into `modeling/legacy/train_world_model.py`.

| Previous files | New location |
| --- | --- |
| `data_recording.py` | `flockworld/cli/` |
| `train_vae.py`, `train_flock_dit.py`, `precompute_latents.py` | `modeling/cli/` |
| `eval_*.py`, `compare_experiments.py` | `modeling/cli/` |
| `analyze_*.py`, `probe_*.py` | `scripts/analysis/` |
| `diag_*.py`, `diagnose_*.py`, `validate_*.py`, `tune_heading_identity.py`, `overfit_rollout.py` | `scripts/diagnostics/` |
| `test_*.py` | `scripts/diagnostics/`, with the `test_` prefix removed |
| `gen_*.py`, `plot_eval_sweep.py` | `scripts/figures/` |
| `scripts/benchmark_render_speed.py` | `scripts/benchmarks/` |

For example, `python test_vae_roundtrip.py ...` becomes
`python -m scripts.diagnostics.vae_roundtrip ...`, and
`python probe_action_influence.py ...` becomes
`python -m scripts.analysis.probe_action_influence ...`.

Shared functions formerly imported from root scripts now live in:

- `modeling.eval.common`: `build_model`, `find_best_checkpoint`, `write_mp4`.
- `modeling.models.decoding`: `build_decode_fn`.
- `modeling.eval.multi`: multi-agent rollout, detection, metrics, and CSV helpers.
- `modeling.eval.experiments`: experiment configurations and checkpoint paths.

Saved evaluation JSON/CSV files preserve the original run provenance and may
still refer to the old command names. Use this mapping to reproduce those runs.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q flockworld modeling scripts tests
```

The automated checks exercise command imports/help, config loading, checkpoint
selection, and a small CPU model rollout. They do not retrain models or require
downloaded paper checkpoints. Manual dataset-dependent diagnostics belong in
`scripts/diagnostics/`, rather than files named `test_*.py` at the root.

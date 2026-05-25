# FlockWorld

JAX-based boid flocking simulation wrapped as a Gymnasium environment.
The default configuration is tuned for offline video generation: 1500 boids,
reflecting canvas boundaries with an agent-size inset, speed-hue tinting, and
the original five-point JS/PixiJS dart shape.
Designed for generating training videos for video-generation world models
with realistic multi-agent behavior.

## Installation

```bash
uv sync            # or: pip install -e .
```

Requires Python >= 3.13. On Windows JAX runs on CPU; on Linux it will
automatically use CUDA 12.

## Quick start

Record a 30-second video with default settings:

```bash
python data_recording.py
```

Override any parameter via the CLI (OmegaConf dot-list syntax):

```bash
python data_recording.py boids.num_agents=100 canvas.width=512 canvas.height=512
python data_recording.py video.duration=5 video.fps=60 seed=123
python data_recording.py device=cpu
python data_recording.py video.chunk_size=64
python data_recording.py video.warmup=120
python data_recording.py generation.num_envs=8 video.duration=5
python data_recording.py trajectory.enabled=true
python data_recording.py video.backend=pynv video.chunk_size=256
python data_recording.py collection.enabled=true collection.total_episodes=100 generation.num_envs=8 collection.partial_agents=2
```

Output videos are written to `output/full_obs.mp4` (entire canvas) and
`output/partial_obs.mp4` (square crop around the controlled agent). Multi-env
headless runs use `generation.*_path_template` and write one full/partial pair
per environment.
Set `trajectory.enabled=true` to also save per-frame state/action trajectories
as `.parquet`. Each row is one recorded frame, with agent fields stored in wide
columns like `a1_pos_x`, `a1_vel_x`, `a1_acc_x`, and `a1_acc_y`.

Structured collection mode writes a timestamped dataset under `outputs/` with
`settings.yaml`, `metadata.json`, `video_global/00000.mp4`,
`video_a1/00000.mp4`, additional partial-agent folders up to
`collection.partial_agents`, and per-episode trajectory files under
`trajectory/`.

For maximum NVIDIA throughput, `video.backend=pynv` keeps batched JAX frames on
the GPU and encodes with PyNvVideoCodec/NVENC through DLPack. It requires
`torch`, `PyNvVideoCodec`, `ffmpeg` for `.mp4` muxing, and a chunked run
(`video.chunk_size > 1`) with an uncontrolled or straight controlled-agent
policy.

## Gymnasium API

```python
from omegaconf import OmegaConf

from flockworld.env.gym_wrapper import FlockEnv

cfg = OmegaConf.load("config/data_recording.yaml")
cfg.canvas.width = 256
cfg.canvas.height = 256
cfg.boids.num_agents = 20

env = FlockEnv(cfg=cfg)
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()   # random heading angle
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()
```

- **Observation**: `(H, W, 3)` uint8 RGB frame
- **Action**: `Box(-pi, pi, shape=(1,))` — heading angle when `env.controlled_agent=true`
- **Reward**: `0.0` placeholder (to be defined)

## Configuration

All defaults live in `config/data_recording.yaml`. See the full argument and config reference in [docs/CONFIG.md](/home/wangy/wsl_projects/flockworld/docs/CONFIG.md).

For a fixed color theme instead of JS speed-hue tinting:

```bash
python data_recording.py rendering.color_mode=fixed rendering.agent_color=[0.7,0.9,1.0]
```

## Render benchmark

Compare CPU and GPU rendering throughput across agent counts:

```bash
python scripts/benchmark_render_speed.py --frames 120 --agents 100,250,500,1000,1500
```

The benchmark writes `output/benchmarks/render_speed.json`,
`output/benchmarks/render_speed.csv`, and
`output/benchmarks/render_speed.svg`. It also writes
`output/benchmarks/render_speed_logx.svg` with a logarithmic x-axis.

## Project structure

```
flockworld/
  core/
    types.py          State dataclasses (BoidState, EnvState)
    boids.py          Flocking rules: separation, alignment, cohesion
  rendering/
    primitives.py     SDF math: smoothstep, rotate_2d, sd_triangle
    renderer.py       Compose full frame from BoidState
  env/
    flock_env.py      Pure-JAX reset / step functions
    gym_wrapper.py    gymnasium.Env subclass
  video/
    recorder.py       Full-obs & partial-obs video writer
config/
  data_recording.yaml Data recording defaults
data_recording.py     CLI entry point
```

## World-model training (Solaris baseline)

The `modeling/` package is an adaptation of the Solaris multi-agent video
world-model (Wan-2.1-style flow-matching diffusion in JAX/Flax-nnx) to the
FlockWorld data format. By default it trains the single-player variant on one
chosen agent's 64×64 partial view, conditioning on that agent's steering
acceleration.

Dataset layout the trainer expects (already produced by `data_recording.py`):

```
data/recording/<timestamp>/
  metadata.json
  video_a1/00000.mp4    # one chosen agent's partial view per episode
  trajectory/00000.parquet   # contains a{k}_acc_x/a{k}_acc_y columns
```

Smoke-test one training step on CPU:

```bash
uv sync
JAX_PLATFORMS=cpu uv run python train_world_model.py \
  device.batch_size=1 \
  num_frames_context=9 \
  runner.params.total_steps=1 \
  device.num_workers=0
```

On a GPU host you can drop `JAX_PLATFORMS=cpu` and raise the batch/steps
counts. Common overrides:

```bash
# Train for 10k steps on GPU
uv run python train_world_model.py runner.params.total_steps=10000

# Use a different agent's view
uv run python train_world_model.py dataset.additional_params.agent_index=5

# Point at a different recording timestamp
uv run python train_world_model.py dataset.train_dataset_name=20260520_090911
```

### Pretrained Wan VAE / CLIP weights

The Solaris architecture uses a frozen Wan-2.1 VAE and the WanX image-CLIP
encoder. If `pretrained/vae.pt` and `pretrained/clip.pt` (Orbax checkpoints)
exist they are restored; otherwise both modules are initialised randomly with
a warning and the run becomes a structural smoke test. To exercise the real
baseline, drop the converted Wan 2.1 checkpoints into `pretrained/` before
training.

## Extensibility

The codebase is structured to support future additions:

- **Per-agent parameters** — `BoidState` can carry a `params` array for individual behavior weights
- **State-based behavior** — swap the policy function passed to `step`
- **Agent types** — add a `type_id` array to `BoidState`; dispatch rendering colour and policy per type
- **Reward modeling** — replace the `reward = 0.0` placeholder with predator-prey or other reward functions

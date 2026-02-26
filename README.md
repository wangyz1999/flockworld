# FlockWorld

JAX-based boid flocking simulation wrapped as a Gymnasium environment.
Designed for generating training videos for video-generation world models
with realistic multi-agent behavior.

## Installation

```bash
uv sync            # or: pip install -e .
```

Requires Python >= 3.13. On Windows JAX runs on CPU; on Linux it will
automatically use CUDA 12.

## Quick start

Record a 10-second video with default settings:

```bash
python main.py
```

Override any parameter via the CLI (OmegaConf dot-list syntax):

```bash
python main.py boids.num_agents=100 canvas.width=512 canvas.height=512
python main.py video.duration=5 video.fps=60 seed=123
```

Output videos are written to `output/full_obs.mp4` (entire canvas) and
`output/partial_obs.mp4` (square crop around the controlled agent).

## Gymnasium API

```python
from flockworld.env.gym_wrapper import FlockEnv
from flockworld.env.flock_env import EnvConfig

env = FlockEnv(env_config=EnvConfig(canvas_w=256, canvas_h=256, num_agents=20))
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()   # random heading angle
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()
```

- **Observation**: `(H, W, 3)` uint8 RGB frame
- **Action**: `Box(-pi, pi, shape=(1,))` — heading angle for the controlled agent (index 0)
- **Reward**: `0.0` placeholder (to be defined)

## Configuration

All defaults live in `config/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `canvas` | `width` / `height` | 800 | Canvas pixel dimensions |
| `boids` | `num_agents` | 50 | Number of boid agents |
| `boids` | `max_speed` / `min_speed` | 2.0 / 0.5 | Speed bounds |
| `boids` | `separation_radius` | 25.0 | Separation neighbourhood |
| `boids` | `alignment_radius` | 50.0 | Alignment neighbourhood |
| `boids` | `cohesion_radius` | 50.0 | Cohesion neighbourhood |
| `boids` | `*_weight` | 1.5 / 1.0 / 1.0 | Rule weights |
| `boids` | `agent_size` | 10.0 | Triangle size in pixels |
| `env` | `max_steps` | 1000 | Episode length |
| `env` | `boundary` | `wrap` | `wrap` or `reflect` |
| `rendering` | `background_color` | `[0.05, 0.05, 0.1]` | RGB background |
| `rendering` | `agent_color` | `[0.2, 0.8, 0.2]` | Default boid colour |
| `rendering` | `controlled_color` | `[1.0, 0.3, 0.3]` | Controlled agent colour |
| `video` | `fps` | 30 | Video frame rate |
| `video` | `duration` | 10.0 | Recording length (seconds) |
| `video` | `partial_obs_size` | 128 | Partial-obs crop side length |
| `seed` | | 42 | Random seed |

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
  default.yaml        All configurable defaults
main.py               CLI entry point
```

## Extensibility

The codebase is structured to support future additions:

- **Per-agent parameters** — `BoidState` can carry a `params` array for individual behavior weights
- **State-based behavior** — swap the policy function passed to `step`
- **Agent types** — add a `type_id` array to `BoidState`; dispatch rendering colour and policy per type
- **Reward modeling** — replace the `reward = 0.0` placeholder with predator-prey or other reward functions

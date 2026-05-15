# FlockWorld

JAX-based boid flocking simulation wrapped as a Gymnasium environment.
The default configuration clones the referenced JS/PixiJS flock behavior for
offline video generation: 1500 boids, JS wraparound, speed-hue tinting, and
the original five-point dart shape.
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
- **Action**: `Box(-pi, pi, shape=(1,))` — heading angle when `env.controlled_agent=true`
- **Reward**: `0.0` placeholder (to be defined)

## Configuration

All defaults live in `config/default.yaml`. See the full argument and config reference in [docs/CONFIG.md](/home/wangy/wsl_projects/flockworld/docs/CONFIG.md).

For a fixed color theme instead of JS speed-hue tinting:

```bash
python main.py rendering.color_mode=fixed rendering.agent_color=[0.7,0.9,1.0]
```

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

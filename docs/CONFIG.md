# Configuration Reference

FlockWorld loads `config/default.yaml` and applies CLI overrides using OmegaConf dot-list syntax:

```bash
python main.py boids.num_agents=100 canvas.width=512 canvas.height=512
python main.py rendering.color_mode=fixed rendering.agent_color=[0.7,0.9,1.0]
```

## Root

| Key | Default | Description |
| --- | --- | --- |
| `seed` | `42` | Random seed used for the initial flock state and policy state. |
| `device` | `"auto"` | JAX execution platform. Use `"auto"` for JAX's default selection, `"cpu"` to force CPU, or `"gpu"`/`"cuda"` to force GPU. |

## `generation`

| Key | Default | Description |
| --- | --- | --- |
| `generation.num_envs` | `1` | Number of independent homogeneous environments to generate in one headless run. Values above `1` batch simulation/rendering across environments. |
| `generation.seed_stride` | `1` | Environment `i` uses seed `seed + i * seed_stride`. |
| `generation.full_obs_path_template` | `"output/env_{env:04d}_seed_{seed}_full_obs.mp4"` | Full-observation output template for multi-env runs. Available fields: `{env}`, `{seed}`. |
| `generation.partial_obs_path_template` | `"output/env_{env:04d}_seed_{seed}_partial_obs.mp4"` | Partial-observation output template for multi-env runs. Available fields: `{env}`, `{seed}`. |

## `trajectory`

| Key | Default | Description |
| --- | --- | --- |
| `trajectory.enabled` | `false` | If `true`, save a `.npy` object dict containing per-frame per-agent trajectories after each simulated/rendered frame. |
| `trajectory.path` | `"output/trajectory.npy"` | Single-env trajectory output path. |
| `trajectory.path_template` | `"output/env_{env:04d}_seed_{seed}_trajectory.npy"` | Multi-env trajectory output template. Available fields: `{env}`, `{seed}`. |

Trajectory files contain `positions` `(T, N, 2)`, `velocities` `(T, N, 2)`,
`accelerations` `(T, N, 2)`, `headings` `(T, N)`, `actions` `(T, N)`, and
`step_count` `(T,)`. `actions` are applied movement headings in radians; for
uncontrolled flocking boids this is the heading produced by the boid update.

## `canvas`

| Key | Default | Description |
| --- | --- | --- |
| `canvas.width` | `720` | Full rendered frame width in pixels. |
| `canvas.height` | `720` | Full rendered frame height in pixels. |
| `canvas.partial` | `128` | Side length of the partial-observation crop centered on agent 0. |

## `boids`

| Key | Default | Description |
| --- | --- | --- |
| `boids.num_agents` | `1500` | Number of boids in the flock. |
| `boids.vision` | `25.0` | Neighbour radius in pixels for alignment, cohesion, and separation. |
| `boids.accuracy` | `32.0` | Approximate number of nearby candidate boids sampled per boid. `0` means all candidates. |
| `boids.alignment` | `1.1` | Weight applied to alignment steering. |
| `boids.alignment_bias` | `1.5` | Exponential bias for same-direction neighbour velocities: `bias ** dot(other.vel, self.vel)`. |
| `boids.cohesion` | `1.0` | Weight applied to cohesion steering. |
| `boids.separation` | `1.5` | Weight applied to separation steering. |
| `boids.max_force` | `0.2` | Maximum magnitude for each steering rule before rule weights are applied. |
| `boids.min_speed` | `1.0` | Minimum boid speed in pixels per tick. |
| `boids.max_speed` | `4.0` | Maximum boid speed in pixels per tick. |
| `boids.drag` | `0.005` | Per-tick velocity damping fraction. |
| `boids.noise` | `0.0` | Random heading perturbation scale. The angle range is `(pi / 80) * noise`. |
| `boids.agent_size` | `10.0` | JS dart size in pixels. `10.0` matches the original source points; other values scale it uniformly. Reflect boundaries keep centers inset by half this value so bodies stay visible at the canvas edge. |

## `env`

| Key | Default | Description |
| --- | --- | --- |
| `env.max_steps` | `1000` | Episode length in simulation ticks. |
| `env.boundary` | `"reflect"` | Boundary behavior. Use `"reflect"` for bounce with an `agent_size / 2` inset, or `"wrap"` for JS-style wraparound. |
| `env.controlled_agent` | `false` | If `true`, action/policy overrides agent 0's velocity heading. If `false`, all boids follow flocking only. |
| `env.agent_policy` | `"straight"` | Policy used for agent 0 when `env.controlled_agent=true`. Options: `straight`, `random`, `perlin`, `circle`, `lissajous`. |
| `env.render` | `false` | If `true`, show a live OpenCV window. If `false`, record videos headlessly. |

## `rendering`

Color values are normalized RGB triples in `[0.0, 1.0]`.

| Key | Default | Description |
| --- | --- | --- |
| `rendering.background_color` | `[0.0862745, 0.0862745, 0.0862745]` | Frame background color. The default is JS `0x161616`. |
| `rendering.agent_color` | `[1.0, 1.0, 1.0]` | Fixed boid color used when `rendering.color_mode=fixed`. |
| `rendering.aa_blur` | `1.0` | Anti-aliasing transition width in pixels. |
| `rendering.boid_alpha` | `0.8` | Alpha used when compositing JS dart boids over the background. |
| `rendering.color_mode` | `"speed"` | JS dart color mode. Use `"speed"` for JS HSV speed tinting, or `"fixed"` to render every boid with `rendering.agent_color`. |

Examples:

```bash
python main.py rendering.color_mode=fixed rendering.agent_color=[1.0,1.0,1.0]
python main.py rendering.color_mode=fixed rendering.background_color=[0.0,0.0,0.0] rendering.agent_color=[0.2,0.8,1.0]
python main.py rendering.color_mode=speed
```

## `video`

| Key | Default | Description |
| --- | --- | --- |
| `video.backend` | `"opencv"` | Video writer backend. `opencv` keeps the existing CPU `mp4v` path. `pynv` uses DLPack to hand batched JAX CUDA frames to PyNvVideoCodec/NVENC; it requires `video.chunk_size > 1`, an NVIDIA GPU, `torch`, `PyNvVideoCodec`, and `ffmpeg` for `.mp4` muxing. |
| `video.codec` | `"h264"` | `pynv` backend codec: `h264`, `hevc`, or `av1`, subject to GPU support. |
| `video.gpu_id` | `0` | `pynv` backend GPU id. |
| `video.preset` | `"p1"` | `pynv` backend NVENC preset. `p1` favors throughput; higher values favor quality. |
| `video.bitrate` | `"20M"` | `pynv` backend target bitrate. |
| `video.fps` | `30` | Output video frame rate. |
| `video.duration` | `30.0` | Requested recording duration in seconds. |
| `video.warmup` | `60` | Number of simulation steps to run before recording video or trajectory frames. Warmup does not count toward `video.duration`; the engine steps `video.warmup + fps * duration` times unless clipped by `env.max_steps`. |
| `video.chunk_size` | `256` | Number of frames generated per JAX batch before copying to CPU for video encoding. Use `1` for the old per-frame path. In multi-env runs, the generated chunk has shape `(chunk_size, generation.num_envs, H, W, 3)`. |
| `video.full_obs_path` | `"output/full_obs.mp4"` | Output path for the full-frame video. |
| `video.partial_obs_path` | `"output/partial_obs.mp4"` | Output path for the crop centered on agent 0. |
| `video.full_obs_only` | `false` | If `true`, skip partial-observation video output. |
| `video.partial_only` | `false` | If `true`, skip full-observation video output. |

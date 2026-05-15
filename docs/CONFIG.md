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
| `boids.separation` | `1.1` | Weight applied to separation steering. |
| `boids.max_force` | `0.2` | Maximum magnitude for each steering rule before rule weights are applied. |
| `boids.min_speed` | `1.0` | Minimum boid speed in pixels per tick. |
| `boids.max_speed` | `4.0` | Maximum boid speed in pixels per tick. |
| `boids.drag` | `0.005` | Per-tick velocity damping fraction. |
| `boids.noise` | `1.0` | Random heading perturbation scale. The angle range is `(pi / 80) * noise`. |
| `boids.agent_size` | `10.0` | Size used by legacy `simple` and `fancy` renderers. The JS dart shape has fixed source points. |

## `env`

| Key | Default | Description |
| --- | --- | --- |
| `env.max_steps` | `1000` | Episode length in simulation ticks. |
| `env.boundary` | `"wrap"` | Boundary behavior. Use `"wrap"` for JS-style wraparound or `"reflect"` for bounce. |
| `env.controlled_agent` | `false` | If `true`, action/policy overrides agent 0's velocity heading. If `false`, all boids follow flocking only. |
| `env.agent_policy` | `"straight"` | Policy used for agent 0 when `env.controlled_agent=true`. Options: `straight`, `random`, `perlin`, `circle`, `lissajous`. |
| `env.render` | `false` | If `true`, show a live OpenCV window. If `false`, record videos headlessly. |

## `rendering`

Color values are normalized RGB triples in `[0.0, 1.0]`.

| Key | Default | Description |
| --- | --- | --- |
| `rendering.background_color` | `[0.0862745, 0.0862745, 0.0862745]` | Frame background color. The default is JS `0x161616`. |
| `rendering.agent_color` | `[1.0, 1.0, 1.0]` | Fixed boid color used when `rendering.color_mode=fixed`, and by legacy renderers. |
| `rendering.controlled_color` | `[1.0, 1.0, 1.0]` | Agent 0 color used by legacy `simple` and `fancy` renderers. |
| `rendering.aa_blur` | `1.0` | Anti-aliasing transition width. Pixel units for `agent_shape=js`; UV units for legacy renderers. |
| `rendering.boid_alpha` | `0.8` | Alpha used when compositing JS dart boids over the background. |
| `rendering.agent_shape` | `"js"` | Renderer shape. Options: `js`, `simple`, `fancy`. |
| `rendering.color_mode` | `"speed"` | JS dart color mode. Use `"speed"` for JS HSV speed tinting, or `"fixed"` to render every boid with `rendering.agent_color`. |
| `rendering.flap_wings` | `false` | Animate wing spread for `agent_shape=fancy`; ignored by `js` and `simple`. |

Examples:

```bash
python main.py rendering.color_mode=fixed rendering.agent_color=[1.0,1.0,1.0]
python main.py rendering.color_mode=fixed rendering.background_color=[0.0,0.0,0.0] rendering.agent_color=[0.2,0.8,1.0]
python main.py rendering.color_mode=speed
```

## `video`

| Key | Default | Description |
| --- | --- | --- |
| `video.fps` | `30` | Output video frame rate. |
| `video.duration` | `30.0` | Requested recording duration in seconds. |
| `video.full_obs_path` | `"output/full_obs.mp4"` | Output path for the full-frame video. |
| `video.partial_obs_path` | `"output/partial_obs.mp4"` | Output path for the crop centered on agent 0. |
| `video.full_obs_only` | `false` | If `true`, skip partial-observation video output. |
| `video.partial_only` | `false` | If `true`, skip full-observation video output. |

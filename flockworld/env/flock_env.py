"""Pure-functional JAX environment for the boid flocking simulation.

All hot-path functions are ``@jax.jit``-compiled.  The mutable Gymnasium
wrapper lives in ``gym_wrapper.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import jax
import jax.numpy as jnp

from flockworld.core.types import BoidState, EnvState
from flockworld.core.boids import compute_boid_steering, update_boids
from flockworld.rendering.renderer import render_frame, _hsv_to_rgb


@dataclass(frozen=True)
class EnvConfig:
    """Static configuration extracted from OmegaConf for use inside JIT."""

    device: str
    canvas_w: int
    canvas_h: int
    num_agents: int
    vision: float
    accuracy: float
    alignment: float
    alignment_bias: float
    cohesion: float
    separation: float
    max_force: float
    min_speed: float
    max_speed: float
    drag: float
    noise: float
    agent_size: float
    max_steps: int
    boundary: str
    controlled_agent: bool
    background_color: tuple
    agent_color: tuple
    aa_blur: float
    color_mode: str
    boid_alpha: float
    dt: float
    border_width: int
    border_color: tuple
    num_camera_agents: int      # first N boids get identity colors in "agent_id" mode
    background_gradient: bool
    gradient_brightness: float


def env_config_from_omega(cfg) -> EnvConfig:
    """Build an ``EnvConfig`` from a full OmegaConf DictConfig."""
    collection = cfg.get("collection", None)
    num_camera_agents = (
        int(collection.partial_agents)
        if collection is not None and collection.get("partial_agents", None) is not None
        else int(cfg.boids.num_agents)
    )
    return EnvConfig(
        device=cfg.device,
        canvas_w=cfg.canvas.width,
        canvas_h=cfg.canvas.height,
        num_agents=cfg.boids.num_agents,
        vision=cfg.boids.vision,
        accuracy=cfg.boids.accuracy,
        alignment=cfg.boids.alignment,
        alignment_bias=cfg.boids.alignment_bias,
        cohesion=cfg.boids.cohesion,
        separation=cfg.boids.separation,
        max_force=cfg.boids.max_force,
        min_speed=cfg.boids.min_speed,
        max_speed=cfg.boids.max_speed,
        drag=cfg.boids.drag,
        noise=cfg.boids.noise,
        agent_size=cfg.boids.agent_size,
        max_steps=cfg.env.max_steps,
        boundary=cfg.env.boundary,
        controlled_agent=cfg.env.controlled_agent,
        dt=cfg.env.dt,
        background_color=tuple(cfg.rendering.background_color),
        agent_color=tuple(cfg.rendering.agent_color),
        aa_blur=cfg.rendering.aa_blur,
        color_mode=cfg.rendering.color_mode,
        boid_alpha=cfg.rendering.boid_alpha,
        border_width=int(cfg.rendering.get("border_width", 0)),
        border_color=tuple(cfg.rendering.get("border_color", (1.0, 1.0, 1.0))),
        num_camera_agents=num_camera_agents,
        background_gradient=bool(cfg.rendering.get("background_gradient", False)),
        gradient_brightness=float(cfg.rendering.get("gradient_brightness", 0.3)),
    )


def sample_boid_colors(key: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """Random per-boid hues at full saturation and value -> ``(N, 3)`` float32 RGB.

    Hue is uniform over the full circle; S and V are pinned to 1.0, which is HSL
    lightness 0.5 for every hue. So the colour varies only in hue, at the same
    lightness the ``agent_id`` identity hues already use. (Perceived luminance
    still differs between hues — yellow reads brighter than blue — since that is
    a property of sRGB, not of the sampling.)
    """
    hues = jax.random.uniform(key, (int(num_agents),), dtype=jnp.float32)
    ones = jnp.ones((int(num_agents),), dtype=jnp.float32)
    return _hsv_to_rgb(hues, ones, ones)


# ── JAX constants from config (created once, reused) ────────────────────

class EnvParams:
    """Pre-converted JAX arrays from EnvConfig, avoiding repeated conversion."""

    def __init__(self, ec: EnvConfig):
        self.ec = ec
        self.canvas_w = jnp.float32(ec.canvas_w)
        self.canvas_h = jnp.float32(ec.canvas_h)
        self.vision = jnp.float32(ec.vision)
        self.accuracy = jnp.float32(ec.accuracy)
        self.alignment = jnp.float32(ec.alignment)
        self.alignment_bias = jnp.float32(ec.alignment_bias)
        self.cohesion = jnp.float32(ec.cohesion)
        self.separation = jnp.float32(ec.separation)
        self.max_force = jnp.float32(ec.max_force)
        self.min_speed = jnp.float32(ec.min_speed)
        self.max_speed = jnp.float32(ec.max_speed)
        self.drag = jnp.float32(ec.drag)
        self.noise = jnp.float32(ec.noise)
        self.dt = jnp.float32(ec.dt)
        self.agent_size = jnp.float32(ec.agent_size)
        self.agent_render_radius = max(
            1,
            ceil(ec.agent_size * 0.75 + ec.aa_blur + 1.0),
        )
        self.boundary_padding = jnp.float32(
            ec.agent_size * 0.7 + ec.border_width if ec.boundary == "reflect" else 0.0
        )
        self.aa_blur = jnp.float32(ec.aa_blur)
        self.boid_alpha = jnp.float32(ec.boid_alpha)
        self.agent_color = jnp.array(ec.agent_color, dtype=jnp.float32)
        self.background_color = jnp.array(ec.background_color, dtype=jnp.float32)
        self.boundary = ec.boundary
        self.controlled_agent = ec.controlled_agent
        self.color_mode = ec.color_mode
        self.border_width = int(ec.border_width)
        self.border_color = jnp.array(ec.border_color, dtype=jnp.float32)

        # Per-boid identity colors (used only when color_mode == "agent_id"): the
        # first num_camera_agents boids get evenly-spaced hues at full S,V; the rest
        # stay white. So the camera agents are individually identifiable by hue.
        n_cam = min(int(ec.num_camera_agents), int(ec.num_agents))
        if ec.color_mode == "agent_id" and n_cam > 0:
            hues = jnp.arange(n_cam, dtype=jnp.float32) / float(n_cam)
            cam = _hsv_to_rgb(hues, jnp.ones(n_cam), jnp.ones(n_cam))          # (n_cam, 3)
            rest = jnp.ones((int(ec.num_agents) - n_cam, 3), dtype=jnp.float32)
            self.boid_colors = jnp.concatenate([cam, rest], axis=0)
        elif ec.color_mode == "random_hue":
            # Placeholder draw. Callers that vary colour per episode pass their own
            # via ``sample_boid_colors`` -> ``render(..., boid_colors=...)``; this
            # keeps single-shot callers (recorder, eval) rendering a valid frame.
            self.boid_colors = sample_boid_colors(jax.random.PRNGKey(0), ec.num_agents)
        else:
            self.boid_colors = jnp.ones((int(ec.num_agents), 3), dtype=jnp.float32)

        # Background image (H, W, 3): a dim position-encoding gradient, or a constant.
        # Gradient: pixel (x, y) -> gradient_brightness * (x/(W-1), y/(H-1), 0), so the
        # Red channel encodes x and Green encodes y (decode in eval: x = R/brightness*(W-1)).
        if ec.background_gradient:
            xs = jnp.linspace(0.0, 1.0, int(ec.canvas_w))
            ys = jnp.linspace(0.0, 1.0, int(ec.canvas_h))
            xv, yv = jnp.meshgrid(xs, ys)                                      # (H, W)
            self.background_image = (
                jnp.stack([xv, yv, jnp.zeros_like(xv)], axis=-1)
                * jnp.float32(ec.gradient_brightness)
            ).astype(jnp.float32)
        else:
            self.background_image = jnp.broadcast_to(
                self.background_color, (int(ec.canvas_h), int(ec.canvas_w), 3)
            )


# ── reset / step ────────────────────────────────────────────────────────

def reset(key: jnp.ndarray, ec: EnvConfig) -> EnvState:
    """Initialise a new episode with random boid positions and velocities."""
    k1, k2, k3, k_state = jax.random.split(key, 4)
    boundary_padding = (
        ec.agent_size * 0.5 + ec.border_width if ec.boundary == "reflect" else float(ec.border_width)
    )
    max_x = max(boundary_padding, float(ec.canvas_w) - boundary_padding)
    max_y = max(boundary_padding, float(ec.canvas_h) - boundary_padding)

    positions = jax.random.uniform(
        k1, (ec.num_agents, 2),
        minval=jnp.array([boundary_padding, boundary_padding]),
        maxval=jnp.array([max_x, max_y]),
    )
    angles = jax.random.uniform(k2, (ec.num_agents,), minval=-jnp.pi, maxval=jnp.pi)
    speed = jax.random.uniform(
        k3, (ec.num_agents,), minval=ec.min_speed, maxval=ec.max_speed,
    )
    velocities = jnp.stack([jnp.cos(angles) * speed, jnp.sin(angles) * speed], axis=-1)
    accelerations = jnp.zeros_like(velocities)
    headings = jnp.arctan2(velocities[:, 1], velocities[:, 0])

    boids = BoidState(
        positions=positions, velocities=velocities,
        accelerations=accelerations, headings=headings,
    )
    return EnvState(boids=boids, step_count=0, key=k_state)


def step(state: EnvState, action: jnp.ndarray, p: EnvParams):
    """Advance the environment by one tick."""
    controlled_vel = jnp.array([
        jnp.cos(action) * p.max_speed,
        jnp.sin(action) * p.max_speed,
    ])

    new_pos, new_vel, new_headings, new_key = update_boids(
        state.boids.positions, state.boids.velocities,
        state.boids.accelerations, controlled_vel,
        state.key,
        p.dt, p.min_speed, p.max_speed,
        p.drag, p.noise,
        p.canvas_w, p.canvas_h, p.boundary_padding, p.boundary,
        p.controlled_agent,
    )

    new_key, k_accuracy = jax.random.split(new_key)
    new_acc = compute_boid_steering(
        new_pos, new_vel, k_accuracy,
        p.vision, p.accuracy,
        p.alignment, p.alignment_bias,
        p.cohesion, p.separation,
        p.max_speed, p.max_force,
        p.canvas_w, p.canvas_h, p.boundary,
    )

    new_boids = BoidState(
        positions=new_pos, velocities=new_vel,
        accelerations=new_acc, headings=new_headings,
    )
    new_step = state.step_count + 1
    new_state = EnvState(boids=new_boids, step_count=new_step, key=new_key)

    done = new_step >= p.ec.max_steps
    reward = 0.0
    info = {}

    return new_state, reward, done, info


def render(
    state: EnvState,
    p: EnvParams,
    uv_grid: jnp.ndarray,
    boid_colors: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Render the current state to an (H, W, 3) float32 image (JIT-compiled).

    ``boid_colors`` overrides ``p.boid_colors`` for this frame — used by
    ``color_mode="random_hue"`` to vary hues per episode without rebuilding
    ``EnvParams`` or retracing the renderer. Ignored unless the colour mode reads
    per-boid colours (``agent_id`` / ``random_hue``).
    """
    return render_frame(
        state.boids.positions,
        state.boids.velocities,
        uv_grid,
        p.agent_size,
        p.agent_render_radius,
        p.agent_color,
        p.boid_colors if boid_colors is None else boid_colors,
        p.background_image,
        p.aa_blur,
        p.max_speed, p.boid_alpha,
        p.color_mode,
        p.border_width, p.border_color,
    )

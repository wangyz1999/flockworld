"""Pure-functional JAX environment for the boid flocking simulation.

All functions are free of side-effects and designed for ``jax.jit``.
The mutable Gymnasium wrapper lives in ``gym_wrapper.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from flockworld.core.types import BoidState, EnvState
from flockworld.core.boids import compute_boid_steering, update_boids
from flockworld.rendering.renderer import build_uv_grid, render_frame


@dataclass(frozen=True)
class EnvConfig:
    """Static configuration extracted from OmegaConf for use inside JIT."""

    canvas_w: int = 800
    canvas_h: int = 800
    num_agents: int = 50
    max_speed: float = 2.0
    min_speed: float = 0.5
    separation_radius: float = 25.0
    alignment_radius: float = 50.0
    cohesion_radius: float = 50.0
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    agent_size: float = 10.0
    max_steps: int = 1000
    boundary: str = "wrap"
    background_color: tuple = (0.05, 0.05, 0.1)
    agent_color: tuple = (0.2, 0.8, 0.2)
    controlled_color: tuple = (1.0, 0.3, 0.3)
    aa_blur: float = 0.005
    dt: float = 1.0


def env_config_from_omega(cfg) -> EnvConfig:
    """Build an ``EnvConfig`` from a full OmegaConf DictConfig."""
    return EnvConfig(
        canvas_w=cfg.canvas.width,
        canvas_h=cfg.canvas.height,
        num_agents=cfg.boids.num_agents,
        max_speed=cfg.boids.max_speed,
        min_speed=cfg.boids.min_speed,
        separation_radius=cfg.boids.separation_radius,
        alignment_radius=cfg.boids.alignment_radius,
        cohesion_radius=cfg.boids.cohesion_radius,
        separation_weight=cfg.boids.separation_weight,
        alignment_weight=cfg.boids.alignment_weight,
        cohesion_weight=cfg.boids.cohesion_weight,
        agent_size=cfg.boids.agent_size,
        max_steps=cfg.env.max_steps,
        boundary=cfg.env.boundary,
        background_color=tuple(cfg.rendering.background_color),
        agent_color=tuple(cfg.rendering.agent_color),
        controlled_color=tuple(cfg.rendering.controlled_color),
        aa_blur=cfg.rendering.aa_blur,
    )


# ── reset / step ────────────────────────────────────────────────────────

def reset(key: jnp.ndarray, ec: EnvConfig) -> EnvState:
    """Initialise a new episode with random boid positions and velocities."""
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.uniform(
        k1, (ec.num_agents, 2),
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([ec.canvas_w, ec.canvas_h]),
    )
    angles = jax.random.uniform(k2, (ec.num_agents,), minval=-jnp.pi, maxval=jnp.pi)
    speed = jax.random.uniform(
        k3, (ec.num_agents,), minval=ec.min_speed, maxval=ec.max_speed,
    )
    velocities = jnp.stack([jnp.sin(angles) * speed, jnp.cos(angles) * speed], axis=-1)
    headings = jnp.arctan2(velocities[:, 0], velocities[:, 1])

    boids = BoidState(positions=positions, velocities=velocities, headings=headings)
    return EnvState(boids=boids, step_count=0, key=key)


class _SteeringCfg:
    """Lightweight attribute-bag passed to ``compute_boid_steering``."""

    def __init__(self, ec: EnvConfig):
        self.boundary = ec.boundary
        self.canvas_w = float(ec.canvas_w)
        self.canvas_h = float(ec.canvas_h)
        self.separation_radius = ec.separation_radius
        self.alignment_radius = ec.alignment_radius
        self.cohesion_radius = ec.cohesion_radius
        self.separation_weight = ec.separation_weight
        self.alignment_weight = ec.alignment_weight
        self.cohesion_weight = ec.cohesion_weight


def step(state: EnvState, action: float, ec: EnvConfig):
    """Advance the environment by one tick.

    Parameters
    ----------
    state : EnvState
    action : float  — direction angle (radians) for the controlled agent.
    ec : EnvConfig

    Returns
    -------
    next_state, observation, reward, done, info
    """
    scfg = _SteeringCfg(ec)
    acc = compute_boid_steering(state.boids, scfg)

    # Controlled agent velocity from action angle
    controlled_vel = jnp.array([
        jnp.sin(action) * ec.max_speed,
        jnp.cos(action) * ec.max_speed,
    ])

    new_boids = update_boids(
        state.boids, acc, controlled_vel, ec.dt,
        ec.min_speed, ec.max_speed,
        float(ec.canvas_w), float(ec.canvas_h),
        ec.boundary,
    )

    new_step = state.step_count + 1
    new_state = EnvState(boids=new_boids, step_count=new_step, key=state.key)

    done = new_step >= ec.max_steps
    reward = 0.0  # placeholder
    info = {}

    return new_state, reward, done, info


def render(state: EnvState, ec: EnvConfig, uv_grid: jnp.ndarray) -> jnp.ndarray:
    """Render the current state to an (H, W, 3) float32 image."""
    return render_frame(
        state.boids,
        uv_grid,
        ec.canvas_w,
        ec.canvas_h,
        ec.agent_size,
        jnp.array(ec.agent_color),
        jnp.array(ec.controlled_color),
        jnp.array(ec.background_color),
        ec.aa_blur,
    )

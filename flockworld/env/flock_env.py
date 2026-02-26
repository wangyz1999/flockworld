"""Pure-functional JAX environment for the boid flocking simulation.

All hot-path functions are ``@jax.jit``-compiled.  The mutable Gymnasium
wrapper lives in ``gym_wrapper.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from flockworld.core.types import BoidState, EnvState
from flockworld.core.boids import compute_boid_steering, update_boids
from flockworld.rendering.renderer import render_frame


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
    min_separation: float = 25.0
    max_turn_rate: float = 0.15
    max_steps: int = 1000
    boundary: str = "wrap"
    background_color: tuple = (0.05, 0.05, 0.1)
    agent_color: tuple = (0.2, 0.8, 0.2)
    controlled_color: tuple = (1.0, 0.3, 0.3)
    aa_blur: float = 0.005
    agent_shape: str = "simple"
    flap_wings: bool = False
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
        min_separation=cfg.boids.min_separation,
        max_turn_rate=cfg.boids.max_turn_rate,
        max_steps=cfg.env.max_steps,
        boundary=cfg.env.boundary,
        background_color=tuple(cfg.rendering.background_color),
        agent_color=tuple(cfg.rendering.agent_color),
        controlled_color=tuple(cfg.rendering.controlled_color),
        aa_blur=cfg.rendering.aa_blur,
        agent_shape=cfg.rendering.agent_shape,
        flap_wings=cfg.rendering.flap_wings,
    )


# ── JAX constants from config (created once, reused) ────────────────────

class EnvParams:
    """Pre-converted JAX arrays from EnvConfig, avoiding repeated conversion."""

    def __init__(self, ec: EnvConfig):
        self.ec = ec
        self.canvas_w = jnp.float32(ec.canvas_w)
        self.canvas_h = jnp.float32(ec.canvas_h)
        self.max_speed = jnp.float32(ec.max_speed)
        self.min_speed = jnp.float32(ec.min_speed)
        self.dt = jnp.float32(ec.dt)
        self.separation_radius = jnp.float32(ec.separation_radius)
        self.alignment_radius = jnp.float32(ec.alignment_radius)
        self.cohesion_radius = jnp.float32(ec.cohesion_radius)
        self.separation_weight = jnp.float32(ec.separation_weight)
        self.alignment_weight = jnp.float32(ec.alignment_weight)
        self.cohesion_weight = jnp.float32(ec.cohesion_weight)
        self.agent_size = jnp.float32(ec.agent_size)
        self.min_separation = jnp.float32(ec.min_separation)
        self.max_turn_rate = jnp.float32(ec.max_turn_rate)
        self.aa_blur = jnp.float32(ec.aa_blur)
        self.agent_color = jnp.array(ec.agent_color, dtype=jnp.float32)
        self.controlled_color = jnp.array(ec.controlled_color, dtype=jnp.float32)
        self.background_color = jnp.array(ec.background_color, dtype=jnp.float32)
        self.boundary = ec.boundary
        self.fancy_shape = ec.agent_shape == "fancy"
        self.flap_wings = ec.flap_wings


# ── reset / step ────────────────────────────────────────────────────────

def reset(key: jnp.ndarray, ec: EnvConfig) -> EnvState:
    """Initialise a new episode with random boid positions and velocities."""
    k1, k2, k3, k4 = jax.random.split(key, 4)

    positions = jax.random.uniform(
        k1, (ec.num_agents, 2),
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([float(ec.canvas_w), float(ec.canvas_h)]),
    )
    angles = jax.random.uniform(k2, (ec.num_agents,), minval=-jnp.pi, maxval=jnp.pi)
    speed = jax.random.uniform(
        k3, (ec.num_agents,), minval=ec.min_speed, maxval=ec.max_speed,
    )
    velocities = jnp.stack([jnp.sin(angles) * speed, jnp.cos(angles) * speed], axis=-1)
    headings = jnp.arctan2(velocities[:, 0], velocities[:, 1])

    phase_offsets = jax.random.uniform(
        k4, (ec.num_agents,), minval=0.0, maxval=2.0 * jnp.pi * 3.0,
    )

    boids = BoidState(
        positions=positions, velocities=velocities,
        headings=headings, phase_offsets=phase_offsets,
    )
    return EnvState(boids=boids, step_count=0, key=key)


def step(state: EnvState, action: jnp.ndarray, p: EnvParams):
    """Advance the environment by one tick."""
    acc = compute_boid_steering(
        state.boids.positions, state.boids.velocities,
        p.separation_radius, p.alignment_radius, p.cohesion_radius,
        p.separation_weight, p.alignment_weight, p.cohesion_weight,
        p.canvas_w, p.canvas_h, p.boundary,
    )

    controlled_vel = jnp.array([
        jnp.sin(action) * p.max_speed,
        jnp.cos(action) * p.max_speed,
    ])

    new_pos, new_vel, new_headings = update_boids(
        state.boids.positions, state.boids.velocities,
        acc, controlled_vel,
        p.dt, p.min_speed, p.max_speed, p.max_turn_rate,
        p.min_separation,
        p.canvas_w, p.canvas_h, p.boundary,
    )

    new_boids = BoidState(
        positions=new_pos, velocities=new_vel,
        headings=new_headings, phase_offsets=state.boids.phase_offsets,
    )
    new_step = state.step_count + 1
    new_state = EnvState(boids=new_boids, step_count=new_step, key=state.key)

    done = new_step >= p.ec.max_steps
    reward = 0.0
    info = {}

    return new_state, reward, done, info


def render(state: EnvState, p: EnvParams, uv_grid: jnp.ndarray) -> jnp.ndarray:
    """Render the current state to an (H, W, 3) float32 image (JIT-compiled)."""
    return render_frame(
        state.boids.positions,
        state.boids.velocities,
        state.boids.phase_offsets,
        jnp.float32(state.step_count),
        uv_grid,
        p.canvas_w, p.canvas_h,
        p.agent_size,
        p.agent_color, p.controlled_color, p.background_color,
        p.aa_blur,
        p.fancy_shape, p.flap_wings,
    )

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

    canvas_w: int = 720
    canvas_h: int = 720
    num_agents: int = 1500
    vision: float = 25.0
    accuracy: float = 32.0
    alignment: float = 1.1
    alignment_bias: float = 1.5
    cohesion: float = 1.0
    separation: float = 1.1
    max_force: float = 0.2
    min_speed: float = 1.0
    max_speed: float = 4.0
    drag: float = 0.005
    noise: float = 1.0
    agent_size: float = 10.0
    max_steps: int = 1000
    boundary: str = "wrap"
    controlled_agent: bool = False
    background_color: tuple = (0.0862745, 0.0862745, 0.0862745)
    agent_color: tuple = (1.0, 1.0, 1.0)
    controlled_color: tuple = (1.0, 1.0, 1.0)
    aa_blur: float = 1.0
    agent_shape: str = "js"
    color_mode: str = "speed"
    boid_alpha: float = 0.8
    flap_wings: bool = False
    dt: float = 1.0


def env_config_from_omega(cfg) -> EnvConfig:
    """Build an ``EnvConfig`` from a full OmegaConf DictConfig."""
    return EnvConfig(
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
        background_color=tuple(cfg.rendering.background_color),
        agent_color=tuple(cfg.rendering.agent_color),
        controlled_color=tuple(cfg.rendering.controlled_color),
        aa_blur=cfg.rendering.aa_blur,
        agent_shape=cfg.rendering.agent_shape,
        color_mode=cfg.rendering.color_mode,
        boid_alpha=cfg.rendering.boid_alpha,
        flap_wings=cfg.rendering.flap_wings,
    )


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
        self.aa_blur = jnp.float32(ec.aa_blur)
        self.boid_alpha = jnp.float32(ec.boid_alpha)
        self.agent_color = jnp.array(ec.agent_color, dtype=jnp.float32)
        self.controlled_color = jnp.array(ec.controlled_color, dtype=jnp.float32)
        self.background_color = jnp.array(ec.background_color, dtype=jnp.float32)
        self.boundary = ec.boundary
        self.controlled_agent = ec.controlled_agent
        self.agent_shape = ec.agent_shape
        self.color_mode = ec.color_mode
        self.flap_wings = ec.flap_wings


# ── reset / step ────────────────────────────────────────────────────────

def reset(key: jnp.ndarray, ec: EnvConfig) -> EnvState:
    """Initialise a new episode with random boid positions and velocities."""
    k1, k2, k3, k4, k_state = jax.random.split(key, 5)

    positions = jax.random.uniform(
        k1, (ec.num_agents, 2),
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([float(ec.canvas_w), float(ec.canvas_h)]),
    )
    angles = jax.random.uniform(k2, (ec.num_agents,), minval=-jnp.pi, maxval=jnp.pi)
    speed = jax.random.uniform(
        k3, (ec.num_agents,), minval=ec.min_speed, maxval=ec.max_speed,
    )
    velocities = jnp.stack([jnp.cos(angles) * speed, jnp.sin(angles) * speed], axis=-1)
    accelerations = jnp.zeros_like(velocities)
    headings = jnp.arctan2(velocities[:, 1], velocities[:, 0])

    phase_offsets = jax.random.uniform(
        k4, (ec.num_agents,), minval=0.0, maxval=2.0 * jnp.pi * 3.0,
    )

    boids = BoidState(
        positions=positions, velocities=velocities,
        accelerations=accelerations,
        headings=headings, phase_offsets=phase_offsets,
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
        p.canvas_w, p.canvas_h, p.boundary,
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
        accelerations=new_acc,
        headings=new_headings, phase_offsets=state.boids.phase_offsets,
    )
    new_step = state.step_count + 1
    new_state = EnvState(boids=new_boids, step_count=new_step, key=new_key)

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
        p.max_speed, p.boid_alpha,
        p.agent_shape, p.color_mode, p.flap_wings,
    )

"""Boid flocking physics implemented in pure JAX.

All functions are designed to be jit-compatible.  The controllable agent
(index 0) is handled by the caller — these helpers compute the autonomous
steering for the remaining agents.
"""

import jax
import jax.numpy as jnp

from flockworld.core.types import BoidState


# ── helpers ──────────────────────────────────────────────────────────────

def _pairwise_displacement(positions, boundary, canvas_w, canvas_h):
    """Compute (N, N, 2) displacement matrix respecting boundary mode.

    displacement[i, j] = positions[j] - positions[i], optionally wrapped
    for toroidal boundaries.
    """
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)
    if boundary == "wrap":
        size = jnp.array([canvas_w, canvas_h], dtype=jnp.float32)
        diff = jnp.mod(diff + size / 2, size) - size / 2
    return diff


def _pairwise_distance(displacement):
    """(N, N) Euclidean distances from displacement matrix."""
    return jnp.sqrt(jnp.sum(displacement ** 2, axis=-1) + 1e-8)


# ── flocking rules ──────────────────────────────────────────────────────

def separation(displacement, distance, radius, weight):
    """Steer away from nearby neighbours."""
    mask = (distance < radius) & (distance > 1e-6)  # (N, N)
    mask = mask.astype(jnp.float32)
    # repulsion direction: away from neighbour → negative displacement
    steer = -displacement / (distance[..., None] + 1e-8)  # (N, N, 2)
    steer = (steer * mask[..., None]).sum(axis=1)  # (N, 2)
    return steer * weight


def alignment(velocities, distance, radius, weight):
    """Match average heading of neighbours within *radius*."""
    mask = (distance < radius) & (distance > 1e-6)
    count = mask.astype(jnp.float32).sum(axis=1, keepdims=True).clip(min=1)
    avg_vel = (velocities[None, :, :] * mask[:, :, None].astype(jnp.float32)).sum(axis=1) / count
    steer = avg_vel - velocities
    return steer * weight


def cohesion(positions, displacement, distance, radius, weight):
    """Steer toward the centre of mass of neighbours within *radius*."""
    mask = (distance < radius) & (distance > 1e-6)
    count = mask.astype(jnp.float32).sum(axis=1, keepdims=True).clip(min=1)
    avg_disp = (displacement * mask[:, :, None].astype(jnp.float32)).sum(axis=1) / count
    return avg_disp * weight


# ── update step ─────────────────────────────────────────────────────────

def compute_boid_steering(boids: BoidState, cfg) -> jnp.ndarray:
    """Return (N, 2) acceleration for every agent based on boid rules.

    *cfg* is the OmegaConf boids sub-config (or any object with the
    expected attributes).
    """
    disp = _pairwise_displacement(
        boids.positions, cfg.boundary,
        cfg.canvas_w, cfg.canvas_h,
    )
    dist = _pairwise_distance(disp)

    acc = (
        separation(disp, dist, cfg.separation_radius, cfg.separation_weight)
        + alignment(boids.velocities, dist, cfg.alignment_radius, cfg.alignment_weight)
        + cohesion(boids.positions, disp, dist, cfg.cohesion_radius, cfg.cohesion_weight)
    )
    return acc


def update_boids(
    boids: BoidState,
    acceleration: jnp.ndarray,
    controlled_velocity: jnp.ndarray,
    dt: float,
    min_speed: float,
    max_speed: float,
    canvas_w: float,
    canvas_h: float,
    boundary: str,
) -> BoidState:
    """Advance all boids by one timestep.

    *controlled_velocity* is the (2,) velocity vector for agent 0 (set by
    the player action).  Autonomous agents use *acceleration*.
    """
    new_vel = boids.velocities + acceleration * dt

    # Override agent 0 with the controlled velocity
    new_vel = new_vel.at[0].set(controlled_velocity)

    # Clamp speed for autonomous agents (indices 1:)
    speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True) + 1e-8)
    clamped_speed = jnp.clip(speed, min_speed, max_speed)
    direction = new_vel / speed
    new_vel_clamped = direction * clamped_speed

    # Agent 0 keeps its own speed; others get clamped
    is_controlled = jnp.zeros((boids.positions.shape[0], 1))
    is_controlled = is_controlled.at[0].set(1.0)
    new_vel = new_vel * is_controlled + new_vel_clamped * (1.0 - is_controlled)

    new_pos = boids.positions + new_vel * dt

    # Boundary handling
    if boundary == "wrap":
        new_pos = jnp.mod(new_pos, jnp.array([canvas_w, canvas_h]))
    else:  # reflect
        new_pos_x = jnp.where(
            (new_pos[:, 0] < 0) | (new_pos[:, 0] > canvas_w),
            jnp.clip(new_pos[:, 0], 0, canvas_w),
            new_pos[:, 0],
        )
        new_pos_y = jnp.where(
            (new_pos[:, 1] < 0) | (new_pos[:, 1] > canvas_h),
            jnp.clip(new_pos[:, 1], 0, canvas_h),
            new_pos[:, 1],
        )
        new_pos = jnp.stack([new_pos_x, new_pos_y], axis=-1)
        # Reverse velocity component on reflection
        vel_x = jnp.where(
            (new_pos[:, 0] <= 0) | (new_pos[:, 0] >= canvas_w),
            -new_vel[:, 0], new_vel[:, 0],
        )
        vel_y = jnp.where(
            (new_pos[:, 1] <= 0) | (new_pos[:, 1] >= canvas_h),
            -new_vel[:, 1], new_vel[:, 1],
        )
        new_vel = jnp.stack([vel_x, vel_y], axis=-1)

    new_headings = jnp.arctan2(new_vel[:, 0], new_vel[:, 1])

    return BoidState(
        positions=new_pos,
        velocities=new_vel,
        headings=new_headings,
    )

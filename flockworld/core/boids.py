"""Boid flocking physics implemented in pure JAX.

All functions are jit-compatible.  The controllable agent (index 0) is
handled by the caller — these helpers compute the autonomous steering
for the remaining agents.
"""

from functools import partial

import jax
import jax.numpy as jnp


# ── helpers ──────────────────────────────────────────────────────────────

def _pairwise_displacement_wrap(positions, canvas_w, canvas_h):
    diff = positions[None, :, :] - positions[:, None, :]
    size = jnp.array([canvas_w, canvas_h], dtype=jnp.float32)
    return jnp.mod(diff + size / 2, size) - size / 2


def _pairwise_displacement_reflect(positions, canvas_w, canvas_h):
    return positions[None, :, :] - positions[:, None, :]


def _pairwise_distance(displacement):
    return jnp.sqrt(jnp.sum(displacement ** 2, axis=-1) + 1e-8)


# ── flocking rules ──────────────────────────────────────────────────────

def separation(displacement, distance, radius, weight):
    """Steer away from nearby neighbours with inverse-distance scaling.

    Closer agents produce exponentially stronger repulsion to prevent
    overlap and clumping.
    """
    mask = ((distance < radius) & (distance > 1e-6)).astype(jnp.float32)
    inv_dist = 1.0 / (distance + 1e-4)
    # Quadratic falloff: much stronger when very close
    strength = inv_dist * inv_dist
    steer = -displacement / (distance[..., None] + 1e-8) * strength[..., None]
    steer = (steer * mask[..., None]).sum(axis=1)
    return steer * weight


def alignment(velocities, distance, radius, weight):
    mask = ((distance < radius) & (distance > 1e-6)).astype(jnp.float32)
    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    avg_vel = (velocities[None, :, :] * mask[:, :, None]).sum(axis=1) / count
    steer = avg_vel - velocities
    return steer * weight


def cohesion(displacement, distance, radius, weight):
    mask = ((distance < radius) & (distance > 1e-6)).astype(jnp.float32)
    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    avg_disp = (displacement * mask[:, :, None]).sum(axis=1) / count
    return avg_disp * weight


# ── update step ─────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("boundary",))
def compute_boid_steering(
    positions, velocities,
    separation_radius, alignment_radius, cohesion_radius,
    separation_weight, alignment_weight, cohesion_weight,
    canvas_w, canvas_h, boundary,
):
    """Return (N, 2) acceleration for every agent based on boid rules."""
    if boundary == "wrap":
        disp = _pairwise_displacement_wrap(positions, canvas_w, canvas_h)
    else:
        disp = _pairwise_displacement_reflect(positions, canvas_w, canvas_h)

    dist = _pairwise_distance(disp)

    acc = (
        separation(disp, dist, separation_radius, separation_weight)
        + alignment(velocities, dist, alignment_radius, alignment_weight)
        + cohesion(disp, dist, cohesion_radius, cohesion_weight)
    )
    return acc


def enforce_min_separation(positions, min_dist, canvas_w, canvas_h, boundary):
    """Push overlapping boids apart by directly adjusting positions.

    This bypasses the turn-rate limiter so boids physically cannot overlap
    below ``min_dist`` pixels.  Applied *after* the velocity integration.
    """
    if boundary == "wrap":
        disp = _pairwise_displacement_wrap(positions, canvas_w, canvas_h)
    else:
        disp = _pairwise_displacement_reflect(positions, canvas_w, canvas_h)

    dist = _pairwise_distance(disp)

    overlap = ((dist < min_dist) & (dist > 1e-6)).astype(jnp.float32)
    penetration = (min_dist - dist).clip(min=0.0)

    direction = -disp / (dist[..., None] + 1e-8)
    push = direction * (penetration[..., None] * 0.5) * overlap[..., None]
    correction = push.sum(axis=1)

    return positions + correction


@partial(jax.jit, static_argnames=("boundary",))
def update_boids(
    positions, velocities,
    acceleration, controlled_velocity,
    dt, min_speed, max_speed, max_turn_rate,
    min_separation,
    canvas_w, canvas_h, boundary,
):
    """Advance all boids by one timestep.  Returns (new_pos, new_vel, new_headings).

    Agents steer by *rotating* their heading toward the desired direction
    (clamped by ``max_turn_rate`` per tick) and always move nose-first.
    This prevents sideways sliding.
    """
    headings = jnp.arctan2(velocities[:, 0], velocities[:, 1])

    # Desired velocity for autonomous agents
    desired_vel = velocities + acceleration * dt
    desired_heading = jnp.arctan2(desired_vel[:, 0], desired_vel[:, 1])
    desired_speed = jnp.sqrt(jnp.sum(desired_vel ** 2, axis=-1) + 1e-8)

    # Angular difference wrapped to [-pi, pi]
    delta = desired_heading - headings
    delta = jnp.arctan2(jnp.sin(delta), jnp.cos(delta))

    # Clamp turn rate so agents rotate gradually
    delta = jnp.clip(delta, -max_turn_rate, max_turn_rate)
    new_headings = headings + delta

    new_speed = jnp.clip(desired_speed, min_speed, max_speed)

    # Override controlled agent (index 0)
    ctrl_heading = jnp.arctan2(controlled_velocity[0], controlled_velocity[1])
    ctrl_speed = jnp.sqrt(jnp.sum(controlled_velocity ** 2) + 1e-8)
    new_headings = new_headings.at[0].set(ctrl_heading)
    new_speed = new_speed.at[0].set(ctrl_speed)

    # Reconstruct velocity from heading + speed (always nose-first)
    new_vel = jnp.stack([
        jnp.sin(new_headings) * new_speed,
        jnp.cos(new_headings) * new_speed,
    ], axis=-1)

    new_pos = positions + new_vel * dt

    if boundary == "wrap":
        new_pos = jnp.mod(new_pos, jnp.array([canvas_w, canvas_h]))
    else:
        clamped_x = jnp.clip(new_pos[:, 0], 0.0, canvas_w)
        clamped_y = jnp.clip(new_pos[:, 1], 0.0, canvas_h)
        hit_x = (new_pos[:, 0] < 0) | (new_pos[:, 0] > canvas_w)
        hit_y = (new_pos[:, 1] < 0) | (new_pos[:, 1] > canvas_h)
        new_pos = jnp.stack([clamped_x, clamped_y], axis=-1)
        vel_x = jnp.where(hit_x, -new_vel[:, 0], new_vel[:, 0])
        vel_y = jnp.where(hit_y, -new_vel[:, 1], new_vel[:, 1])
        new_vel = jnp.stack([vel_x, vel_y], axis=-1)
        new_headings = jnp.arctan2(new_vel[:, 0], new_vel[:, 1])

    new_pos = enforce_min_separation(
        new_pos, min_separation, canvas_w, canvas_h, boundary,
    )

    if boundary == "wrap":
        new_pos = jnp.mod(new_pos, jnp.array([canvas_w, canvas_h]))
    else:
        new_pos = jnp.stack([
            jnp.clip(new_pos[:, 0], 0.0, canvas_w),
            jnp.clip(new_pos[:, 1], 0.0, canvas_h),
        ], axis=-1)

    return new_pos, new_vel, new_headings

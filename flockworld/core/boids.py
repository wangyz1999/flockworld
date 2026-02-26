"""Boid flocking physics implemented in pure JAX.

All functions are jit-compatible.  The controllable agent (index 0) is
handled by the caller — these helpers compute the autonomous steering
for the remaining agents.

Uses Reynolds-style steering: each rule computes a *desired velocity*,
subtracts the current velocity to get a steering force, and limits
that force by ``max_force``.  This avoids head-on deadlocks because
the steering vector always has a lateral component.
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


def _limit(vec, max_mag):
    """Clamp a batch of 2-d vectors to ``max_mag`` length."""
    mag = jnp.sqrt(jnp.sum(vec ** 2, axis=-1, keepdims=True) + 1e-8)
    scale = jnp.where(mag > max_mag, max_mag / mag, 1.0)
    return vec * scale


# ── flocking rules (Reynolds steering) ───────────────────────────────────

def separation(displacement, distance, radius, weight, velocities, max_speed, max_force):
    """Steer away from nearby neighbours (Reynolds-style).

    Accumulates inverse-distance-weighted unit vectors pointing away from
    each neighbour, normalises, scales to ``max_speed`` to get the desired
    velocity, then subtracts the current velocity and limits by ``max_force``.
    """
    mask = ((distance < radius) & (distance > 1e-6)).astype(jnp.float32)
    diff = -displacement / (distance[..., None] + 1e-8)
    diff = diff / (distance[..., None] + 1e-4)
    steer = (diff * mask[..., None]).sum(axis=1)

    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    steer = steer / count

    mag = jnp.sqrt(jnp.sum(steer ** 2, axis=-1, keepdims=True) + 1e-8)
    desired = steer / mag * max_speed
    steer = jnp.where(mag > 1e-6, desired - velocities, 0.0)
    steer = _limit(steer, max_force)
    return steer * weight


def alignment(velocities, distance, radius, weight, max_speed, max_force):
    """Match average heading of neighbours (Reynolds-style)."""
    mask = ((distance < radius) & (distance > 1e-6)).astype(jnp.float32)
    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    avg_vel = (velocities[None, :, :] * mask[:, :, None]).sum(axis=1) / count

    mag = jnp.sqrt(jnp.sum(avg_vel ** 2, axis=-1, keepdims=True) + 1e-8)
    desired = avg_vel / mag * max_speed
    steer = desired - velocities
    steer = _limit(steer, max_force)
    return steer * weight


def cohesion(displacement, distance, radius, weight, velocities, max_speed, max_force):
    """Steer toward average position of neighbours (Reynolds-style seek)."""
    mask = ((distance < radius) & (distance > 1e-6)).astype(jnp.float32)
    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    avg_disp = (displacement * mask[:, :, None]).sum(axis=1) / count

    mag = jnp.sqrt(jnp.sum(avg_disp ** 2, axis=-1, keepdims=True) + 1e-8)
    desired = avg_disp / mag * max_speed
    steer = desired - velocities
    steer = _limit(steer, max_force)
    return steer * weight


# ── update step ─────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("boundary",))
def compute_boid_steering(
    positions, velocities,
    separation_radius, alignment_radius, cohesion_radius,
    separation_weight, alignment_weight, cohesion_weight,
    max_speed, max_force,
    canvas_w, canvas_h, boundary,
):
    """Return (N, 2) steering acceleration for every agent."""
    if boundary == "wrap":
        disp = _pairwise_displacement_wrap(positions, canvas_w, canvas_h)
    else:
        disp = _pairwise_displacement_reflect(positions, canvas_w, canvas_h)

    dist = _pairwise_distance(disp)

    acc = (
        separation(disp, dist, separation_radius, separation_weight,
                   velocities, max_speed, max_force)
        + alignment(velocities, dist, alignment_radius, alignment_weight,
                    max_speed, max_force)
        + cohesion(disp, dist, cohesion_radius, cohesion_weight,
                   velocities, max_speed, max_force)
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
    """Advance all boids by one timestep.  Returns (new_pos, new_vel, new_headings)."""
    headings = jnp.arctan2(velocities[:, 0], velocities[:, 1])

    new_vel = velocities + acceleration * dt
    new_vel = _limit(new_vel, max_speed)

    speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1) + 1e-8)
    speed = jnp.clip(speed, min_speed, max_speed)
    direction = new_vel / (jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True)) + 1e-8)

    desired_heading = jnp.arctan2(direction[:, 0], direction[:, 1])
    delta = desired_heading - headings
    delta = jnp.arctan2(jnp.sin(delta), jnp.cos(delta))
    delta = jnp.clip(delta, -max_turn_rate, max_turn_rate)
    new_headings = headings + delta

    new_vel = jnp.stack([
        jnp.sin(new_headings) * speed,
        jnp.cos(new_headings) * speed,
    ], axis=-1)

    # Override controlled agent (index 0)
    ctrl_heading = jnp.arctan2(controlled_velocity[0], controlled_velocity[1])
    ctrl_speed = jnp.sqrt(jnp.sum(controlled_velocity ** 2) + 1e-8)
    new_headings = new_headings.at[0].set(ctrl_heading)
    new_vel = new_vel.at[0].set(controlled_velocity)

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

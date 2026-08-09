# Derived from cubeDhuang/boids (https://github.com/cubeDhuang/boids),
# MIT License, Copyright (c) 2023 Daniel Huang. See
# licenses/MIT-cubedhuang-boids.txt. This file has been modified: the algorithm
# is reimplemented in JAX as a batched update and the arena reflects at the
# walls instead of wrapping. See THIRD_PARTY_NOTICES.md.
"""Boid flocking physics implemented in pure JAX.

Faithfully reproduces the algorithm from the JS/PixiJS reference:
  - Single ``vision`` radius for neighbour detection
  - Alignment with velocity-dot-product ``bias``
  - Reynolds-style steering (desired velocity − current velocity, clamped
    by ``max_force``) for all three rules
  - Velocity drag, random heading noise, min/max speed clamping
  - No artificial turn-rate limiter — steering force + drag handle it

Optionally, index 0 can be overridden by the caller for controlled-agent runs.
"""

from functools import partial

import jax
import jax.numpy as jnp


# ── helpers ──────────────────────────────────────────────────────────────

def _pairwise_displacement(positions):
    return positions[None, :, :] - positions[:, None, :]


def _sq_distance(displacement):
    return jnp.sum(displacement ** 2, axis=-1)


def _limit(vec, max_mag):
    """Clamp a batch of 2-d vectors to ``max_mag`` length."""
    sq = jnp.sum(vec ** 2, axis=-1, keepdims=True)
    mag = jnp.sqrt(sq + 1e-8)
    scale = jnp.where(sq > max_mag * max_mag, max_mag / mag, 1.0)
    return vec * scale


def _set_mag(vec, target):
    """Normalise then scale a batch of 2-d vectors to ``target`` length."""
    sq = jnp.sum(vec ** 2, axis=-1, keepdims=True)
    mag = jnp.sqrt(sq + 1e-8)
    return jnp.where(sq > 0.0, vec / mag * target, vec)


def _enforce_min_speed(vel, min_speed, key):
    """Ensure every velocity has at least ``min_speed`` magnitude."""
    sq = jnp.sum(vel ** 2, axis=-1, keepdims=True)
    mag = jnp.sqrt(sq + 1e-8)
    scale = jnp.where(sq < min_speed * min_speed, min_speed / mag, 1.0)
    lifted = vel * scale
    angles = jax.random.uniform(key, vel.shape[:1], minval=-jnp.pi, maxval=jnp.pi)
    random_vel = jnp.stack(
        [jnp.cos(angles) * min_speed, jnp.sin(angles) * min_speed], axis=-1,
    )
    return jnp.where(sq <= 1e-8, random_vel, lifted)


# ── flocking rules (exact JS reference algorithm) ───────────────────────

def _flock(
    displacement, sq_dist, velocities, candidate_mask, sq_vision, accuracy, key,
    alignment_weight, alignment_bias,
    cohesion_weight, separation_weight,
    max_speed, max_force,
):
    """Compute per-boid acceleration from all three flocking rules.

    Mirrors the JS ``Boid.flock()`` method exactly:
      alignment: sum of neighbour velocities weighted by bias^dot
      cohesion:  seek toward average neighbour position
      separation: sum of (self−other)/sqrDist for each neighbour
    Each is then Reynolds-steered: setMag(maxSpeed) − vel, limited by maxForce.
    """
    candidate_count = candidate_mask.astype(jnp.float32).sum(axis=1, keepdims=True)
    sample_prob = jnp.where(
        (accuracy <= 0.0) | (candidate_count <= accuracy),
        1.0,
        accuracy / candidate_count.clip(min=1),
    )
    sampled = (jax.random.uniform(key, sq_dist.shape) < sample_prob).astype(jnp.float32)
    mask = (
        candidate_mask
        & (sq_dist < sq_vision)
        & (sq_dist > 1e-6)
    ).astype(jnp.float32)
    mask = mask * sampled
    count = mask.sum(axis=1, keepdims=True)
    has_neighbours = (count > 0.5).astype(jnp.float32)

    # ── alignment ────────────────────────────────────────────────────
    # bias = alignment_bias ^ dot(other.vel, this.vel)
    # dot product: (N, N) — row i, col j = dot(vel_j, vel_i)
    vel_dot = jnp.sum(
        velocities[None, :, :] * velocities[:, None, :], axis=-1
    )
    bias_weights = alignment_bias ** vel_dot
    aln = (velocities[None, :, :] * (mask * bias_weights)[:, :, None]).sum(axis=1)
    aln = _set_mag(aln, max_speed) - velocities
    aln = _limit(aln, max_force)

    # ── cohesion (seek toward average neighbour position) ────────────
    # In the JS code: csn accumulates other.position, then
    #   csn.div(count).sub(this).setMag(maxSpeed).sub(vel).max(maxForce)
    # displacement[i,j] = pos_j - pos_i, so avg displacement = avg neighbour pos - self pos
    avg_disp = (displacement * mask[:, :, None]).sum(axis=1) / count.clip(min=1)
    csn = _set_mag(avg_disp, max_speed) - velocities
    csn = _limit(csn, max_force)

    # ── separation ───────────────────────────────────────────────────
    # JS: sep += (this - other) / sqrDist  for each neighbour
    # (this - other) = -displacement
    inv_sq = 1.0 / (sq_dist + 1e-5)
    sep = (-displacement * inv_sq[:, :, None] * mask[:, :, None]).sum(axis=1)
    sep = _set_mag(sep, max_speed) - velocities
    sep = _limit(sep, max_force)

    # Zero out steering when there are no neighbours
    aln = aln * has_neighbours
    csn = csn * has_neighbours
    sep = sep * has_neighbours

    acc = (aln * alignment_weight
           + csn * cohesion_weight
           + sep * separation_weight)
    return acc


# ── public API ───────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("boundary",))
def compute_boid_steering(
    positions, velocities, key,
    vision, accuracy, alignment_weight, alignment_bias,
    cohesion_weight, separation_weight,
    max_speed, max_force,
    canvas_w, canvas_h, boundary,
):
    """Return (N, 2) acceleration for every agent based on flocking rules."""
    # The JS reference wraps positions at the canvas edge, but neighbour
    # distances are ordinary screen-space distances, not toroidal distances.
    disp = _pairwise_displacement(positions)

    sq_dist = _sq_distance(disp)
    sq_vision = vision * vision
    cell = jnp.maximum(vision, 1.0)
    rows = jnp.floor(positions[:, 1] / cell).astype(jnp.int32)
    cols = jnp.floor(positions[:, 0] / cell).astype(jnp.int32)
    candidate_mask = (
        (jnp.abs(rows[:, None] - rows[None, :]) <= 1)
        & (jnp.abs(cols[:, None] - cols[None, :]) <= 1)
        & (vision > 0.0)
    )

    return _flock(
        disp, sq_dist, velocities, candidate_mask, sq_vision, accuracy, key,
        alignment_weight, alignment_bias,
        cohesion_weight, separation_weight,
        max_speed, max_force,
    )


@partial(jax.jit, static_argnames=("boundary",))
def update_boids(
    positions, velocities,
    acceleration, controlled_velocity,
    key,
    dt, min_speed, max_speed,
    drag, noise,
    canvas_w, canvas_h, boundary_padding, boundary,
    controlled_agent,
):
    """Advance all boids by one timestep.  Returns (new_pos, new_vel, new_headings, new_key).

    Mirrors the JS ``Boid.update()`` exactly:
      1. vel += acc * dt
      2. vel *= (1 - drag)
      3. vel.rotate(random noise)
      4. enforce min_speed / max_speed
      5. pos += vel * dt
      6. wrap or bounce
    """
    n = positions.shape[0]
    key, k_noise, k_zero = jax.random.split(key, 3)

    # 1. apply acceleration
    new_vel = velocities + acceleration * dt

    # 2. drag
    new_vel = new_vel * (1.0 - drag)

    # 3. random heading perturbation
    noise_range = (jnp.pi / 80.0) * noise
    angles = jax.random.uniform(k_noise, (n,), minval=-noise_range, maxval=noise_range)
    cos_a = jnp.cos(angles)
    sin_a = jnp.sin(angles)
    rx = new_vel[:, 0] * cos_a - new_vel[:, 1] * sin_a
    ry = new_vel[:, 0] * sin_a + new_vel[:, 1] * cos_a
    new_vel = jnp.stack([rx, ry], axis=-1)

    # 4. enforce min/max speed
    new_vel = _enforce_min_speed(new_vel, min_speed, k_zero)
    new_vel = _limit(new_vel, max_speed)

    new_vel = jax.lax.cond(
        controlled_agent,
        lambda v: v.at[0].set(controlled_velocity),
        lambda v: v,
        new_vel,
    )

    # 5. integrate position
    new_pos = positions + new_vel * dt

    # 6. boundary handling
    if boundary == "wrap":
        new_pos = jnp.mod(new_pos, jnp.array([canvas_w, canvas_h]))
    else:
        min_x = boundary_padding
        min_y = boundary_padding
        max_x = jnp.maximum(min_x, canvas_w - boundary_padding)
        max_y = jnp.maximum(min_y, canvas_h - boundary_padding)
        hit_x = (new_pos[:, 0] < min_x) | (new_pos[:, 0] > max_x)
        hit_y = (new_pos[:, 1] < min_y) | (new_pos[:, 1] > max_y)
        new_pos = jnp.stack([
            jnp.clip(new_pos[:, 0], min_x, max_x),
            jnp.clip(new_pos[:, 1], min_y, max_y),
        ], axis=-1)
        vel_x = jnp.where(hit_x, -new_vel[:, 0], new_vel[:, 0])
        vel_y = jnp.where(hit_y, -new_vel[:, 1], new_vel[:, 1])
        new_vel = jnp.stack([vel_x, vel_y], axis=-1)

    new_headings = jnp.arctan2(new_vel[:, 1], new_vel[:, 0])

    return new_pos, new_vel, new_headings, key

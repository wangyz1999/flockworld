"""Boid flocking physics implemented in pure JAX.

Faithfully reproduces the algorithm from the JS/PixiJS reference:
  - Single ``vision`` radius for neighbour detection
  - Alignment with velocity-dot-product ``bias``
  - Reynolds-style steering (desired velocity − current velocity, clamped
    by ``max_force``) for all three rules
  - Velocity drag, random heading noise, min/max speed clamping
  - No artificial turn-rate limiter — steering force + drag handle it

The controllable agent (index 0) is overridden by the caller.
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
    mag = jnp.sqrt(jnp.sum(vec ** 2, axis=-1, keepdims=True) + 1e-8)
    return vec / mag * target


def _enforce_min_speed(vel, min_speed):
    """Ensure every velocity has at least ``min_speed`` magnitude."""
    sq = jnp.sum(vel ** 2, axis=-1, keepdims=True)
    mag = jnp.sqrt(sq + 1e-8)
    scale = jnp.where(sq < min_speed * min_speed, min_speed / mag, 1.0)
    return vel * scale


# ── flocking rules (exact JS reference algorithm) ───────────────────────

def _flock(displacement, sq_dist, velocities, sq_vision,
           alignment_weight, alignment_bias,
           cohesion_weight, separation_weight,
           max_speed, max_force):
    """Compute per-boid acceleration from all three flocking rules.

    Mirrors the JS ``Boid.flock()`` method exactly:
      alignment: sum of neighbour velocities weighted by bias^dot
      cohesion:  seek toward average neighbour position
      separation: sum of (self−other)/sqrDist for each neighbour
    Each is then Reynolds-steered: setMag(maxSpeed) − vel, limited by maxForce.
    """
    n = velocities.shape[0]
    mask = ((sq_dist < sq_vision) & (sq_dist > 1e-6)).astype(jnp.float32)
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
    positions, velocities,
    vision, alignment_weight, alignment_bias,
    cohesion_weight, separation_weight,
    max_speed, max_force,
    canvas_w, canvas_h, boundary,
):
    """Return (N, 2) acceleration for every agent based on flocking rules."""
    if boundary == "wrap":
        disp = _pairwise_displacement_wrap(positions, canvas_w, canvas_h)
    else:
        disp = _pairwise_displacement_reflect(positions, canvas_w, canvas_h)

    sq_dist = _sq_distance(disp)
    sq_vision = vision * vision

    return _flock(
        disp, sq_dist, velocities, sq_vision,
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
    canvas_w, canvas_h, boundary,
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
    key, k_noise = jax.random.split(key)

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
    new_vel = _enforce_min_speed(new_vel, min_speed)
    new_vel = _limit(new_vel, max_speed)

    # Override controlled agent (index 0)
    new_vel = new_vel.at[0].set(controlled_velocity)

    # 5. integrate position
    new_pos = positions + new_vel * dt

    # 6. boundary handling
    if boundary == "wrap":
        new_pos = jnp.mod(new_pos, jnp.array([canvas_w, canvas_h]))
    else:
        hit_x = (new_pos[:, 0] < 0) | (new_pos[:, 0] > canvas_w)
        hit_y = (new_pos[:, 1] < 0) | (new_pos[:, 1] > canvas_h)
        new_pos = jnp.stack([
            jnp.clip(new_pos[:, 0], 0.0, canvas_w),
            jnp.clip(new_pos[:, 1], 0.0, canvas_h),
        ], axis=-1)
        vel_x = jnp.where(hit_x, -new_vel[:, 0], new_vel[:, 0])
        vel_y = jnp.where(hit_y, -new_vel[:, 1], new_vel[:, 1])
        new_vel = jnp.stack([vel_x, vel_y], axis=-1)

    new_headings = jnp.arctan2(new_vel[:, 0], new_vel[:, 1])

    return new_pos, new_vel, new_headings, key

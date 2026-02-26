"""JAX-compatible state types for the boid simulation."""

from typing import NamedTuple

import jax.numpy as jnp


class BoidState(NamedTuple):
    """State of all boids in the simulation.

    Agent at index 0 is always the controllable agent.
    """

    positions: jnp.ndarray   # (N, 2) — x, y in pixel coordinates
    velocities: jnp.ndarray  # (N, 2) — vx, vy
    headings: jnp.ndarray    # (N,)   — angle in radians


class EnvState(NamedTuple):
    """Full environment state passed through step/reset."""

    boids: BoidState
    step_count: int
    key: jnp.ndarray  # PRNGKey

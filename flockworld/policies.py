"""Heuristic movement policies for the controlled agent (index 0).

Each policy is a callable with the signature::

    action, policy_state = policy(state, params, policy_state, key)

where ``action`` is a scalar heading angle (radians) and ``policy_state``
is an opaque dict carried across ticks (initialised by ``init_policy``).

Available policies
------------------
- ``straight``   — maintain current heading
- ``random``     — uniform random heading each tick
- ``perlin``     — smooth organic wandering via layered Perlin-style noise
- ``circle``     — fly in a fixed-radius circle
- ``lissajous``  — trace a figure-eight (Lissajous curve) across the canvas
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp

from flockworld.core.types import EnvState


class PolicyFn(Protocol):
    def __call__(
        self,
        state: EnvState,
        params,
        policy_state: dict,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]: ...


# ── straight ─────────────────────────────────────────────────────────────

def _straight(state, params, policy_state, key):
    """Keep flying in the current direction."""
    action = state.boids.headings[0]
    return action, policy_state


# ── random ───────────────────────────────────────────────────────────────

def _random(state, params, policy_state, key):
    """Pick a uniformly random heading each tick."""
    key, k = jax.random.split(key)
    action = jax.random.uniform(k, (), minval=-jnp.pi, maxval=jnp.pi)
    policy_state = {**policy_state, "key": key}
    return action, policy_state


# ── perlin (smooth organic wandering) ────────────────────────────────────

def _perlin(state, params, policy_state, key):
    """Smooth wandering with large-scale sweeping motion.

    A slowly accumulating base angle is perturbed by layered sinusoids at
    incommensurate frequencies.  The result is an absolute heading that
    drifts continuously, producing wide arcs across the canvas rather than
    spinning in place.
    """
    t = policy_state.get("t", 0.0)

    drift = t * 0.008
    wobble = (
        1.20 * jnp.sin(t * 0.011 + 0.0)
        + 0.70 * jnp.sin(t * 0.029 + 2.1)
        + 0.35 * jnp.sin(t * 0.071 + 5.3)
    )
    action = drift + wobble

    policy_state = {**policy_state, "t": t + 1.0}
    return action, policy_state


# ── circle ───────────────────────────────────────────────────────────────

def _circle(state, params, policy_state, key):
    """Fly in a circle around the canvas centre.

    The heading is set tangent to a circle of radius ≈ 1/3 canvas size.
    """
    t = policy_state.get("t", 0.0)
    speed = float(params.max_speed)
    cx, cy = float(params.canvas_w) / 2, float(params.canvas_h) / 2
    radius = min(cx, cy) * 0.35

    omega = speed / radius
    theta = t * omega

    # tangent direction (perpendicular to radius vector)
    action = theta + jnp.pi / 2

    policy_state = {**policy_state, "t": t + 1.0}
    return action, policy_state


# ── lissajous (figure-eight) ─────────────────────────────────────────────

def _lissajous(state, params, policy_state, key):
    """Trace a figure-eight (Lissajous curve) across the canvas.

    The agent follows a parametric path ``(A sin(t), B sin(2t))`` centred on
    the canvas.  The action heading is set to the tangent of that curve.
    """
    t = policy_state.get("t", 0.0)
    speed = float(params.max_speed)
    cx, cy = float(params.canvas_w) / 2, float(params.canvas_h) / 2
    ax = min(cx, cy) * 0.35
    ay = min(cx, cy) * 0.35

    freq = speed / (ax * 2)
    phase = t * freq

    # derivative of (ax*sin(phase), ay*sin(2*phase))
    dx = ax * jnp.cos(phase) * freq
    dy = ay * 2.0 * jnp.cos(2.0 * phase) * freq

    action = jnp.arctan2(dy, dx)

    policy_state = {**policy_state, "t": t + 1.0}
    return action, policy_state


# ── registry ─────────────────────────────────────────────────────────────

POLICIES: dict[str, PolicyFn] = {
    "straight": _straight,
    "random": _random,
    "perlin": _perlin,
    "circle": _circle,
    "lissajous": _lissajous,
}


def get_policy(name: str) -> PolicyFn:
    """Look up a policy by name.  Raises ``ValueError`` on unknown name."""
    if name not in POLICIES:
        available = ", ".join(sorted(POLICIES))
        raise ValueError(
            f"Unknown agent policy {name!r}. Available: {available}"
        )
    return POLICIES[name]


def init_policy(name: str, key: jnp.ndarray) -> dict:
    """Return the initial ``policy_state`` dict for the given policy."""
    return {"key": key, "t": 0.0}

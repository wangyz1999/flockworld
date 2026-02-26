"""Gymnasium-compatible wrapper around the pure-JAX flock environment."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from flockworld.env.flock_env import (
    EnvConfig,
    env_config_from_omega,
    render,
    reset as jax_reset,
    step as jax_step,
)
from flockworld.rendering.renderer import build_uv_grid


class FlockEnv(gym.Env):
    """Gymnasium ``Env`` that wraps the pure-JAX boid simulation.

    Observations are rendered RGB frames (uint8).
    The action is a single float — the heading angle for the controlled
    agent (index 0).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg=None, env_config: EnvConfig | None = None, seed: int = 42):
        super().__init__()

        if env_config is not None:
            self.ec = env_config
        elif cfg is not None:
            self.ec = env_config_from_omega(cfg)
        else:
            self.ec = EnvConfig()

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.ec.canvas_h, self.ec.canvas_w, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32,
        )

        self._uv_grid = build_uv_grid(self.ec.canvas_w, self.ec.canvas_h)
        self._render_jit = jax.jit(lambda boids: render_frame_jit(
            boids, self._uv_grid, self.ec,
        ))

        self._key = jax.random.PRNGKey(seed)
        self._state = None

    # ── gym interface ────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, sub = jax.random.split(self._key)
        self._state = jax_reset(sub, self.ec)
        obs = self._render_obs()
        return obs, {}

    def step(self, action):
        action_val = float(action[0]) if hasattr(action, "__len__") else float(action)
        self._state, reward, done, info = jax_step(self._state, action_val, self.ec)
        obs = self._render_obs()
        return obs, float(reward), bool(done), False, info

    def render(self):
        return self._render_obs()

    # ── internals ────────────────────────────────────────────────────

    def _render_obs(self) -> np.ndarray:
        frame = render(self._state, self.ec, self._uv_grid)
        frame_np = np.asarray(frame)
        return np.clip(frame_np * 255, 0, 255).astype(np.uint8)

    @property
    def state(self):
        return self._state


def render_frame_jit(boids, uv_grid, ec):
    """Thin wrapper kept outside the class for JIT compatibility."""
    return render(boids, ec, uv_grid)

"""Gymnasium-compatible wrapper around the pure-JAX flock environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from flockworld.runtime import configure_jax_platform


class FlockEnv(gym.Env):
    """Gymnasium ``Env`` that wraps the pure-JAX boid simulation.

    Observations are rendered RGB frames (uint8).
    The action is a single float. It is used as the heading angle for index 0
    only when ``EnvConfig.controlled_agent`` is enabled.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg=None, env_config: EnvConfig | None = None, seed: int = 42):
        super().__init__()
        if env_config is None and cfg is None:
            raise ValueError(
                "FlockEnv requires either cfg (OmegaConf) or env_config (EnvConfig); "
                "all fields are defined in config/default.yaml.",
            )

        device = env_config.device if env_config is not None else cfg.device

        configure_jax_platform(device)

        import jax
        import jax.numpy as jnp

        from flockworld.env.flock_env import (
            EnvParams,
            env_config_from_omega,
            render,
            reset as jax_reset,
            step as jax_step,
        )
        from flockworld.rendering.renderer import build_uv_grid

        self._jax = jax
        self._jnp = jnp
        self._jax_reset = jax_reset
        self._jax_step = jax_step
        self._render_frame = render

        if env_config is not None:
            self.ec = env_config
        else:
            self.ec = env_config_from_omega(cfg)

        self.params = EnvParams(self.ec)

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.ec.canvas_h, self.ec.canvas_w, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32,
        )

        self._uv_grid = build_uv_grid(self.ec.canvas_w, self.ec.canvas_h)
        self._key = self._jax.random.PRNGKey(seed)
        self._state = None

    # ── gym interface ────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = self._jax.random.PRNGKey(seed)
        self._key, sub = self._jax.random.split(self._key)
        self._state = self._jax_reset(sub, self.ec)
        obs = self._render_obs()
        return obs, {}

    def step(self, action):
        action_val = (
            self._jnp.float32(action[0])
            if hasattr(action, "__len__")
            else self._jnp.float32(action)
        )
        self._state, reward, done, info = self._jax_step(
            self._state, action_val, self.params,
        )
        obs = self._render_obs()
        return obs, float(reward), bool(done), False, info

    def render(self):
        return self._render_obs()

    # ── internals ────────────────────────────────────────────────────

    def _render_obs(self) -> np.ndarray:
        frame = self._render_frame(self._state, self.params, self._uv_grid)
        frame_np = np.asarray(frame)
        return np.clip(frame_np * 255, 0, 255).astype(np.uint8)

    @property
    def state(self):
        return self._state

"""Runtime helpers for selecting the JAX execution platform."""

from __future__ import annotations

import os
import sys


def normalize_device(device: str | None) -> str:
    """Normalize user-facing device names to JAX platform names."""
    value = (device or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "default": "auto",
        "cpu": "cpu",
        "gpu": "gpu",
        "cuda": "gpu",
    }
    if value not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported device {device!r}. Expected one of: {valid}")
    return aliases[value]


def configure_jax_platform(device: str | None) -> str:
    """Configure JAX's default platform before the backend is initialized.

    ``auto`` leaves JAX's normal device selection unchanged. ``cpu`` and
    ``gpu`` map to JAX's ``jax_platform_name`` setting.
    """
    platform = normalize_device(device)
    if platform == "auto":
        return platform

    jax_platforms = "cuda" if platform == "gpu" else platform
    os.environ["JAX_PLATFORM_NAME"] = platform
    os.environ["JAX_PLATFORMS"] = jax_platforms
    if "jax" in sys.modules:
        from jax import config as jax_config

        jax_config.update("jax_platform_name", platform)
        jax_config.update("jax_platforms", jax_platforms)
    return platform

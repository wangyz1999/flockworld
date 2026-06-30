"""Ground-truth boid state + projection into agent partial-views.

Geometry (verified from `flockworld/video/recorder.py:_crop_partial` and the
renderer): a partial view is a ``size``x``size`` axis-aligned crop **centered on
the agent**, with **1:1 world-to-pixel scale and no rotation**. So a boid at world
``(wx, wy)`` appears in agent k's view at pixel

    u = wx - round(ax) + half ,   v = wy - round(ay) + half      (half = size // 2)

and is visible iff ``0 <= u < size`` and ``0 <= v < size``. ``u`` is the column
(x), ``v`` the row (y) — index a frame as ``frame[v, u]``.

Ground truth comes from ``state_action/<episode>.parquet`` (wide columns
``a{k}_pos_x/y``, ``a{k}_heading`` for 1-indexed boid k). Parquet row i lines up
with video / decoded-pixel frame i (the 60-frame warmup precedes recording).
Camera agent index ``j`` (0-based, ``video_a{j+1}``) is boid column ``a{j+1}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

PARTIAL_SIZE = 128  # px; matches config canvas.partial


def load_gt_positions(parquet_path: str | Path, num_agents: int) -> np.ndarray:
    """Read every boid's world (x, y) per frame -> ``(T, num_agents, 2)`` float32."""
    df = pl.read_parquet(parquet_path)
    pos = np.empty((df.height, num_agents, 2), dtype=np.float32)
    for k in range(num_agents):
        pos[:, k, 0] = df[f"a{k + 1}_pos_x"].to_numpy()
        pos[:, k, 1] = df[f"a{k + 1}_pos_y"].to_numpy()
    return pos


def load_gt_headings(parquet_path: str | Path, num_agents: int) -> np.ndarray:
    """Read every boid's heading (radians) per frame -> ``(T, num_agents)`` float32."""
    df = pl.read_parquet(parquet_path)
    h = np.empty((df.height, num_agents), dtype=np.float32)
    for k in range(num_agents):
        h[:, k] = df[f"a{k + 1}_heading"].to_numpy()
    return h


def project(boid_xy: np.ndarray, agent_xy: np.ndarray, size: int = PARTIAL_SIZE) -> np.ndarray:
    """World -> pixel in an agent's crop. ``boid_xy`` (..., 2), ``agent_xy`` (2,) or (...,2).

    Returns ``(..., 2)`` pixel coords ``(u=col, v=row)``. Uses the integer-rounded
    agent center, exactly as the renderer/recorder crop does.
    """
    half = size // 2
    ax = np.round(agent_xy[..., 0])
    ay = np.round(agent_xy[..., 1])
    u = boid_xy[..., 0] - ax + half
    v = boid_xy[..., 1] - ay + half
    return np.stack([u, v], axis=-1)


def visible(pixel: np.ndarray, size: int = PARTIAL_SIZE) -> np.ndarray:
    """Boolean mask: pixel (..., 2) lies inside the [0, size) crop."""
    u, v = pixel[..., 0], pixel[..., 1]
    return (u >= 0) & (u < size) & (v >= 0) & (v < size)


def visible_boids(positions_t: np.ndarray, agent_idx: int, size: int = PARTIAL_SIZE):
    """Boids visible in ``agent_idx``'s view at one frame.

    Args:
        positions_t: ``(num_agents, 2)`` world positions at a single frame.
    Returns ``(boid_indices, pixels)`` where ``pixels`` are the in-crop ``(u, v)``
    of those boids (includes the agent itself, which sits near (half, half)).
    """
    px = project(positions_t, positions_t[agent_idx], size)  # (num_agents, 2)
    vis = visible(px, size)
    idx = np.nonzero(vis)[0]
    return idx, px[idx]


def covisible_boids(positions_t: np.ndarray, agent_a: int, agent_b: int, size: int = PARTIAL_SIZE):
    """Boids visible in BOTH agents' views at one frame (the Tier B test set).

    Returns ``(boid_indices, pixels_in_a, pixels_in_b)``.
    """
    pa = project(positions_t, positions_t[agent_a], size)
    pb = project(positions_t, positions_t[agent_b], size)
    both = visible(pa, size) & visible(pb, size)
    idx = np.nonzero(both)[0]
    return idx, pa[idx], pb[idx]

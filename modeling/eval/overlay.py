"""Overlay GT boid markers on decoded partial-view frames, and tile P views.

Used by the qualitative multi-view eval: decode each agent's (predicted or GT)
latents to pixels, draw a marker at every GT-projected boid position, and tile
the P views into one grid video. The focal agent gets a red ring, others green,
so you can eyeball whether the model rendered darts where GT says they should be
(and whether co-located agents agree).
"""

from __future__ import annotations

import cv2
import numpy as np

from modeling.eval import gt_project as gp


def to_uint8(clip) -> np.ndarray:
    """(T, 3, H, W) in [-1, 1] (torch or np) -> (T, H, W, 3) uint8 RGB."""
    x = np.asarray(clip.detach().cpu() if hasattr(clip, "detach") else clip)
    x = ((np.clip(x, -1.0, 1.0) + 1.0) / 2.0 * 255.0).round().astype(np.uint8)
    return np.transpose(x, (0, 2, 3, 1))


def draw_gt_marks(frames_thwc, positions, agent_world_idx, size=gp.PARTIAL_SIZE, radius=3):
    """Draw a ring at every GT-visible boid's projected pixel, per frame.

    Args:
        frames_thwc: ``(T, H, W, 3)`` uint8 RGB (the decoded view, possibly upscaled).
        positions:   ``(T_gt, num_agents, 2)`` GT world positions (parquet).
        agent_world_idx: 0-based boid index this view is centered on.
    Focal agent -> red ring; other boids -> green. Returns a marked copy.
    """
    out = frames_thwc.copy()
    H = out.shape[1]
    scale = H / size  # support upscaled frames (e.g. 128 -> 256)
    T = min(len(out), len(positions))
    for t in range(T):
        idx, px = gp.visible_boids(positions[t], agent_world_idx, size)
        for bi, (u, v) in zip(idx, px):
            c = (255, 0, 0) if bi == agent_world_idx else (0, 255, 0)  # RGB
            cv2.circle(out[t], (int(round(u * scale)), int(round(v * scale))),
                       radius, c, 1, lineType=cv2.LINE_AA)
    return out


def tile(views, pad=2, bg=40) -> np.ndarray:
    """List of ``(T, H, W, 3)`` uint8 (same shape) -> one ``(T, gridH, gridW, 3)`` grid."""
    n = len(views)
    T, H, W, _ = views[0].shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    gh, gw = rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad
    grid = np.full((T, gh, gw, 3), bg, np.uint8)
    for i, v in enumerate(views):
        r, c = divmod(i, cols)
        y0, x0 = pad + r * (H + pad), pad + c * (W + pad)
        grid[:, y0:y0 + H, x0:x0 + W] = v[:T]
    return grid


def upscale(frames_thwc, k: int) -> np.ndarray:
    """Nearest-neighbour upscale (T, H, W, 3) by integer factor k (for visibility)."""
    return np.repeat(np.repeat(frames_thwc, k, axis=1), k, axis=2)

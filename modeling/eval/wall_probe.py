"""Wall/corner-anchored consistency probe (Setup 1) — shared helpers.

The probe uses the white world border as a GT-free fiducial: an agent near a
corner can read its own ABSOLUTE position from where the two walls sit in its
egocentric frame, so we can check whether two corner-co-located agents render a
mutually-consistent world WITHOUT anchoring on gt_pos. See the memory note
`wall-anchored-consistency-probe`.

Step 1 (this file, GT-only): find the candidate events — frames where two camera
agents share the SAME corner and their 128px crops overlap. Later steps roll the
model out to these events, detect the corner in the generated frames to recover
each agent's absolute position, and score cross-view consistency.
"""

from __future__ import annotations

import numpy as np

WORLD = 720      # canvas is 720x720
HALF = 64        # crop half-size: a wall is in view when the agent is within HALF of that boundary
CROP = 128       # two crops overlap when agents are within CROP in both axes
CORNERS = ("TL", "TR", "BL", "BR")   # matches corner_flags order below


def detect_corner(frame_rgb, v_thresh: float = 0.25, wall_len: int = 40,
                  world: int = WORLD, half: int = HALF):
    """Detect a corner (two perpendicular border walls) in an egocentric frame and
    recover the agent's ABSOLUTE position from it -- GT-free.

    The white border shows as a long bright vertical line (a wall at column wx) and
    a long bright horizontal line (row wy); a corner is when both are present. The
    agent is at frame center (64,64), the corner is the known world point, so:
        agent = corner_world - (corner_px - 64).
    Left wall -> wx<64 / world x=0; right -> wx>64 / world x=WORLD (same for T/B).

    Returns {corner: 'TL'|'TR'|'BL'|'BR', agent_xy: (x,y), corner_px: (wx,wy)} or None.
    """
    f = np.asarray(frame_rgb, np.float32)
    f = f / 255.0 if f.max() > 1.5 else f
    bright = f.max(-1) > v_thresh                 # (H, W)
    col = bright.sum(0)                           # bright pixels per column (finds a vertical wall)
    row = bright.sum(1)                           # per row (finds a horizontal wall)
    if col.max() < wall_len or row.max() < wall_len:
        return None                               # need BOTH a vertical and horizontal wall -> a corner
    wx, wy = int(col.argmax()), int(row.argmax())
    cx_world = 0 if wx < half else world
    cy_world = 0 if wy < half else world
    corner = ("T" if wy < half else "B") + ("L" if wx < half else "R")
    return {"corner": corner, "agent_xy": (cx_world - (wx - half), cy_world - (wy - half)),
            "corner_px": (wx, wy)}


def latent_index(pixel: int, temporal: int = 4) -> int:
    """WanVAE causal time mapping: pixel-frame index -> covering latent-frame index.

    The VAE is 1 + 4k (latent 0 -> pixel 0; latent L -> pixels 4L-3..4L). Events are
    found in pixel frames (parquet rows); the rollout runs in latent frames.
    """
    return (pixel + temporal - 1) // temporal


def wall_flags(xy: np.ndarray, world: int = WORLD, half: int = HALF) -> np.ndarray:
    """(...,2) world positions -> (...,4) bools [near_left, near_right, near_top, near_bottom]."""
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([x < half, x > world - half, y < half, y > world - half], axis=-1)


def corner_flags(wf: np.ndarray) -> np.ndarray:
    """wall flags (...,4)[L,R,T,B] -> (...,4) corner bools [TL, TR, BL, BR]."""
    L, R, T, B = wf[..., 0], wf[..., 1], wf[..., 2], wf[..., 3]
    return np.stack([L & T, R & T, L & B, R & B], axis=-1)


def _runs(active: np.ndarray, min_len: int):
    """Yield (t0, t1_inclusive) maximal runs of True with length >= min_len."""
    if not active.any():
        return
    idx = np.flatnonzero(active)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [len(idx) - 1]])
    for s, e in zip(starts, ends):
        t0, t1 = int(idx[s]), int(idx[e])
        if t1 - t0 + 1 >= min_len:
            yield t0, t1


def find_seed_frame(cam_a: np.ndarray, cam_b: np.ndarray, t0: int,
                    min_gen: int = 12, max_lookback: int = 60, crop: int = CROP):
    """Latest frame in [t0-max_lookback, t0-min_gen] where a and b's crops do NOT
    overlap -> an independent-context seed, so the co-location must EMERGE during
    generation (not be handed to the model). Returns the pixel-frame index or None.

    cam_a, cam_b: (T, 2) position series of the two agents.
    """
    hi = t0 - min_gen
    lo = max(0, t0 - max_lookback)
    for t in range(hi, lo - 1, -1):                 # latest (shortest rollout) first
        d = np.abs(cam_a[t] - cam_b[t])
        if d[0] >= crop or d[1] >= crop:            # crops share no pixels -> independent
            return int(t)
    return None


def find_corner_events(gt_pos: np.ndarray, cam_idx, world: int = WORLD, half: int = HALF,
                       crop: int = CROP, min_len: int = 1):
    """Find events where two camera agents share a corner AND their crops overlap.

    Args:
        gt_pos:  (T, N, 2) GT world positions.
        cam_idx: 0-based boid indices of the camera agents.
    Returns a list of dicts: {a, b (0-based cam-slot), corner (str), t0, t1, len}
    where an event is a maximal run of consecutive qualifying frames.
    """
    cam = gt_pos[:, list(cam_idx), :]                 # (T, P, 2)
    P = cam.shape[1]
    cf = corner_flags(wall_flags(cam, world, half))   # (T, P, 4)
    out = []
    for a in range(P):
        for b in range(a + 1, P):
            d = np.abs(cam[:, a] - cam[:, b])
            overlap = (d[:, 0] < crop) & (d[:, 1] < crop)
            for c in range(4):
                active = cf[:, a, c] & cf[:, b, c] & overlap
                for t0, t1 in _runs(active, min_len):
                    out.append({"a": a, "b": b, "corner": CORNERS[c],
                                "t0": t0, "t1": t1, "len": t1 - t0 + 1})
    return out

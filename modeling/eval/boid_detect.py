"""Detect rendered boids (darts) in a decoded partial-view frame.

Threshold on the **HSV Value channel (V = max RGB), not grayscale luminance** —
a full-saturation colored dart can have low luminance (pure blue ~0.11) and be
missed by a grayscale threshold, but V~1 for any bright dart, so V finds white
*and* colored darts uniformly against the black background. Then connected
components -> centroids. Also reads each blob's hue (identifies camera agents in
the color setups; NaN for achromatic white darts). One detector for every setup.

Gradient setups: subtract the expected gradient from the frame first (see
``gt_project.expected_gradient_crop``) so the background is ~black again, then
detect as usual.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_float01(frame) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    return f / 255.0 if f.max() > 1.5 else f


def _blob_hue(f: np.ndarray, lbl: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Mean-RGB hue per blob in [0,1); NaN if the blob is achromatic (white)."""
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    hues = np.full(len(ids), np.nan, dtype=np.float32)
    for i, lid in enumerate(ids):
        m = lbl == lid
        r, g, b = float(R[m].mean()), float(G[m].mean()), float(B[m].mean())
        mx, mn = max(r, g, b), min(r, g, b)
        if mx - mn < 0.05:                       # achromatic -> no meaningful hue
            continue
        if mx == r:
            h = ((g - b) / (mx - mn)) % 6.0
        elif mx == g:
            h = (b - r) / (mx - mn) + 2.0
        else:
            h = (r - g) / (mx - mn) + 4.0
        hues[i] = (h / 6.0) % 1.0
    return hues


def detect_boids(frame_rgb, v_thresh: float = 0.25, min_size: int = 2,
                 max_elong: float = 5.0, max_extent: int = 40,
                 border_px: int = 5, border_min_size: int = 25) -> dict:
    """Find darts in one RGB frame.

    Boids render as compact blobs (bbox elongation ~1.4, extent ~13px, size ~64);
    the white world border renders as a thin straight LINE (elongation up to ~128),
    an L-shaped corner (huge extent), or -- where it just clips the crop -- a small
    compact fragment hugging the frame edge (size ~11). After thresholding +
    connected components we reject blobs that are:
      * too elongated (``> max_elong``)      -> border lines,
      * too large     (``> max_extent``)     -> corners,
      * small AND on the frame border        -> border-clip fragments,
    all outside the real-dart range measured on GT frames (elong p99 2.6; extent
    p99 25; darts median size 64). Merged darts stay compact and interior, so they
    survive; a half-clipped real boid at the edge is size >~30 so it also survives.

    Args:
        frame_rgb: ``(H, W, 3)`` RGB, float in [0,1] or uint8.
    Returns dict:
        ``centroids`` ``(N, 2)`` as ``(u=col, v=row)`` (V-weighted),
        ``sizes`` ``(N,)`` pixel counts,
        ``hue`` ``(N,)`` in [0,1) (NaN for white darts).
    """
    f = _to_float01(frame_rgb)
    V = f.max(axis=-1)                            # HSV value
    lbl, n = ndimage.label(V > v_thresh)
    if n == 0:
        z = np.zeros((0,), np.float32)
        return {"centroids": np.zeros((0, 2), np.float32), "sizes": z, "hue": z}
    ids = np.arange(1, n + 1)
    sizes = np.asarray(ndimage.sum(np.ones_like(lbl, np.float32), lbl, ids))
    objs = ndimage.find_objects(lbl)              # per-label bbox slices (label i -> objs[i-1])
    hh = np.array([s[0].stop - s[0].start for s in objs], np.float32)
    ww = np.array([s[1].stop - s[1].start for s in objs], np.float32)
    extent = np.maximum(hh, ww)                   # longest bbox side
    elong = extent / np.maximum(np.minimum(hh, ww), 1.0)
    coms = np.asarray(ndimage.center_of_mass(V, lbl, ids))   # (row, col), V-weighted
    H, W = V.shape
    on_border = ((coms[:, 1] < border_px) | (coms[:, 1] > W - border_px) |
                 (coms[:, 0] < border_px) | (coms[:, 0] > H - border_px))
    keep = ((sizes >= min_size) & (elong <= max_elong) & (extent <= max_extent)   # reject border lines/corners
            & ~(on_border & (sizes < border_min_size)))                           # + border-clip fragments
    ids, sizes, coms = ids[keep], sizes[keep], coms[keep]
    if len(ids) == 0:
        z = np.zeros((0,), np.float32)
        return {"centroids": np.zeros((0, 2), np.float32), "sizes": z, "hue": z}
    centroids = np.stack([coms[:, 1], coms[:, 0]], axis=-1).astype(np.float32)  # (u, v)
    return {"centroids": centroids, "sizes": sizes.astype(np.float32), "hue": _blob_hue(f, lbl, ids)}

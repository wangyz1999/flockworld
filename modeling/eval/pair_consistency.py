"""GT-free cross-view consistency for the colored-agent multi-view setup.

Replaces the old Tier B (removed): NOTHING here touches ``gt_pos`` -- not the
scoring coordinate frame, not event selection. The two camera agents localize
**each other**. In the ``agent_id`` setup, camera agent ``k`` renders as a dart
of hue ``k / n_cam`` (flock_env), so in agent a's egocentric view agent b shows
up as a b-hued dart whose pixel gives b's world offset from a -- with zero ground
truth. Covisibility is whatever the model renders: "A sees B" == A drew a B-hued
dart. So this measures the model's INTERNAL coherence and is deliberately blind
to whether it rendered the sightings reality demanded (that is fidelity, Tier A).

Because it is self-triggered, an agreement RATE is meaningless on its own -- a
near-empty world trivially agrees. Every rate is therefore returned next to a
VOLUME (``n_sightings``, ``sightings_per_frame``, ``n_reciprocal``); read the
rates against those. A sparse model that renders nothing must not win.

Geometry (gt_project): an agent-centered ``size`` px crop, 1:1 scale, no
rotation, so a dart at pixel ``p`` implies world offset ``p - HALF`` from that
agent. Identity inversion: ``round(hue * n_cam) % n_cam`` recovers the boid index
(hue == index / n_cam at generation), with a half-slot tolerance so a wrong-hue
dart is rejected rather than misassigned (achromatic darts read as white = NaN).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from modeling.eval import gt_project as gp

HALF = gp.PARTIAL_SIZE // 2


def _identify(hue: np.ndarray, n_cam: int, tol: float = 0.5) -> np.ndarray:
    """hue (N,) in [0,1) (NaN = white) -> identity boid index (N,), -1 if white/ambiguous.

    A camera dart's hue should be exactly ``ident / n_cam``; assign to the nearest
    slot and reject when the circular hue error exceeds ``tol / n_cam`` (half a
    slot by default -> unambiguous nearest).
    """
    hue = np.asarray(hue, np.float32)
    ident = np.full(len(hue), -1, np.int64)
    ok = ~np.isnan(hue)
    if ok.any():
        q = (np.round(hue[ok] * n_cam).astype(np.int64)) % n_cam
        err = np.abs(hue[ok] - q.astype(np.float32) / n_cam)
        err = np.minimum(err, 1.0 - err)            # circular distance
        ident[np.nonzero(ok)[0]] = np.where(err <= tol / n_cam, q, -1)
    return ident


def _find(view: dict, target: int, n_cam: int):
    """Pixel ``(u, v)`` of the dart identified as ``target`` in ``view``, or None.

    On multiple candidates picks the one whose hue is closest to the exact
    expected ``target / n_cam``. (No hue-error is returned: it would only ever
    reflect darts already accepted as the right color, so it cannot measure
    identity stability -- a color flip shows up as a missed/mismatched sighting.)
    """
    cents = np.asarray(view["centroids"], np.float32)
    if len(cents) == 0:
        return None
    hues = np.asarray(view["hue"], np.float32)
    cand = np.nonzero(_identify(hues, n_cam) == target)[0]
    if len(cand) == 0:
        return None
    d = np.abs(hues[cand] - target / n_cam)
    d = np.minimum(d, 1.0 - d)
    return cents[int(cand[int(np.argmin(d))])]


def _whites(view: dict) -> np.ndarray:
    """White (achromatic, NaN-hue) dart centroids (M,2) -- the third-party boids."""
    cents = np.asarray(view["centroids"], np.float32)
    if len(cents) == 0:
        return cents
    return cents[np.isnan(np.asarray(view["hue"], np.float32))]


def _overlap_whites(wa: np.ndarray, wb: np.ndarray, d_hat: np.ndarray,
                    size: int = gp.PARTIAL_SIZE, max_dist: float = 8.0):
    """Third-party whites each view puts in the shared region, matched across views.

    ``d_hat`` is b's world offset from a (px); the a->b pixel map is ``p_b = p_a - d_hat``.
    Returns ``(n_matched, n_overlap, count_a, count_b)``.
    """
    def _in(pts, shift):                     # pts mapped by +shift land inside the other crop
        if len(pts) == 0:
            return pts
        q = pts + shift
        keep = (q[:, 0] >= 0) & (q[:, 0] < size) & (q[:, 1] >= 0) & (q[:, 1] < size)
        return pts[keep]

    in_a = _in(wa, -d_hat)                    # a-whites that fall in b's crop
    in_b = _in(wb, +d_hat)                    # b-whites that fall in a's crop
    ca, cb = len(in_a), len(in_b)
    if ca == 0 or cb == 0:
        return 0, max(ca, cb), ca, cb
    cost = np.linalg.norm((in_a - d_hat)[:, None, :] - in_b[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(cost)
    n_match = int(sum(1 for i, j in zip(ri, ci) if cost[i, j] <= max_dist))
    return n_match, max(ca, cb), ca, cb


def pair_consistency(dets, cam_idx, n_cam: int, max_dist: float = 8.0) -> dict:
    """GT-free cross-view consistency over all camera-agent pairs / frames.

    Args:
        dets:    ``dets[a][t]`` -> dict ``{'centroids': (N,2) (u,v), 'hue': (N,)}``.
        cam_idx: camera slot a -> 0-based boid index (its identity hue = idx / n_cam).
        n_cam:   number of identity-colored camera agents (the hue period).
    Returns a metric dict. NO ``gt_pos``: covisibility is whatever the model drew.
    """
    P = len(cam_idx)
    n_sight = n_recip = n_pairframes = 0
    disp_res, motion_res, count_err = [], [], []
    n_match_w = n_over_w = 0
    for a in range(P):
        for b in range(a + 1, P):
            prev = None                       # (d_ab, d_ba) from previous frame, iff reciprocal
            for t in range(min(len(dets[a]), len(dets[b]))):
                n_pairframes += 1
                ra = _find(dets[a][t], cam_idx[b], n_cam)   # b in a's view
                rb = _find(dets[b][t], cam_idx[a], n_cam)   # a in b's view
                n_sight += int(ra is not None) + int(rb is not None)
                cur = None
                if ra is not None and rb is not None:       # reciprocal sighting
                    n_recip += 1
                    d_ab, d_ba = ra - HALF, rb - HALF
                    disp_res.append(float(np.linalg.norm(d_ab + d_ba)))
                    d_hat = 0.5 * (d_ab - d_ba)             # best pose (d_ba ~ -d_ab)
                    m, o, ca, cb = _overlap_whites(_whites(dets[a][t]), _whites(dets[b][t]),
                                                   d_hat, max_dist=max_dist)
                    n_match_w += m; n_over_w += o
                    count_err.append(abs(ca - cb))
                    if prev is not None:                    # consecutive reciprocal frames only
                        motion_res.append(float(np.linalg.norm(
                            (d_ab - prev[0]) + (d_ba - prev[1]))))
                    cur = (d_ab, d_ba)
                prev = cur
    return {
        # --- agreement (read against the volume below) ---
        "reciprocity_rate": (2 * n_recip) / max(n_sight, 1),        # frac of sightings that are mutual
        "displacement_error": float(np.mean(disp_res)) if disp_res else float("nan"),   # px, ||d_ab+d_ba||
        "motion_error": float(np.mean(motion_res)) if motion_res else float("nan"),     # px/frame
        "white_correspondence": n_match_w / max(n_over_w, 1),       # third-party matched / chances
        "white_count_error": float(np.mean(count_err)) if count_err else float("nan"),
        # --- volume / sample sizes (a self-triggered rate is meaningless without these) ---
        "sightings_per_frame": n_sight / max(n_pairframes, 1),
        "n_sightings": n_sight,
        "n_reciprocal": n_recip,
        "n_matched_white": n_match_w,
        "n_overlap_white": n_over_w,
    }


# --------------------------------------------------------------------------- #
# Dense warped-overlap similarity (PSNR / SSIM) -- the pixel-level counterpart
# --------------------------------------------------------------------------- #
def _to01(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, np.float32)
    return x / 255.0 if x.max() > 1.5 else x


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR (dB) for two [0,1] RGB crops of equal shape (data range 1.0)."""
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse < 1e-10 else float(-10.0 * np.log10(mse))


def _ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Mean windowed SSIM over [0,1] RGB crops (box window; scipy, no skimage)."""
    from scipy.ndimage import uniform_filter
    h, w = a.shape[:2]
    win = min(win, h, w)
    win -= 1 - (win % 2)                       # force odd
    if win < 3:
        return float("nan")
    c1, c2 = 0.01 ** 2, 0.03 ** 2              # (K*L)^2 with L=1
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx, my = uniform_filter(x, win), uniform_filter(y, win)
        vx = uniform_filter(x * x, win) - mx * mx
        vy = uniform_filter(y * y, win) - my * my
        vxy = uniform_filter(x * y, win) - mx * my
        s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        vals.append(float(np.mean(s)))
    return float(np.mean(vals))


def pixel_consistency(frames, dets, cam_idx, n_cam: int, min_overlap: int = 8) -> dict:
    """Dense warped-overlap similarity for every reciprocal sighting (GT-free).

    For each frame where a and b see each other, recover the pose ``d_hat`` from
    the sighting (no GT), crop both views to the implied shared rectangle, align by
    the integer offset, and score PSNR + SSIM. Because both views render the same
    identities and walls, a spatially-consistent model yields near-identical crops.

    Args:
        frames:  ``frames[a]`` -> ``(T, H, W, 3)`` decoded RGB for camera slot a.
        dets:    ``dets[a][t]`` -> detect_boids dict (used only to recover d_hat).
    Returns overall PSNR/SSIM, event count, mean overlap fraction, and the raw
    per-event ``(overlap_frac, psnr, ssim)`` list for area stratification.
    """
    P = len(cam_idx)
    psnrs, ssims, events = [], [], []
    for a in range(P):
        for b in range(a + 1, P):
            n = min(len(dets[a]), len(dets[b]), len(frames[a]), len(frames[b]))
            for t in range(n):
                ra = _find(dets[a][t], cam_idx[b], n_cam)
                rb = _find(dets[b][t], cam_idx[a], n_cam)
                if ra is None or rb is None:
                    continue
                d_hat = 0.5 * ((ra - HALF) - (rb - HALF))           # b - a, symmetrized
                dx, dy = int(round(float(d_hat[0]))), int(round(float(d_hat[1])))
                size = frames[a].shape[1]
                u0, u1 = max(0, dx), min(size, size + dx)            # overlap cols in a's frame
                v0, v1 = max(0, dy), min(size, size + dy)            # overlap rows in a's frame
                if (u1 - u0) < min_overlap or (v1 - v0) < min_overlap:
                    continue
                ca = _to01(frames[a][t][v0:v1, u0:u1])
                cb = _to01(frames[b][t][v0 - dy:v1 - dy, u0 - dx:u1 - dx])  # same region in b's frame
                if ca.shape != cb.shape:
                    continue
                ps, ss = _psnr(ca, cb), _ssim(ca, cb)
                frac = (u1 - u0) * (v1 - v0) / float(size * size)
                psnrs.append(ps); ssims.append(ss); events.append((frac, ps, ss))
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "mean_overlap_frac": float(np.mean([e[0] for e in events])) if events else float("nan"),
        "n_pixel_events": len(events),
        "events": events,                       # (overlap_frac, psnr, ssim) for area stratification
    }

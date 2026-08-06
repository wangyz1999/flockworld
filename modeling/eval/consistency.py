"""Per-view fidelity (Tier A) + cross-view position-matching helpers.

Tier A is GT-ANCHORED fidelity: match detections to GT-projected boids ->
detection rate + position error. It legitimately drops for a divergent model
(that is what a fidelity axis is for) and is kept separate from consistency.

Cross-view CONSISTENCY used to live here as "Tier B", but it leaked GT through
the camera: it placed each view's boids in world coordinates using ``gt_pos``,
which conflated genuine cross-view disagreement with per-view camera drift. It
has been removed in favour of the fully GT-free
``modeling.eval.pair_consistency`` -- in the colored-agent setup the camera
agents localize each other, so no GT enters the scoring frame or event selection.

The position-matching helpers below (``implied_world``, ``_in_crop``,
``match_cross_view``) are GT-free *given* a camera position, and are still used by
the corner-anchored probe (``probe_step3_quantify.py``), which recovers that
camera position from rendered walls rather than from GT -- so they stay.

Detections are ``dets[agent][frame] -> (N,2)`` centroids; ``cam_idx[a]`` is the
0-based boid index of camera agent a.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from modeling.eval import gt_project as gp

HALF = gp.PARTIAL_SIZE // 2


def implied_world(centroids: np.ndarray, agent_xy) -> np.ndarray:
    """Detected pixels (N,2) in an agent's view -> implied world positions (N,2)."""
    if len(centroids) == 0:
        return np.zeros((0, 2), np.float32)
    return centroids.astype(np.float32) - HALF + np.asarray(agent_xy, np.float32)


def _in_crop(world_pts: np.ndarray, agent_xy, size: int = gp.PARTIAL_SIZE) -> np.ndarray:
    if len(world_pts) == 0:
        return np.zeros(0, bool)
    return gp.visible(gp.project(world_pts, np.asarray(agent_xy, np.float32), size), size)


def match_cross_view(wa: np.ndarray, wb: np.ndarray, max_dist: float = 8.0, amb: float = 0.7):
    """Match A-world-points to B-world-points by position (3-stage minus heading).

    Stage 1 (position): optimal assignment on world distance, capped at max_dist.
    Stage 3 (min-separation fallback): if a second B-point is nearly as close
    (best > amb * second-best), the pair is ambiguous -> excluded.
    (Stage 2, the heading tiebreaker, is deferred — those cases are excluded for now.)

    Returns ``(pairs, n_excluded)`` with pairs a list of ``(ia, ib)``.
    """
    if len(wa) == 0 or len(wb) == 0:
        return [], 0
    cost = np.linalg.norm(wa[:, None, :] - wb[None, :, :], axis=-1)  # (Na, Nb)
    ri, ci = linear_sum_assignment(cost)
    pairs, excl = [], 0
    for i, j in zip(ri, ci):
        if cost[i, j] > max_dist:
            continue                                  # too far -> not the same boid
        srt = np.sort(cost[i])
        if len(srt) > 1 and cost[i, j] > amb * srt[1]:
            excl += 1                                 # ambiguous -> excluded (stage 3)
            continue
        pairs.append((i, j))
    return pairs, excl


def tier_a(dets, gt_pos: np.ndarray, cam_idx, max_dist: float = 6.0) -> dict:
    """Per-view fidelity vs GT: detection rate + position error (matched to GT)."""
    P, T = len(cam_idx), gt_pos.shape[0]
    hits = total = 0
    errs = []
    det_counts, gt_counts = [], []
    for a in range(P):
        for t in range(min(T, len(dets[a]))):
            idx, gpx = gp.visible_boids(gt_pos[t], cam_idx[a])   # GT-visible boids + pixels
            c = np.asarray(dets[a][t])
            det_counts.append(len(c)); gt_counts.append(len(gpx))
            for u, v in gpx:
                if len(c) == 0:
                    total += 1; continue
                d = np.min(np.linalg.norm(c - np.array([u, v]), axis=1))
                total += 1
                if d <= max_dist:
                    hits += 1; errs.append(float(d))
    return {
        "detection_rate": hits / max(total, 1),
        "position_error": float(np.mean(errs)) if errs else float("nan"),  # px, matched only
        "mean_detected_per_frame": float(np.mean(det_counts)) if det_counts else 0.0,
        "mean_gt_visible_per_frame": float(np.mean(gt_counts)) if gt_counts else 0.0,
    }

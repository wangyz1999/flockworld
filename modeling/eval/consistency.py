"""Cross-view consistency (Tier B) + per-view fidelity (Tier A) metrics.

Tier B is GT-ROBUST: it matches the model's OWN darts **across the two views**
by implied world position and measures their disagreement, using GT only for the
camera positions and the co-visible candidate count. So a model rendering a
plausible-but-not-GT future still scores consistent if both views agree.

Tier A is GT-ANCHORED fidelity: match darts to GT-projected boids -> detection
rate + position error. It legitimately drops for a divergent model (different axis).

Detections are passed in as ``dets[agent][frame] -> (N,2)`` centroids (decoupled
from decode/detection). ``cam_idx[a]`` is the 0-based boid index of camera agent a.

NOTE (deferred refinements): the heading tiebreaker (stage 2 of matching) and a
proper per-boid temporal std (needs cross-frame tracking) are not in yet — the
current temporal metric is the std of matched residuals (a variability proxy).
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


def tier_b(dets, gt_pos: np.ndarray, cam_idx, max_dist: float = 8.0) -> dict:
    """Cross-view consistency over all camera-agent pairs / frames.

    Args:
        dets:    dets[agent][frame] -> (N,2) detected centroids (u,v).
        gt_pos:  (T, num_boids, 2) GT world positions.
        cam_idx: list of 0-based boid index for each camera agent.
    Returns a metric dict (see keys below).
    """
    P, T = len(cam_idx), gt_pos.shape[0]
    res_series = {}            # (a,b) -> list of matched residual norms
    n_matched = n_excl = n_overlap = 0
    covis_total = covis_frames = total_frames = 0
    for a in range(P):
        for b in range(a + 1, P):
            res_series[(a, b)] = []
            for t in range(min(T, len(dets[a]), len(dets[b]))):
                total_frames += 1
                ax, bx = gt_pos[t, cam_idx[a]], gt_pos[t, cam_idx[b]]
                wa = implied_world(np.asarray(dets[a][t]), ax)
                wb = implied_world(np.asarray(dets[b][t]), bx)
                wa_o, wb_o = wa[_in_crop(wa, bx)], wb[_in_crop(wb, ax)]   # overlap region
                covis = gp.covisible_boids(gt_pos[t], cam_idx[a], cam_idx[b])[0]
                covis_total += len(covis)
                covis_frames += int(len(covis) > 0)
                pairs, excl = match_cross_view(wa_o, wb_o, max_dist)
                n_matched += len(pairs); n_excl += excl
                n_overlap += max(len(wa_o), len(wb_o))
                for i, j in pairs:
                    res_series[(a, b)].append(float(np.linalg.norm(wa_o[i] - wb_o[j])))
    flat = np.array([r for v in res_series.values() for r in v], np.float32)
    stds = [np.std(v) for v in res_series.values() if len(v) > 1]
    return {
        "positional_consistency": float(flat.mean()) if len(flat) else float("nan"),  # mean ‖r‖ (px)
        "temporal_std": float(np.mean(stds)) if stds else float("nan"),               # variability proxy
        "correspondence": n_matched / max(n_overlap, 1),    # content completeness (matched / overlap darts)
        "exclusion_rate": n_excl / max(n_matched + n_excl, 1),
        "covisibility_rate": covis_frames / max(total_frames, 1),  # frac of pair-frames with a co-visible boid
        "mean_covisible_per_frame": covis_total / max(total_frames, 1),
        "n_matched": n_matched,
    }


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

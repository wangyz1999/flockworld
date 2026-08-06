"""GT-free cross-view identity + consistency for the no-color, no-bg-gradient
setup (Setup 1).

Without color there is no way to read "which boid is agent B" off a single
frame the way ``pair_consistency.py`` does via hue. Identity is instead
INFERRED: every camera agent's own egocentric video already shows that agent
itself, always within ``FOCAL_EXCLUDE_PX`` of crop center (same convention as
``pair_consistency``), with a heading (``boid_detect._boid_heading``,
validated to ~1 deg median error against GT on real frames -- see
``validate_heading_detect.py``). That is a GT-free, per-agent heading
"fingerprint" over time, read straight from that agent's own rendered output.

A candidate boid in agent A's view is hypothesized to be agent B if, over a
short bootstrap window, its heading trace tracks B's own fingerprint AND
(independently, in B's own view) a candidate boid there tracks A's fingerprint,
AND the two candidates' implied relative offsets are mutually reciprocal
(``pair_consistency``'s ``d_ab + d_ba ~ 0`` check) over that same window. All
three holding at once, for several consecutive frames, is the accept
criterion -- a single frame's position+heading match is only the seed, never
the identification. Once accepted, the pairing rides the two per-view
tracklets (``link_tracks``) forward for as long as both stay linked -- no
per-frame re-seeding (temporal tracking, not per-frame independent matching).

Reported per ``heading_identity_consistency()``:
  * ``acceptance_rate`` / ``ambiguity_rate`` + volumes -- how often a mutually
    consistent identification spontaneously emerges at all, and how often an
    accepted one had >1 surviving candidate (i.e. was itself ambiguous). This
    is the headline number: an "emergent identity consistency" rate, read next
    to its volume like every other rate in this eval suite.
  * ``displacement_error`` / ``motion_error`` -- the same pair_consistency
    formulas, computed ONLY on the held-out frames AFTER each pairing's
    bootstrap window, so the score isn't computed on the exact frames used to
    accept the pairing.

Caveat -- read before trusting any number this produces: boid flocking
includes heading ALIGNMENT with neighbors, so nearby boids' heading traces are
correlated by construction, exactly the regime where a wrong identification is
most likely and most costly (a decoy right next to the real match). The
reciprocity check is the main defense against this (it constrains relative
geometry, which alignment does not touch), but this remains a soft,
non-deterministic identifier by design -- it is not meant to be read as a
ground-truth-grade correspondence.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from modeling.eval import gt_project as gp

HALF = gp.PARTIAL_SIZE // 2
FOCAL_EXCLUDE_PX = 8.0   # matches pair_consistency: the viewer itself sits at crop center


def _ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular absolute angular difference (radians)."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def self_heading_trace(dets: list) -> np.ndarray:
    """Per-frame heading (radians, NaN if absent) of the view owner itself.

    The owner sits within ``FOCAL_EXCLUDE_PX`` of crop center every
    frame (egocentric crop, by construction) -- the one detection whose identity
    is known for free: it's whoever owns this video, zero GT needed.
    """
    T = len(dets)
    head = np.full(T, np.nan, np.float32)
    for t in range(T):
        cents = np.asarray(dets[t]["centroids"], np.float32)
        if len(cents) == 0:
            continue
        d = np.linalg.norm(cents - HALF, axis=1)
        i = int(np.argmin(d))
        if d[i] <= FOCAL_EXCLUDE_PX:
            head[t] = dets[t]["heading"][i]
    return head


def _other_boids(dets_t: dict):
    """Non-focal (centroids, headings) in one frame -- candidate sightings of someone else."""
    cents = np.asarray(dets_t["centroids"], np.float32)
    if len(cents) == 0:
        return cents, np.zeros(0, np.float32)
    away = np.linalg.norm(cents - HALF, axis=1) > FOCAL_EXCLUDE_PX
    return cents[away], np.asarray(dets_t["heading"], np.float32)[away]


def link_tracks(dets: list, max_step: float = 20.0, min_len: int = 1) -> list:
    """Greedy nearest-position frame-to-frame linker over non-focal boids (GT-free).

    No identity to link on, only proximity -- cheap, and breaks on ambiguity
    (crossing paths, occlusion) rather than mis-linking, which is fine here: a
    broken track just ends a pairing early instead of silently swapping who's
    who mid-track.

    Returns a list of tracks: ``{'t0', 't1' (exclusive), 'pos' (L,2), 'heading' (L,)}``.
    """
    tracks = []
    open_tracks = []   # each: {'t0', 'pos': [...], 'heading': [...]}; last pos == current position
    for t in range(len(dets)):
        cents, heads = _other_boids(dets[t])
        prev_pos = (np.stack([tr["pos"][-1] for tr in open_tracks])
                    if open_tracks else np.zeros((0, 2), np.float32))
        matched_cur = {}
        if len(prev_pos) and len(cents):
            cost = np.linalg.norm(prev_pos[:, None, :] - cents[None, :, :], axis=-1)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] <= max_step:
                    matched_cur[i] = j
        new_open = []
        for i, tr in enumerate(open_tracks):
            if i in matched_cur:
                j = matched_cur[i]
                tr["pos"].append(cents[j]); tr["heading"].append(heads[j])
                new_open.append(tr)
            elif len(tr["pos"]) >= min_len:
                tracks.append(_finalize(tr))
        used_cur = set(matched_cur.values())
        for j in range(len(cents)):
            if j not in used_cur:
                new_open.append({"t0": t, "pos": [cents[j]], "heading": [heads[j]]})
        open_tracks = new_open
    for tr in open_tracks:
        if len(tr["pos"]) >= min_len:
            tracks.append(_finalize(tr))
    return tracks


def _finalize(tr: dict) -> dict:
    pos = np.stack(tr["pos"]).astype(np.float32)
    heading = np.asarray(tr["heading"], np.float32)
    return {"t0": tr["t0"], "t1": tr["t0"] + len(pos), "pos": pos, "heading": heading}


def _track_slice(track: dict, g0: int, g1: int):
    """(pos, heading, s0, s1) for the overlap of ``track`` with global frames [g0, g1).

    ``(s0, s1)`` is that overlap's own global bounds -- callers compare it
    against ``(g0, g1)`` to require full window coverage. ``None`` if no overlap.
    """
    s0, s1 = max(g0, track["t0"]), min(g1, track["t1"])
    if s1 <= s0:
        return None
    i0, i1 = s0 - track["t0"], s1 - track["t0"]
    return track["pos"][i0:i1], track["heading"][i0:i1], s0, s1


def _heading_match(track_heading: np.ndarray, self_heading: np.ndarray,
                    thresh_rad: float, min_frac: float = 0.75):
    """Mean circular angular error between a candidate track and a self-heading
    fingerprint over an already-aligned window. Requires >= min_frac of the
    window to have both values defined (else inconclusive -> reject).
    """
    valid = ~np.isnan(track_heading) & ~np.isnan(self_heading)
    if valid.sum() < min_frac * len(track_heading):
        return False, float("nan")
    err = _ang_diff(track_heading[valid], self_heading[valid])
    mean_err = float(np.mean(err))
    return mean_err <= thresh_rad, mean_err


def _reciprocity(pos_a: np.ndarray, pos_b: np.ndarray, max_dist: float):
    """Mean ``||d_ab + d_ba||`` (px) over an already-aligned window."""
    d_ab, d_ba = pos_a - HALF, pos_b - HALF
    disp = np.linalg.norm(d_ab + d_ba, axis=1)
    mean_disp = float(np.mean(disp))
    return mean_disp <= max_dist, mean_disp


def heading_identity_consistency(dets: list, heading_thresh_deg: float = 30.0,
                                  w_boot: int = 8, max_dist: float = 4.0,
                                  max_step: float = 20.0, return_pairs: bool = False):
    """GT-free identity resolution + consistency scoring across all camera-agent pairs.

    Args:
        dets: ``dets[a][t]`` -> ``detect_boids()`` output dict (needs 'heading').
        heading_thresh_deg: max mean angular error (deg) to accept a heading match.
        w_boot: bootstrap window length (frames) used only to decide acceptance;
            scoring runs on the held-out frames after it.
        max_dist: max mean reciprocity error (px) to accept a candidate pairing.
            Tuned via ``tune_heading_identity.py`` against real GT identity (not
            used at runtime, GT-free once tuned): reciprocity strictness, not
            heading strictness, is what actually separates correct from wrong
            identifications (flocking heading-alignment makes heading a weak
            discriminator between nearby boids; relative-geometry reciprocity
            is a much stronger, largely independent one). 4.0px took pair-level
            precision from 83% (8.0px) to 100% at only ~19% less volume.
        max_step: max px/frame for the per-view tracker to keep linking a boid.
        return_pairs: if True, also return the raw list of accepted pairing
            records (``{'a', 'b', 't0', 'g1', 'surv_end', 'ta', 'tb'}``) --
            for external validation (e.g. cross-checking against GT), not used
            by the metric itself.
    """
    thresh = np.radians(heading_thresh_deg)
    P = len(dets)
    self_head = [self_heading_trace(dets[a]) for a in range(P)]
    tracks = [link_tracks(dets[a], max_step=max_step) for a in range(P)]

    n_attempts = n_accepted = n_ambiguous = 0
    survival = []
    disp_res, motion_res = [], []
    pairs = []

    for a in range(P):
        for b in range(P):
            if a == b:
                continue
            for ta in tracks[a]:
                g0, g1 = ta["t0"], ta["t0"] + w_boot
                if ta["t1"] < g1:
                    continue                                    # too short to bootstrap
                n_attempts += 1
                sl_a = _track_slice(ta, g0, g1)
                ok_h, _ = _heading_match(sl_a[1], self_head[b][g0:g1], thresh)
                if not ok_h:
                    continue
                candidates = []
                for tb in tracks[b]:
                    sl_b = _track_slice(tb, g0, g1)
                    if sl_b is None or sl_b[2] != g0 or sl_b[3] != g1:
                        continue                                # must fully cover the window
                    ok_hb, _ = _heading_match(sl_b[1], self_head[a][g0:g1], thresh)
                    if not ok_hb:
                        continue
                    ok_r, disp = _reciprocity(sl_a[0], sl_b[0], max_dist)
                    if not ok_r:
                        continue
                    candidates.append((disp, tb))
                if not candidates:
                    continue
                if len(candidates) > 1:
                    n_ambiguous += 1
                candidates.sort(key=lambda c: c[0])
                _, tb = candidates[0]
                n_accepted += 1
                surv_end = min(ta["t1"], tb["t1"])
                survival.append(surv_end - g0)
                if return_pairs:
                    pairs.append({"a": a, "b": b, "t0": g0, "g1": g1, "surv_end": surv_end,
                                  "ta": ta, "tb": tb})
                prev = None                                     # held-out scoring
                for t in range(g1, surv_end):
                    pa = ta["pos"][t - ta["t0"]] - HALF
                    pb = tb["pos"][t - tb["t0"]] - HALF
                    disp_res.append(float(np.linalg.norm(pa + pb)))
                    if prev is not None:
                        motion_res.append(float(np.linalg.norm((pa - prev[0]) + (pb - prev[1]))))
                    prev = (pa, pb)

    result = {
        # --- headline: how often + how long identity spontaneously emerges ---
        "acceptance_rate": n_accepted / max(n_attempts, 1),
        "ambiguity_rate": n_ambiguous / max(n_accepted, 1),
        "n_attempts": n_attempts,
        "n_accepted": n_accepted,
        "n_ambiguous": n_ambiguous,
        "median_survival_frames": float(np.median(survival)) if survival else float("nan"),
        "mean_survival_frames": float(np.mean(survival)) if survival else float("nan"),
        # --- secondary: held-out consistency among accepted pairings ---
        "displacement_error": float(np.mean(disp_res)) if disp_res else float("nan"),
        "motion_error": float(np.mean(motion_res)) if motion_res else float("nan"),
        "n_held_out_frames": len(disp_res),
    }
    if return_pairs:
        return result, pairs
    return result

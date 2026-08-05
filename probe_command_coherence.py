"""Diagnostic 2 — is the action we keep feeding still coherent with what the model drew?

Diagnostic 1 (probe_action_sensitivity.py) asks whether the model RESPONDS to the action.
This one asks the prior question: how wrong is the action by the time we feed it?

The rollout protocol conditions on the LOGGED acceleration for the whole horizon. That
acceleration is a function of the true neighbourhood:

    a_t = w_a*steer(alignment) + w_c*steer(cohesion) + w_s*steer(separation)
    steer(u) = clip(max_speed*unit(u) - v_own, max_force)

over neighbours inside ``vision`` = 25 px (flockworld/core/boids.py). The crop is 128 px
and pure translation (gt_project.project: u = x - x_agent + 64), so pixel offsets from
crop centre ARE world offsets 1:1 and the entire 25 px neighbourhood is inside the frame.
``accuracy``=32 exceeds the 3x3-cell candidate count at this density, so ``sample_prob``
is 1.0 and the map is effectively deterministic: the commanded action is a function of
what the frame shows.

So we can READ the implied action off any frame, generated or real:

  Stage 1 (calibrate on REAL frames). Decode the GT latents, detect darts, build the
    rule's own equivariant features (unit cohesion / separation / alignment directions,
    own-velocity direction from the centre dart's heading, and the raw vectors), and fit
    them to the recorded action by least squares. Weights are shared across x and y --
    the rule is rotation-equivariant, so this is a 7-parameter fit, not 2x8 free params.
    Episode-held-out R^2 answers "how much of the commanded action is redundant given a
    single rendered frame" -- the training-time premise of the shortcut.

  Stage 2 (apply to ROLLOUT frames). Roll out under GT actions as usual, read the implied
    action off the model's own generated frames, and compare it to the command actually
    fed, as a function of rollout time. The same read on the GT-decoded frames is the
    detector+probe noise floor.

The gap between those two curves IS the concern, quantified: by rollout second t, the
action we are still feeding is off by X degrees from what the rendered neighbourhood
implies.

Usage:
  uv run python probe_command_coherence.py --include exp08_tiled_two_stage \
      --episodes 6 --seconds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling.eval import gt_project as gp
from modeling.eval.boid_detect import detect_boids
from modeling.eval.pair_consistency import FOCAL_EXCLUDE_PX
from eval_flock_multi import _load_model
from compare_experiments import EXPERIMENTS, RUN_ROOT
from modeling import flow_matching as fm

HALF = gp.PARTIAL_SIZE / 2.0
VISION = 25.0          # boids.vision (px) -- same radius for all three rules
MAX_SPEED = 4.0        # boids.max_speed, the setMag target in Reynolds steering
N_FEAT = 7             # unit(coh), unit(sep), unit(aln), unit(v_own), raw coh, sep, aln


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros(2, np.float64)


def frame_features(det) -> np.ndarray | None:
    """One decoded frame's detections -> (N_FEAT, 2) equivariant features, or None.

    Rows are 2-vectors that the rule combines linearly (up to its clip), so a single
    weight per row shared across x and y is the equivariant parameterisation.
    """
    cents = np.asarray(det["centroids"], np.float64)
    if len(cents) == 0:
        return None
    head = np.asarray(det["heading"], np.float64)
    rel = cents - HALF                                   # crop-centre offsets == world offsets
    dist = np.linalg.norm(rel, axis=1)

    own = dist <= FOCAL_EXCLUDE_PX                       # the view owner's own dart
    v_own = np.zeros(2, np.float64)
    if own.any():
        h = head[own][np.argmin(dist[own])]
        if np.isfinite(h):
            # heading = atan2(row_dir, col_dir), so (col, row) = (cos h, sin h) and rel's
            # axes are already (u=col, v=row) -- no swap.
            v_own = np.array([np.cos(h), np.sin(h)])

    nb = (~own) & (dist < VISION) & (dist > 1e-3)
    if not nb.any():
        return np.zeros((N_FEAT, 2), np.float64)
    r, d = rel[nb], dist[nb]
    coh = r.mean(0)                                      # seek toward mean neighbour position
    sep = (-r / (d ** 2)[:, None]).sum(0)                # sum (self-other)/sqrDist
    hn = head[nb]
    ok = np.isfinite(hn)
    aln = (np.stack([np.cos(hn[ok]), np.sin(hn[ok])], 1).mean(0)
           if ok.any() else np.zeros(2))
    return np.stack([MAX_SPEED * _unit(coh), MAX_SPEED * _unit(sep), MAX_SPEED * _unit(aln),
                     v_own, coh, sep, aln])


def oracle_features(pos, vel, focal: int) -> np.ndarray:
    """Same features, built from the sim's EXACT boid state instead of detections.

    ``pos`` (N, 2) and ``vel`` (N, 2) at one tick, ``focal`` = the camera agent's boid
    index. This is the control for stage 1: if the readout cannot recover the action from
    perfect state, the feature set is wrong; if it can, any failure on decoded frames is
    the detector's doing, not the readout's.
    """
    rel = pos - pos[focal]
    dist = np.linalg.norm(rel, axis=1)
    nb = (dist < VISION) & (dist > 1e-3)
    v_own = _unit(vel[focal])
    if not nb.any():
        return np.zeros((N_FEAT, 2), np.float64)
    r, d = rel[nb], dist[nb]
    coh, sep = r.mean(0), (-r / (d ** 2)[:, None]).sum(0)
    aln = np.stack([_unit(v) for v in vel[nb]]).mean(0)
    return np.stack([MAX_SPEED * _unit(coh), MAX_SPEED * _unit(sep), MAX_SPEED * _unit(aln),
                     v_own, coh, sep, aln])


def collect_oracle(pos_seq, cam_idx, actions, n_lat):
    """pos_seq (T_pix, N, 2) + cam_idx -> (X, Y, K) using exact state (velocity by
    finite differences, dt = 1 tick, frame_stride = 1)."""
    vel = np.diff(pos_seq, axis=0, prepend=pos_seq[:1])
    X, Y, K = [], [], []
    for j, focal in enumerate(cam_idx):
        feats = agg_to_latent([oracle_features(pos_seq[t], vel[t], focal)
                               for t in range(len(pos_seq))], n_lat)
        for k, f in enumerate(feats):
            if f is None:
                continue
            X.append(f); Y.append(actions[j, k]); K.append(k)
    return np.array(X), np.array(Y), np.array(K)


def agg_to_latent(feats, n_lat):
    """Pixel-rate features -> latent rate, using the SAME causal 1+4k grouping as
    StreamingFlockDataset._agg_actions so features line up with the fed actions."""
    out = []
    for k in range(n_lat):
        grp = [feats[0]] if k == 0 else feats[1 + 4 * (k - 1): 1 + 4 * k]
        grp = [g for g in grp if g is not None]
        out.append(np.mean(grp, axis=0) if grp else None)
    return out


def collect(frames_all, actions, n_lat, background_size=None):
    """(per-agent decoded frames, actions (P,L,2)) -> (X (M,N_FEAT), Y (M,2), lat_idx (M,))."""
    X, Y, K = [], [], []
    for j, ag in enumerate(frames_all):
        feats = agg_to_latent([frame_features(detect_boids(f, background_size=background_size))
                               for f in ag], n_lat)
        for k, f in enumerate(feats):
            if f is None:
                continue
            X.append(f); Y.append(actions[j, k]); K.append(k)
    if not X:
        return (np.zeros((0, N_FEAT)), np.zeros((0, 2)), np.zeros((0,), int))
    return np.array(X), np.array(Y), np.array(K)


def fit_equivariant(X, Y, lam=1e-3):
    """Shared weights across x and y: stack the two axes into one least-squares problem.
    X (M, N_FEAT, 2), Y (M, 2) -> w (N_FEAT,)."""
    A = np.concatenate([X[:, :, 0], X[:, :, 1]], 0)      # (2M, N_FEAT)
    b = np.concatenate([Y[:, 0], Y[:, 1]], 0)            # (2M,)
    G = A.T @ A + lam * np.eye(N_FEAT) * max(1.0, np.trace(A.T @ A) / N_FEAT)
    return np.linalg.solve(G, A.T @ b)


def apply_w(X, w):
    """X (M, N_FEAT, 2), w (N_FEAT,) -> predicted actions (M, 2)."""
    return np.einsum("mfa,f->ma", X, w)


def r2(pred, Y):
    ss_res = float(((pred - Y) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def agree(pred, Y):
    """Directional + magnitude agreement between an implied action and the commanded one."""
    pn = np.linalg.norm(pred, axis=1); yn = np.linalg.norm(Y, axis=1)
    ok = (pn > 1e-6) & (yn > 1e-6)
    if not ok.any():
        return {"cos": float("nan"), "angle_deg": float("nan"), "mag_ratio": float("nan"),
                "n": 0}
    c = np.clip((pred[ok] * Y[ok]).sum(1) / (pn[ok] * yn[ok]), -1, 1)
    return {"cos": float(c.mean()), "angle_deg": float(np.degrees(np.arccos(c)).mean()),
            "mag_ratio": float((pn[ok] / yn[ok]).mean()), "n": int(ok.sum())}


def buckets(pred, Y, K, n_lat, n_bins=5):
    """Agreement per rollout-time bucket (equal splits of the latent-frame axis)."""
    edges = np.linspace(0, n_lat, n_bins + 1).astype(int)
    out = []
    for i in range(n_bins):
        m = (K >= edges[i]) & (K < edges[i + 1])
        out.append({"lat_range": [int(edges[i]), int(edges[i + 1])],
                    **(agree(pred[m], Y[m]) if m.any() else
                       {"cos": float("nan"), "angle_deg": float("nan"),
                        "mag_ratio": float("nan"), "n": 0})})
    return out


def main():
    ap = argparse.ArgumentParser(description="Command-coherence probe (diagnostic 2).")
    ap.add_argument("--include", nargs="+", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--fit-episodes", type=int, default=None,
                    help="episodes reserved for fitting the readout (default: half).")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--out", default=f"{RUN_ROOT}/eval_logs/command_coherence.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from modeling.eval.streaming_eval_dataset import (
        StreamingMultiEvalDataset, fit_train_stats, make_decode_fn, raw_decode_fn,
        normalize_frames,
    )
    from modeling.models.frozen_vae import FrozenVAE
    from modeling.eval import overlay as ov

    first_cfg = load_cfg(EXPERIMENTS[args.include[0]][0], [])
    vae = FrozenVAE(str(first_cfg.vae.checkpoint_path), device=device)
    ds = StreamingMultiEvalDataset(
        first_cfg, vae, device, num_episodes=int(args.episodes),
        seconds=float(args.seconds), sim_fps=int(args.sim_fps), eval_seed=int(args.eval_seed))

    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(first_cfg.data.num_future_frames)
    n_ep = min(int(args.episodes), len(ds.episodes))
    n_fit = int(args.fit_episodes) if args.fit_episodes is not None else max(1, n_ep // 2)
    print(f"{n_ep} episodes ({n_fit} for the readout fit, {n_ep - n_fit} held out), "
          f"{args.seconds}s = {n_lat} latent frames")

    raw_dec = raw_decode_fn(vae, device)
    eps = [ds.full_episode(e) for e in range(n_ep)]

    # ---- Stage 0: oracle control -- is the action recoverable from EXACT state? ----
    print("\n=== Stage 0: oracle readout from the sim's exact boid state ===")
    orc = []
    for ep in eps:
        T = min(n_lat, ep["frames"].shape[1])
        pos = ds.gt_positions(ep["episode_id"])
        pos = np.asarray(pos[0] if pos.ndim == 4 else pos, np.float64)   # (T_pix, N, 2)
        orc.append(collect_oracle(pos, [ai - 1 for ai in ep["agent_indices"]],
                                  ep["actions"].numpy(), T))
    Xo = np.concatenate([c[0] for c in orc[:n_fit]]); Yo = np.concatenate([c[1] for c in orc[:n_fit]])
    Xoh = np.concatenate([c[0] for c in orc[n_fit:]]); Yoh = np.concatenate([c[1] for c in orc[n_fit:]])
    w_o = fit_equivariant(Xo, Yo)
    stage0 = {"weights": w_o.tolist(), "r2_fit": r2(apply_w(Xo, w_o), Yo),
              "r2_heldout": r2(apply_w(Xoh, w_o), Yoh),
              "agree": agree(apply_w(Xoh, w_o), Yoh)}
    print(f"  oracle R^2: fit={stage0['r2_fit']:.3f}  held-out={stage0['r2_heldout']:.3f}   "
          f"angle={stage0['agree']['angle_deg']:.1f} deg  cos={stage0['agree']['cos']:.3f}")
    print("  HIGH => the action really is a function of the visible neighbourhood and the "
          "feature set captures it; the decoded-frame numbers below are then a detector "
          "question. LOW => the readout itself is wrong, and stage 1/2 mean nothing.")

    # ---- Stage 1: read the action off REAL frames ---------------------------
    print("\n=== Stage 1: fit the frame -> action readout on GT-decoded frames ===")
    ceil = []
    for i, ep in enumerate(eps):
        T = min(n_lat, ep["frames"].shape[1])
        lat = ep["frames"][:, :T]
        frames_all = [ov.to_uint8(raw_dec(lat[j]).detach().cpu()) for j in range(lat.shape[0])]
        ceil.append(collect(frames_all, ep["actions"].numpy(), T))
        print(f"  ceiling ep{ep['episode_id']} ({i + 1}/{n_ep}) samples={len(ceil[-1][0])}",
              flush=True)

    Xf = np.concatenate([c[0] for c in ceil[:n_fit]]); Yf = np.concatenate([c[1] for c in ceil[:n_fit]])
    Xh = np.concatenate([c[0] for c in ceil[n_fit:]]); Yh = np.concatenate([c[1] for c in ceil[n_fit:]])
    Kh = np.concatenate([c[2] for c in ceil[n_fit:]])
    w = fit_equivariant(Xf, Yf)
    ceil_pred = apply_w(Xh, w)
    stage1 = {"weights": w.tolist(), "r2_fit": r2(apply_w(Xf, w), Yf),
              "r2_heldout": r2(ceil_pred, Yh), "n_fit": len(Xf), "n_heldout": len(Xh),
              "agree": agree(ceil_pred, Yh)}
    print(f"  readout R^2: fit={stage1['r2_fit']:.3f}  held-out episodes={stage1['r2_heldout']:.3f}"
          f"  (n_fit={len(Xf)}, n_heldout={len(Xh)})")
    print(f"  ceiling agreement: cos={stage1['agree']['cos']:.3f}  "
          f"angle={stage1['agree']['angle_deg']:.1f} deg  "
          f"mag_ratio={stage1['agree']['mag_ratio']:.2f}")
    print("  R^2 here = how much of the commanded action is REDUNDANT given one rendered "
          "frame. High R^2 => the shortcut the model can take instead of using the action.")

    ceil_buckets = buckets(ceil_pred, Yh, Kh, n_lat, args.bins)

    # ---- Stage 2: read the action off the model's OWN frames ----------------
    results = {}
    for key in args.include:
        cfg_path, out_dir = EXPERIMENTS[key]
        print(f"\n=== Stage 2: {key} ({out_dir}) ===", flush=True)
        ecfg = load_cfg(cfg_path, [f"output_dir={out_dir}"])
        model = _load_model(ecfg, ecfg.output_dir, device)
        mean_e, std_e = fit_train_stats(ecfg, vae, device)
        dec = make_decode_fn(vae, device, mean_e, std_e)

        Xs, Ys, Ks = [], [], []
        for i, ep in enumerate(eps[n_fit:]):                 # held-out episodes only
            T = min(n_lat, ep["frames"].shape[1])
            ctx = int(ep["context_len"])
            fin = normalize_frames(ep["frames"], mean_e, std_e).unsqueeze(0).to(device)
            acts = ep["actions"].unsqueeze(0).to(device)
            with torch.no_grad():
                lat = fm.multi_autoregressive_rollout(
                    model, fin[:, :, :ctx], acts, T, window_future=wf,
                    num_steps=int(args.steps))[0]
            frames_all = [ov.to_uint8(dec(lat[j]).detach().cpu()) for j in range(lat.shape[0])]
            x, y, k = collect(frames_all, ep["actions"].numpy(), T)
            Xs.append(x); Ys.append(y); Ks.append(k)
            print(f"  [{key}] ep{ep['episode_id']} samples={len(x)} "
                  f"({i + 1}/{n_ep - n_fit})", flush=True)
        X = np.concatenate(Xs); Y = np.concatenate(Ys); K = np.concatenate(Ks)
        pred = apply_w(X, w)
        results[key] = {"overall": agree(pred, Y), "r2": r2(pred, Y),
                        "buckets": buckets(pred, Y, K, n_lat, args.bins)}
        del model
        torch.cuda.empty_cache()

    # ---- report -------------------------------------------------------------
    secs = float(args.seconds)
    def lab(b):
        lo, hi = b["lat_range"]
        return f"{lo * secs / n_lat:.1f}-{hi * secs / n_lat:.1f}s"

    print("\n############ COMMAND COHERENCE ############")
    print("Angle between the action FED to the model and the action the RENDERED "
          "neighbourhood implies.")
    print(f"\n{'rollout window':>16}" + f"{'ceiling':>12}" +
          "".join(f"{k[:11]:>12}" for k in results))
    for i in range(len(ceil_buckets)):
        row = f"{lab(ceil_buckets[i]):>16}{ceil_buckets[i]['angle_deg']:>12.1f}"
        for k in results:
            row += f"{results[k]['buckets'][i]['angle_deg']:>12.1f}"
        print(row + "   deg")
    print(f"\n{'overall':>16}{stage1['agree']['angle_deg']:>12.1f}" +
          "".join(f"{results[k]['overall']['angle_deg']:>12.1f}" for k in results) + "   deg")
    print(f"{'cos':>16}{stage1['agree']['cos']:>12.3f}" +
          "".join(f"{results[k]['overall']['cos']:>12.3f}" for k in results))
    print(f"{'R^2':>16}{stage1['r2_heldout']:>12.3f}" +
          "".join(f"{results[k]['r2']:>12.3f}" for k in results))
    print("\nceiling = the SAME readout on GT-decoded frames: detector + probe noise, the "
          "floor this metric can reach.")
    print("A ceiling-flat model column means the command stayed coherent with what was drawn.")
    print("A column that climbs with rollout time is the concern, measured: the logged action "
          "no longer describes the generated neighbourhood.")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\noracle control (exact state): R^2={stage0['r2_heldout']:.3f}  "
          f"angle={stage0['agree']['angle_deg']:.1f} deg -- validates the readout itself.")
    out.write_text(json.dumps({"args": vars(args), "stage0": stage0, "stage1": stage1,
                               "ceiling_buckets": ceil_buckets, "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

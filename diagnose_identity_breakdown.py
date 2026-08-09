"""Follow-up to diagnose_identity.py: WHERE is diffusion-forcing's colored-boid
suppression concentrated?

  (a) per-hue-slot: is the ~3x drop in colored (non-self) boids uniform across
      all 10 camera-agent identities, or concentrated in a few -- e.g. the same
      agents (2/5/9/10) whose hue was already shown to be VAE-unstable even in
      the pure ground-truth ceiling (no model involved)?
  (b) per-distance-from-center: does suppression concentrate on boids far from
      the viewing agent (near the crop edge) vs close by?

Usage:
  uv run python diagnose_identity_breakdown.py --include exp01_baseline exp03_diffusion_forcing
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling.eval.pair_consistency import _identify, FOCAL_EXCLUDE_PX, HALF
from eval_flock_multi import _decode_detect, _load_model, _num_camera_agents, _rollout_model
from compare_experiments import EXPERIMENTS

DIST_BINS = [(FOCAL_EXCLUDE_PX, 20), (20, 35), (35, 50), (50, 65), (65, 91)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", nargs="+", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eval-seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from modeling.eval.streaming_eval_dataset import (
        StreamingMultiEvalDataset, fit_train_stats, make_decode_fn, normalize_frames,
    )
    from modeling.models.frozen_vae import FrozenVAE

    first_cfg = load_cfg(EXPERIMENTS[args.include[0]][0], [])
    vae = FrozenVAE(str(first_cfg.vae.checkpoint_path), device=device)
    ds = StreamingMultiEvalDataset(
        first_cfg, vae, device, num_episodes=int(args.episodes), seconds=float(args.seconds),
        sim_fps=int(args.sim_fps), eval_seed=int(args.eval_seed),
    )
    n_cam = _num_camera_agents(first_cfg, first_cfg.data.num_agents)
    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(first_cfg.data.num_future_frames)
    n_ep = min(int(args.episodes), len(ds.episodes))
    print(f"{n_ep} episodes x {args.seconds}s, n_cam={n_cam}")

    results = {}
    for name in args.include:
        cfg_path, out_dir = EXPERIMENTS[name]
        cfg = load_cfg(cfg_path, [f"output_dir={out_dir}"])
        model = _load_model(cfg, cfg.output_dir, device)
        mean, std = fit_train_stats(cfg, vae, device)
        decode_fn = make_decode_fn(vae, device, mean, std)

        slot_identified = np.zeros(n_cam, dtype=np.int64)
        slot_ambiguous_near = np.zeros(n_cam, dtype=np.int64)  # unused placeholder, kept simple below
        bin_total = np.zeros(len(DIST_BINS), dtype=np.int64)
        bin_white = np.zeros(len(DIST_BINS), dtype=np.int64)
        bin_identified = np.zeros(len(DIST_BINS), dtype=np.int64)
        bin_ambiguous = np.zeros(len(DIST_BINS), dtype=np.int64)

        for e in range(n_ep):
            ep = ds.full_episode(e)
            ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
            frames_n = normalize_frames(ep["frames"], mean, std).unsqueeze(0).to(device)
            actions = ep["actions"].unsqueeze(0).to(device)
            lat = _rollout_model(model, frames_n, actions, ctx, T, wf, int(args.steps))
            _, dets = _decode_detect(lat, decode_fn)
            for a in range(len(dets)):
                for t in range(len(dets[a])):
                    cents = np.asarray(dets[a][t]["centroids"], np.float32)
                    if len(cents) == 0:
                        continue
                    hues = np.asarray(dets[a][t]["hue"], np.float32)
                    dist = np.linalg.norm(cents - HALF, axis=1)
                    self_mask = dist <= FOCAL_EXCLUDE_PX
                    other_hues = hues[~self_mask]
                    other_dist = dist[~self_mask]
                    if len(other_hues) == 0:
                        continue
                    white_mask = np.isnan(other_hues)
                    colored_idx = np.nonzero(~white_mask)[0]
                    ids_full = np.full(len(other_hues), -2, np.int64)   # -2 = white, else _identify's -1/slot
                    if len(colored_idx) > 0:
                        ids_full[colored_idx] = _identify(other_hues[colored_idx], n_cam)
                    for k in range(n_cam):
                        slot_identified[k] += int((ids_full == k).sum())
                    for bi, (lo, hi) in enumerate(DIST_BINS):
                        m = (other_dist >= lo) & (other_dist < hi)
                        bin_total[bi] += int(m.sum())
                        bin_white[bi] += int((m & white_mask).sum())
                        bin_identified[bi] += int((m & (ids_full >= 0)).sum())
                        bin_ambiguous[bi] += int((m & (ids_full == -1)).sum())
            print(f"  [{name}] ep{ep['episode_id']} ({e + 1}/{n_ep})", flush=True)

        results[name] = dict(slot_identified=slot_identified, bin_total=bin_total,
                              bin_white=bin_white, bin_identified=bin_identified,
                              bin_ambiguous=bin_ambiguous)
        del model
        torch.cuda.empty_cache()

    cols = args.include
    print(f"\n=== identified count per camera-agent hue slot (0-based; 1-based agent index = slot+1) ===")
    print(f"{'slot (agent)':>16}" + "".join(f"{c:>24}" for c in cols)
          + (f"{'ratio ' + cols[1] + '/' + cols[0]:>22}" if len(cols) == 2 else ""))
    for k in range(n_cam):
        row = f"{f'{k} (agent {k+1})':>16}" + "".join(f"{results[c]['slot_identified'][k]:>24}" for c in cols)
        if len(cols) == 2:
            a, b = results[cols[0]]['slot_identified'][k], results[cols[1]]['slot_identified'][k]
            row += f"{(b / max(a, 1)):>22.3f}"
        print(row)

    print(f"\n=== breakdown by distance-from-center (px), non-self detections ===")
    for bi, (lo, hi) in enumerate(DIST_BINS):
        print(f"\n-- [{lo},{hi}) px --")
        print(f"{'':>20}" + "".join(f"{c:>24}" for c in cols))
        print(f"{'total':>20}" + "".join(f"{results[c]['bin_total'][bi]:>24}" for c in cols))
        for key, label in [("bin_white", "frac white"), ("bin_identified", "frac identified"),
                            ("bin_ambiguous", "frac ambiguous")]:
            print(f"{label:>20}" + "".join(
                f"{results[c][key][bi] / max(results[c]['bin_total'][bi], 1):>24.4f}" for c in cols))


if __name__ == "__main__":
    main()

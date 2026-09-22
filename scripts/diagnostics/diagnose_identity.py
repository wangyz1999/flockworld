"""Why does diffusion-forcing have so few cross-view reciprocal sightings despite
near-correct overall detection volume (Tier A)? Decompose every NON-SELF detection
(the view owner itself, within FOCAL_EXCLUDE_PX of center, is dropped -- same
rule pair_consistency uses) into three buckets:

  white       -- achromatic (hue=NaN), the ordinary background boids.
  identified  -- colored boid whose hue lands within tolerance of some camera
                 agent's k/n_cam slot (a genuine "I can see camera-agent k" signal).
  ambiguous   -- colored boid that does NOT cleanly match any slot (rejected by
                 pair_consistency._identify) -- neither a clean self/other id nor
                 a background boid; a wrong/blended/off-slot hue.

If DF's "identified" fraction is much lower than baseline's (with white/ambiguous
picking up the difference), that's the mechanism: DF renders roughly the right
NUMBER of boids, but fewer of them carry a hue clean enough to count as a valid
cross-view identity sighting.

Usage:
  uv run python -m scripts.diagnostics.diagnose_identity --include exp01_baseline exp03_diffusion_forcing
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling.eval.pair_consistency import _identify, FOCAL_EXCLUDE_PX, HALF
from modeling.eval.multi import _decode_detect, _load_model, _num_camera_agents, _rollout_model
from modeling.eval.experiments import EXPERIMENTS, RUN_ROOT


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

        n_self = n_white = n_identified = n_ambiguous = 0
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
                    n_self += int(self_mask.sum())
                    other_hues = hues[~self_mask]
                    if len(other_hues) == 0:
                        continue
                    white_mask = np.isnan(other_hues)
                    n_white += int(white_mask.sum())
                    colored = other_hues[~white_mask]
                    if len(colored) == 0:
                        continue
                    ids = _identify(colored, n_cam)
                    n_identified += int((ids != -1).sum())
                    n_ambiguous += int((ids == -1).sum())
            print(f"  [{name}] ep{ep['episode_id']} ({e + 1}/{n_ep})", flush=True)

        n_nonself = n_white + n_identified + n_ambiguous
        results[name] = dict(n_self=n_self, n_nonself=n_nonself, n_white=n_white,
                              n_identified=n_identified, n_ambiguous=n_ambiguous)
        del model
        torch.cuda.empty_cache()

    print(f"\n=== non-self detection breakdown ({n_ep} episodes x {args.seconds}s, steps={args.steps}) ===")
    cols = args.include
    print(f"{'':>28}" + "".join(f"{c:>24}" for c in cols))
    print(f"{'n_self (excluded)':>28}" + "".join(f"{results[c]['n_self']:>24}" for c in cols))
    print(f"{'n_nonself (total)':>28}" + "".join(f"{results[c]['n_nonself']:>24}" for c in cols))
    for key, label in [("n_white", "white (bg boid)"), ("n_identified", "identified (valid hue)"),
                        ("n_ambiguous", "ambiguous (off-slot hue)")]:
        print(f"{label:>28}" + "".join(f"{results[c][key]:>24}" for c in cols))
    print()
    for key, label in [("n_white", "frac white"), ("n_identified", "frac identified"),
                        ("n_ambiguous", "frac ambiguous")]:
        print(f"{label:>28}" + "".join(
            f"{results[c][key] / max(results[c]['n_nonself'], 1):>24.4f}" for c in cols))


if __name__ == "__main__":
    main()

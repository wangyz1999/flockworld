"""Sanity-test heading_identity_consistency on real GT (VAE round-tripped)
frames -- NOT a model rollout. On real trajectories, reciprocity is exact and
headings are the true ones (modulo extractor noise), so this checks the
identity-resolution MECHANISM works at all before ever pointing it at a
model's generated video.

    uv run python test_heading_consistency.py
"""
import numpy as np, torch
from pathlib import Path
from modeling.configs import load_cfg
from train_flock_dit import build_decode_fn
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.eval import overlay as ov
from modeling.eval.boid_detect import detect_boids
from modeling.eval.heading_consistency import heading_identity_consistency

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
NEP, NLAT = 5, 38

cfg = load_cfg("config/train_flockdit_latent_multi.yaml",
                [f"data.root={ROOT}", f"output_dir={OUTD}", "vae.checkpoint_path=pretrained/no_color/last.ckpt"])
decode_fn = build_decode_fn(cfg)
ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))

agg = {"n_attempts": 0, "n_accepted": 0, "n_ambiguous": 0, "n_held_out_frames": 0}
disp_all, motion_all, surv_all = [], [], []

with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e)
        P = ep["frames"].shape[0]
        dets = []
        for j in range(P):
            frames = ov.to_uint8(decode_fn(ep["frames"][j, :NLAT]).detach().cpu())
            dets.append([detect_boids(frames[t]) for t in range(len(frames))])
        res = heading_identity_consistency(dets)
        print(f"ep{ep['episode_id']}: {res}", flush=True)
        for k in agg:
            agg[k] += res[k]
        if not np.isnan(res["displacement_error"]):
            disp_all.append((res["displacement_error"], res["n_held_out_frames"]))
        if not np.isnan(res["motion_error"]):
            motion_all.append((res["motion_error"], res["n_held_out_frames"]))
        if not np.isnan(res["median_survival_frames"]):
            surv_all.append(res["median_survival_frames"])

print("\n=== aggregate ===")
print(agg)
print(f"acceptance_rate {agg['n_accepted'] / max(agg['n_attempts'], 1):.2%}  "
      f"ambiguity_rate {agg['n_ambiguous'] / max(agg['n_accepted'], 1):.2%}")
if disp_all:
    w = np.array([n for _, n in disp_all]); v = np.array([d for d, _ in disp_all])
    print(f"displacement_error (frame-weighted mean): {np.average(v, weights=w):.2f} px")
if motion_all:
    w = np.array([n for _, n in motion_all]); v = np.array([d for d, _ in motion_all])
    print(f"motion_error (frame-weighted mean): {np.average(v, weights=w):.2f} px/frame")
if surv_all:
    print(f"median survival across episodes: {np.median(surv_all):.1f} frames")

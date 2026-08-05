"""Cross-check heading_identity_consistency's ACCEPTED pairings against GT --
purely a validation step (mirrors validate_heading_detect.py), not part of the
metric itself, which never touches GT. On real GT-rendered (VAE round-tripped)
frames, camera agent index IS a known boid index, so for every accepted (a, b)
pairing we can ask: does the dart we identified as "b" in a's view actually
imply agent b's true world position, frame by frame?

    uv run python validate_heading_identity.py
"""
import numpy as np, torch
from pathlib import Path
from modeling.configs import load_cfg
from train_flock_dit import build_decode_fn
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.eval import gt_project as gp, overlay as ov
from modeling.eval.boid_detect import detect_boids
from modeling.eval.heading_consistency import heading_identity_consistency, HALF

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
NEP, NLAT, NUM_BOIDS, CORRECT_PX = 5, 38, 100, 10.0

cfg = load_cfg("config/train_flockdit_latent_multi.yaml",
                [f"data.root={ROOT}", f"output_dir={OUTD}", "vae.checkpoint_path=pretrained/no_color/last.ckpt"])
decode_fn = build_decode_fn(cfg)
ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))

n_correct_frames = n_total_frames = 0
n_correct_pairs = n_total_pairs = 0     # a pairing counts "correct" if >=half its frames match
gt_errs = []

with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e)
        P = ep["frames"].shape[0]
        dets = []
        for j in range(P):
            frames = ov.to_uint8(decode_fn(ep["frames"][j, :NLAT]).detach().cpu())
            dets.append([detect_boids(frames[t]) for t in range(len(frames))])
        pos = gp.load_gt_positions(Path(cfg.data.root) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
        _, pairs = heading_identity_consistency(dets, return_pairs=True)

        for p in pairs:
            a, b, t0, surv_end = p["a"], p["b"], p["t0"], p["surv_end"]
            ta = p["ta"]
            n_correct_this, n_total_this = 0, 0
            for t in range(t0, min(surv_end, pos.shape[0])):
                implied_world = ta["pos"][t - ta["t0"]] - HALF + pos[t, a]
                err = float(np.linalg.norm(implied_world - pos[t, b]))
                gt_errs.append(err)
                n_total_this += 1
                if err <= CORRECT_PX:
                    n_correct_this += 1
            n_total_frames += n_total_this
            n_correct_frames += n_correct_this
            n_total_pairs += 1
            if n_total_this and n_correct_this / n_total_this >= 0.5:
                n_correct_pairs += 1
        print(f"ep{ep['episode_id']}: {len(pairs)} accepted pairings checked against GT", flush=True)

gt_errs = np.asarray(gt_errs)
print(f"\n=== validation vs GT (accepted pairings only) ===")
print(f"n_pairs {n_total_pairs}  n_frames {n_total_frames}")
print(f"frame-level precision (<= {CORRECT_PX}px of true GT boid): "
      f"{n_correct_frames / max(n_total_frames, 1):.2%}")
print(f"pair-level precision (majority of its frames correct): "
      f"{n_correct_pairs / max(n_total_pairs, 1):.2%}")
if len(gt_errs):
    q = np.percentile(gt_errs, [50, 75, 90, 99, 100])
    print(f"implied-vs-true world position error (px): median {q[0]:.1f}  p75 {q[1]:.1f}  "
          f"p90 {q[2]:.1f}  p99 {q[3]:.1f}  max {q[4]:.1f}")

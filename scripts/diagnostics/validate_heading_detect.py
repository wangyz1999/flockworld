"""Validate the boid-heading extractor (boid_detect._boid_heading) against
GT headings on real recorded (VAE round-tripped) frames -- NOT a model rollout,
so this isolates the extractor's own noise from any model drift. Every detected
detection is matched to its nearest GT-visible boid (mirrors scripts/diagnostics/diag_boid_shapes.py's
matching); among matched detections we compare extracted heading to that boid's true
heading (gt_project.load_gt_headings).

    uv run python -m scripts.diagnostics.validate_heading_detect
"""
import numpy as np, torch
from pathlib import Path
from modeling.configs import load_cfg
from modeling.models.decoding import build_decode_fn
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.eval import gt_project as gp, overlay as ov
from modeling.eval.boid_detect import detect_boids

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
NEP, NLAT, MATCH, NUM_BOIDS = 5, 24, 6.0, 100

cfg = load_cfg("config/train_flockdit_latent_multi.yaml",
                [f"data.root={ROOT}", f"output_dir={OUTD}", "vae.checkpoint_path=pretrained/no_color/last.ckpt"])
decode_fn = build_decode_fn(cfg)
ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))

errs = []       # angular error (deg), matched + headed detections only
n_detections = n_matched = n_headed = 0
with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e)
        pos = gp.load_gt_positions(Path(cfg.data.root) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
        head = gp.load_gt_headings(Path(cfg.data.root) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
        for j in range(ep["frames"].shape[0]):
            frames = ov.to_uint8(decode_fn(ep["frames"][j, :NLAT]).detach().cpu())   # frame t <-> pos[t]
            cam = ep["agent_indices"][j] - 1
            for t in range(min(len(frames), pos.shape[0])):
                gidx, gpx = gp.visible_boids(pos[t], cam)
                gpx = np.asarray(gpx, np.float32).reshape(-1, 2)
                det = detect_boids(frames[t])
                cents, headings = det["centroids"], det["heading"]
                n_detections += len(cents)
                for (u, v), h in zip(cents, headings):
                    if len(gpx) == 0:
                        continue
                    d = np.linalg.norm(gpx - np.array([u, v]), axis=1)
                    k = int(np.argmin(d))
                    if d[k] > MATCH:
                        continue
                    n_matched += 1
                    if np.isnan(h):
                        continue
                    n_headed += 1
                    true_h = head[t, gidx[k]]
                    err = np.degrees(abs(np.arctan2(np.sin(h - true_h), np.cos(h - true_h))))
                    errs.append(err)
        print(f"ep{ep['episode_id']}: matched {n_matched} headed {n_headed} (cumulative)", flush=True)

errs = np.asarray(errs)
print(f"\nn_detections {n_detections}  n_matched {n_matched}  n_headed(of matched) {n_headed} "
      f"({n_headed / max(n_matched, 1):.2%})")
if len(errs):
    q = np.percentile(errs, [50, 75, 90, 99, 100])
    print(f"angular error (deg): median {q[0]:.1f}  p75 {q[1]:.1f}  p90 {q[2]:.1f}  p99 {q[3]:.1f}  max {q[4]:.1f}")
    print(f"frac <5deg: {(errs < 5).mean():.2%}   frac <15deg: {(errs < 15).mean():.2%}   "
          f"frac <30deg: {(errs < 30).mean():.2%}   frac >90deg (likely nose/tail flip): {(errs > 90).mean():.2%}")
else:
    print("no matched+headed detections -- something upstream is broken")

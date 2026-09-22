"""Validate the corner detector on GT-decoded frames (repo root) -- step 3a gate.

For camera-agent frames where GT says the agent is at a corner, decode the REAL
latents and run wall_probe.detect_corner. Checks: does it fire, get the right
corner (TL/TR/BL/BR), and recover the agent's absolute position close to GT?
Must pass before trusting corner detection / position recovery on GENERATED frames.

    uv run python -m scripts.diagnostics.validate_corner_detect
"""
import numpy as np, torch
from pathlib import Path
from modeling.configs import load_cfg
from modeling.models.decoding import build_decode_fn
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.eval import gt_project as gp, overlay as ov, wall_probe as wp

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
N_EP, NUM_BOIDS, POS_TOL = 5, 100, 6.0

cfg = load_cfg("config/train_flockdit_latent_multi.yaml", [f"data.root={ROOT}", f"output_dir={OUTD}"])
device = "cuda" if torch.cuda.is_available() else "cpu"
decode_fn = build_decode_fn(cfg)
ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))

n_gt = n_det = n_type = 0
errs = []
with torch.no_grad():
    for e in range(min(N_EP, len(ds.episodes))):
        ep = ds.full_episode(e)
        pos = gp.load_gt_positions(Path(ROOT) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
        cam_idx = [ai - 1 for ai in ep["agent_indices"]]
        cam = pos[:, cam_idx, :]                                   # (T, P, 2)
        cf = wp.corner_flags(wp.wall_flags(cam))                   # (T, P, 4)
        for j in range(cam.shape[1]):
            at = cf[:, j].any(-1)
            if not at.any():
                continue
            frames = ov.to_uint8(decode_fn(ep["frames"][j]).detach().cpu())   # (T_pix, H, W, 3), frame t <-> cam[t]
            for t in np.flatnonzero(at):
                if t >= len(frames):
                    continue
                n_gt += 1
                gt_corner = wp.CORNERS[int(cf[t, j].argmax())]
                det = wp.detect_corner(frames[t])
                if det is None:
                    continue
                n_det += 1
                if det["corner"] == gt_corner:
                    n_type += 1
                    errs.append(float(np.linalg.norm(np.array(det["agent_xy"]) - cam[t, j])))

print(f"GT near-corner frames: {n_gt}")
print(f"  detected a corner:     {n_det/max(n_gt,1):.3f}  ({n_det})")
print(f"  correct corner type:   {n_type/max(n_det,1):.3f}  (of detected)")
if errs:
    e = np.array(errs)
    print(f"  position error (px):   median {np.median(e):.1f}  mean {e.mean():.1f}  "
          f"p90 {np.percentile(e,90):.1f}  frac<= {POS_TOL}px: {(e<=POS_TOL).mean():.3f}")
print("\ngate: want detection ~high, type ~1.0, position error small (few px) before trusting on generated frames.")

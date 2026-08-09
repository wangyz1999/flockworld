"""Quick threshold sweep for heading_identity_consistency on real GT (VAE
round-tripped) frames -- decodes/detects once per episode (the expensive VAE
part), then re-runs the cheap identity-resolution + GT-precision check for
several (heading_thresh_deg, max_dist, w_boot) configs, to see how precision
trades off against acceptance volume before spending rollout compute.

    uv run python tune_heading_identity.py
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

CONFIGS = [
    dict(heading_thresh_deg=30.0, max_dist=8.0, w_boot=8),    # baseline (already measured)
    dict(heading_thresh_deg=15.0, max_dist=8.0, w_boot=8),    # tighter heading gate
    dict(heading_thresh_deg=30.0, max_dist=4.0, w_boot=8),    # stricter reciprocity
    dict(heading_thresh_deg=15.0, max_dist=4.0, w_boot=8),    # both tighter
    dict(heading_thresh_deg=30.0, max_dist=8.0, w_boot=16),   # longer bootstrap window
    dict(heading_thresh_deg=15.0, max_dist=8.0, w_boot=16),   # tighter heading + longer window
]

cfg = load_cfg("config/train_flockdit_latent_multi.yaml",
                [f"data.root={ROOT}", f"output_dir={OUTD}", "vae.checkpoint_path=pretrained/no_color/last.ckpt"])
decode_fn = build_decode_fn(cfg)
ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))

episodes = []   # (episode_id, dets, pos)
with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e)
        P = ep["frames"].shape[0]
        dets = []
        for j in range(P):
            frames = ov.to_uint8(decode_fn(ep["frames"][j, :NLAT]).detach().cpu())
            dets.append([detect_boids(frames[t]) for t in range(len(frames))])
        pos = gp.load_gt_positions(Path(cfg.data.root) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
        episodes.append((ep["episode_id"], dets, pos))
        print(f"decoded+detected ep{ep['episode_id']}", flush=True)

print(f"\n{'config':<45} {'n_acc':>6} {'acc%':>6} {'amb%':>6} {'framePrec%':>10} {'pairPrec%':>9} {'survFr':>7}")
for c in CONFIGS:
    n_accepted_total = n_attempts_total = n_ambiguous_total = 0
    n_correct_frames = n_total_frames = 0
    n_correct_pairs = n_total_pairs = 0
    survs = []
    for eid, dets, pos in episodes:
        res, pairs = heading_identity_consistency(dets, return_pairs=True, **c)
        n_accepted_total += res["n_accepted"]; n_attempts_total += res["n_attempts"]
        n_ambiguous_total += res["n_ambiguous"]
        if not np.isnan(res["median_survival_frames"]):
            survs.append(res["median_survival_frames"])
        for p in pairs:
            a, b, t0, surv_end = p["a"], p["b"], p["t0"], p["surv_end"]
            ta = p["ta"]
            nc = nt = 0
            for t in range(t0, min(surv_end, pos.shape[0])):
                implied_world = ta["pos"][t - ta["t0"]] - HALF + pos[t, a]
                if float(np.linalg.norm(implied_world - pos[t, b])) <= CORRECT_PX:
                    nc += 1
                nt += 1
            n_total_frames += nt; n_correct_frames += nc
            n_total_pairs += 1
            if nt and nc / nt >= 0.5:
                n_correct_pairs += 1
    label = f"h={c['heading_thresh_deg']:.0f} d={c['max_dist']:.0f} w={c['w_boot']}"
    acc_pct = 100 * n_accepted_total / max(n_attempts_total, 1)
    amb_pct = 100 * n_ambiguous_total / max(n_accepted_total, 1)
    frame_prec = 100 * n_correct_frames / max(n_total_frames, 1)
    pair_prec = 100 * n_correct_pairs / max(n_total_pairs, 1)
    surv = np.median(survs) if survs else float("nan")
    print(f"{label:<45} {n_accepted_total:>6} {acc_pct:>6.2f} {amb_pct:>6.2f} "
          f"{frame_prec:>10.2f} {pair_prec:>9.2f} {surv:>7.1f}")

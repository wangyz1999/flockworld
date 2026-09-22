"""heading_identity_consistency on an actual model rollout vs the GT ceiling,
for Setup 1 (no color, no bg gradient). Mirrors modeling/cli/eval_flock_multi.py's
ceiling/model structure but for the GT-free heading-identity metric, which
pair_consistency.py can't compute here (no hue to key off).

    uv run python -m modeling.cli.eval_heading_identity --episodes 6 --seconds 10

Both columns run the SAME GT-free metric -- "ceiling" decodes the real
recorded latents (best case: VAE + detector + identity-resolution noise only),
"model" rolls out the trained multi-agent model. Neither uses GT for scoring;
GT enters nowhere in this script except to load the dataset/checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.eval import overlay as ov
from modeling.eval.boid_detect import detect_boids
from modeling.eval.heading_consistency import heading_identity_consistency
from modeling.eval.multi import _load_model, _rollout_model, _rollout_single
from modeling.models.decoding import build_decode_fn


def _decode_detect(lat, decode_fn):
    dets_all = []
    for j in range(lat.shape[0]):
        frames = ov.to_uint8(decode_fn(lat[j]).detach().cpu())
        dets_all.append([detect_boids(frames[t]) for t in range(len(frames))])
    return dets_all


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train_flockdit_latent_multi.yaml")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--baseline", choices=["single", "none"], default="single",
                    help="single = independent per-agent no-color model (the cross-agent-unaware floor).")
    ap.add_argument("--baseline-config", default="config/train_flockdit_latent.yaml")
    ap.add_argument("--baseline-output-dir", default="output/flock_dit_latent_10k")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decode_fn = build_decode_fn(cfg)
    ds = FlockingLatentMultiDataset(
        cache_dir=Path(cfg.data.root) / cfg.data.get("latent_cache_dir", "latent_cache"),
        split="val", val_fraction=cfg.data.val_fraction,
        num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
        random_clip=False, num_agents=int(cfg.data.num_agents))

    print("multi-agent model:")
    model = _load_model(cfg, cfg.output_dir, device)
    baseline_model = None
    if args.baseline == "single":
        bcfg = load_cfg(args.baseline_config, [f"output_dir={args.baseline_output_dir}"])
        print("single-agent baseline model (no-color, confirmed via rendered frame):")
        baseline_model = _load_model(bcfg, bcfg.output_dir, device)

    pix_frames = round(args.seconds * args.sim_fps)
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(cfg.data.num_future_frames)
    n_ep = min(args.episodes, len(ds.episodes))
    cols = ["ceiling", "model"] + (["baseline"] if args.baseline == "single" else [])
    R = {c: [] for c in cols}
    print(f"heading-identity metric: {args.seconds}s = {n_lat} latent frames, {n_ep} episodes")

    for e in range(n_ep):
        ep = ds.full_episode(e); ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        frames = ep["frames"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        lat = {
            "ceiling": ep["frames"][:, :T],
            "model": _rollout_model(model, frames, actions, ctx, T, wf, args.steps),
        }
        if args.baseline == "single":
            lat["baseline"] = _rollout_single(baseline_model, frames, actions, ctx, T, wf, args.steps)
        for c in cols:
            dets = _decode_detect(lat[c], decode_fn)
            R[c].append(heading_identity_consistency(dets))
        print(f"  ep{ep['episode_id']} ({e + 1}/{n_ep}): "
              + " ".join(f"{c} n_accepted={R[c][-1]['n_accepted']}" for c in cols), flush=True)

    def total(key):
        return {c: int(sum(d[key] for d in R[c])) for c in cols}

    def wmean(key, wkey):
        out = {}
        for c in cols:
            vals = np.array([d[key] for d in R[c]]); w = np.array([d[wkey] for d in R[c]])
            ok = ~np.isnan(vals) & (w > 0)
            out[c] = float(np.average(vals[ok], weights=w[ok])) if ok.any() else float("nan")
        return out

    n_att, n_acc, n_amb = total("n_attempts"), total("n_accepted"), total("n_ambiguous")
    print(f"\n{'metric':>26}" + "".join(f"{c:>12}" for c in cols))
    print(f"{'n_attempts':>26}" + "".join(f"{n_att[c]:>12}" for c in cols))
    print(f"{'n_accepted':>26}" + "".join(f"{n_acc[c]:>12}" for c in cols))
    print(f"{'acceptance_rate':>26}" + "".join(f"{n_acc[c] / max(n_att[c], 1):>12.4f}" for c in cols))
    print(f"{'n_ambiguous':>26}" + "".join(f"{n_amb[c]:>12}" for c in cols))
    print(f"{'ambiguity_rate':>26}" + "".join(f"{n_amb[c] / max(n_acc[c], 1):>12.4f}" for c in cols))

    surv = wmean("median_survival_frames", "n_accepted")
    disp = wmean("displacement_error", "n_held_out_frames")
    mot = wmean("motion_error", "n_held_out_frames")
    print(f"{'median_survival_frames':>26}" + "".join(f"{surv[c]:>12.2f}" for c in cols))
    print(f"{'displacement_error':>26}" + "".join(f"{disp[c]:>12.2f}" for c in cols))
    print(f"{'motion_error':>26}" + "".join(f"{mot[c]:>12.3f}" for c in cols))
    print(f"{'n_held_out_frames':>26}" + "".join(f"{total('n_held_out_frames')[c]:>12}" for c in cols))

    print("\nceiling = GT latents decoded (VAE + detector + identity-resolution noise only, best case)")
    print("model   = the trained multi-agent rollout")
    print("higher acceptance_rate / lower ambiguity_rate+displacement+motion = better; "
          "compare model to ceiling, not to zero.")


if __name__ == "__main__":
    main()

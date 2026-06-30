"""Multi-agent cross-view evaluation for FlockDiT (qualitative overlay).

Rolls out all P agents' views, decodes to pixels, overlays GT-projected boid
marks (red ring = the view's focal agent, green = every other boid), and tiles
the P views into one grid video. Two sources:

  --source gt    decode the REAL (GT) latents -> geometry/alignment sanity
                 (marks must sit on the darts; validates projection + alignment).
  --source pred  roll out the MODEL -> the actual prediction overlay
                 (do the model's darts land on the GT marks? do co-located
                 agents agree?).

Usage:
  uv run python eval_flock_multi.py --config config/train_flockdit_latent_multi.yaml \
    --source pred --episodes 3 --seconds 10 \
    data.root=data/recording/20260623_135735 \
    output_dir=output/flock_dit_multi_10kep_nocolor_nobg

(The quantitative consistency metrics build on this same rollout+decode and land
in a later step; this file is the qualitative read.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from modeling.configs import load_cfg
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import gt_project as gp, overlay as ov
from eval_flock_dit import find_best_checkpoint, build_model, write_mp4
from train_flock_dit import build_decode_fn

NUM_SIM_BOIDS = 100  # all boids are in the parquet; any can be a GT mark


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Multi-agent overlay eval (qualitative).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["gt", "pred"], default="pred")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50, help="Euler steps per window (pred only).")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    if not bool(cfg.data.get("latent", False)):
        raise SystemExit("eval_flock_multi requires latent mode (data.latent=true).")
    device = "cuda" if (torch.cuda.is_available() and str(cfg.device) != "cpu") else "cpu"
    decode_fn = build_decode_fn(cfg)

    ds = FlockingLatentMultiDataset(
        cache_dir=Path(cfg.data.root) / cfg.data.get("latent_cache_dir", "latent_cache"),
        split="val", val_fraction=cfg.data.val_fraction,
        num_context_frames=cfg.data.num_context_frames,
        num_future_frames=cfg.data.num_future_frames,
        random_clip=False, num_agents=int(cfg.data.num_agents),
    )

    model = None
    if args.source == "pred":
        ckpt, val = find_best_checkpoint(Path(cfg.output_dir) / "checkpoints")
        print(f"checkpoint: {ckpt} (val_loss={val:.5f})")
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        model.eval()

    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4               # latent frames for `seconds` (Wan 4x temporal)
    wf = int(cfg.data.num_future_frames)
    out = Path(cfg.output_dir) / "eval" / f"{args.source}_overlay_{args.seconds:g}s"
    out.mkdir(parents=True, exist_ok=True)
    n_ep = min(int(args.episodes), len(ds.episodes))
    print(f"{args.source} overlay: {args.seconds}s = {n_lat} latent frames, {n_ep} episodes -> {out}")

    for e in range(n_ep):
        ep = ds.full_episode(e)
        ctx = int(ep["context_len"])
        T = min(n_lat, ep["frames"].shape[1])
        if args.source == "gt":
            lat = ep["frames"][:, :T]                                   # (P, T, z, h, w)
        else:
            frames = ep["frames"].unsqueeze(0).to(device)
            actions = ep["actions"].unsqueeze(0).to(device)
            clip = fm.multi_autoregressive_rollout(
                model, frames[:, :, :ctx], actions, T, window_future=wf, num_steps=int(args.steps))
            lat = clip[0]                                               # (P, T, z, h, w)
        positions = gp.load_gt_positions(
            Path(cfg.data.root) / "state_action" / f"{ep['episode_id']}.parquet", NUM_SIM_BOIDS)
        views = []
        for j in range(lat.shape[0]):
            frames_u8 = ov.to_uint8(decode_fn(lat[j]).detach().cpu())   # (T_pix, H, W, 3)
            views.append(ov.draw_gt_marks(ov.upscale(frames_u8, int(args.upscale)),
                                          positions, ep["agent_indices"][j] - 1))
        write_mp4(ov.tile(views), str(out / f"ep{ep['episode_id']}_{args.source}.mp4"), int(args.sim_fps))
        print(f"  wrote ep{ep['episode_id']} ({e + 1}/{n_ep})")
    print("done ->", out)


if __name__ == "__main__":
    main()

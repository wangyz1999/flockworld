"""Overfit diagnostic: memorize a few clips, then watch the rollout.

Trains a fresh FlockDiT on a handful of fixed *training* clips, then runs the
Euler rollout on those SAME clips and saves GT-vs-prediction videos. This
cleanly separates two failure causes:

* overfit rollout is sharp & persistent  -> recipe (model/flow-matching/sampler)
  is sound; the dissolving on val is a data/generalization problem -> more data +
  longer training.
* even a memorized clip dissolves          -> a recipe bug (noise schedule,
  sampler, horizon, context handling) to fix first; more data won't help.

Usage:
    uv run python -m scripts.diagnostics.overfit_rollout --config config/train_flockdit.yaml \
        [--num-clips 4] [--train-steps 1000] [--sample-steps 50] [overrides...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from modeling.eval.common import build_model, write_mp4
from modeling import flow_matching as fm
from modeling.configs import load_cfg
from modeling.data.flocking_dit_dataset import FlockingDiTDataset
from modeling.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser(description="Overfit a few clips and watch the rollout.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-clips", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    seed_everything(int(cfg.seed))
    device = "cuda" if (torch.cuda.is_available() and str(cfg.device) != "cpu") else "cpu"

    ds = FlockingDiTDataset(
        root=cfg.data.root,
        split="train",
        val_fraction=cfg.data.val_fraction,
        num_context_frames=cfg.data.num_context_frames,
        num_future_frames=cfg.data.num_future_frames,
        frame_stride=cfg.data.frame_stride,
        image_size=cfg.data.image_size,
        action_features=list(cfg.data.action_features),
        partial_agent_indices=cfg.data.get("partial_agent_indices", None),
        random_clip=False,            # deterministic clips so we overfit a fixed set
        split_seed=cfg.seed,
    )
    n = min(int(args.num_clips), len(ds))
    frames = torch.stack([ds[i]["frames"] for i in range(n)]).to(device)    # (n,F,C,H,W)
    actions = torch.stack([ds[i]["actions"] for i in range(n)]).to(device)  # (n,F,A)
    ctx = int(ds[0]["context_len"])
    nf = frames.shape[1]
    print(f"overfitting {n} clips | frames={tuple(frames.shape)} ctx={ctx} device={device}")

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr))
    g = torch.Generator(device=device).manual_seed(int(cfg.seed))

    model.train()
    for step in range(1, int(args.train_steps) + 1):
        t = fm.sample_timesteps((n,), nf, ctx, device, generator=g)
        x_t, eps = fm.add_noise(frames, t)
        v = model(x_t, t, actions)
        loss = ((v[:, ctx:] - fm.velocity_target(eps, frames)[:, ctx:]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.5f}")

    # roll out on the memorized clips and save GT (top) vs prediction (bottom)
    clip = fm.euler_rollout(model, frames[:, :ctx], actions, nf - ctx, num_steps=int(args.sample_steps))
    roll = ((clip[:, ctx:] - frames[:, ctx:]) ** 2).mean().item()
    last = frames[:, ctx - 1:ctx].expand(-1, nf - ctx, -1, -1, -1)
    base = ((last - frames[:, ctx:]) ** 2).mean().item()

    out_dir = Path(cfg.output_dir) / "overfit_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(cfg.logger.get("video_fps", 8)) if cfg.get("logger") else 8
    for i in range(n):
        vid = torch.cat([frames[i], clip[i]], dim=2).clamp(-1, 1)
        vid = ((vid + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        write_mp4(vid, out_dir / f"overfit_{i:02d}.mp4", fps)

    print(f"\noverfit rollout MSE = {roll:.5f}   baseline = {base:.5f}   ({base / max(roll, 1e-9):.2f}x better)")
    print(f"videos: {out_dir}")
    print("-> Watch them: if memorized clips stay SHARP, the recipe is fine and you need more data.")
    print("   If they still dissolve, it's a recipe bug (sampler/schedule), not data.")


if __name__ == "__main__":
    main()

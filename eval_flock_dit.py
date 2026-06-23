"""Evaluate a trained FlockDiT via autoregressive rollouts on held-out episodes.

Loads the best (lowest val_loss) checkpoint, rolls out the future frames with the
Euler sampler given each episode's context frames + per-frame actions, saves
ground-truth-vs-prediction videos, and reports rollout MSE against a
repeat-last-context-frame baseline. Low flow-matching (training) loss is only a
proxy; this measures whether the multi-step generation is actually coherent.

Usage:
    uv run python eval_flock_dit.py --config config/train_flockdit.yaml \
        [--num-episodes 6] [--steps 50] [--checkpoint path/to.pt] [overrides...]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from modeling import flow_matching as fm
from modeling.configs import load_cfg
from modeling.data.flocking_dit_dataset import FlockingDiTDataset
from modeling.models.flock_dit import FlockDiT
from modeling.utils.seed import seed_everything


def find_best_checkpoint(ckpt_dir: Path) -> tuple[Path, float]:
    """Pick the checkpoint with the lowest stored val_loss (mmap = cheap scan)."""
    best, best_val = None, float("inf")
    for p in sorted(ckpt_dir.glob("epoch_*.pt")):
        meta = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        v = meta.get("val_loss")
        if v is not None and float(v) < best_val:
            best, best_val = p, float(v)
    if best is None:
        raise FileNotFoundError(f"No checkpoints with a val_loss found in {ckpt_dir}")
    return best, best_val


def build_model(cfg) -> FlockDiT:
    m = cfg.model
    return FlockDiT(
        in_channels=int(m.in_channels),
        out_channels=int(m.out_channels),
        dim=int(m.dim),
        depth=int(m.depth),
        heads=int(m.heads),
        ffn_dim=int(m.ffn_dim),
        patch=int(m.patch),
        patch_t=int(m.get("patch_t", 1)),
        action_dim=len(list(cfg.data.action_features)),
        num_agents=int(cfg.data.get("num_agents", 1)),
        local_attn_size=int(m.get("local_attn_size", -1)),
    )


def write_mp4(frames_thwc: np.ndarray, path: Path, fps: int) -> Path | None:
    """RGB (T,H,W,C) uint8 -> H.264 mp4 via ffmpeg/libx264 (browser-playable)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[warn] ffmpeg not on PATH; skipping", path)
        return None
    _, h, w, _ = frames_thwc.shape
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames_thwc:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait(timeout=60)
    return path if proc.returncode == 0 else None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Roll out and evaluate a trained FlockDiT.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Default: lowest-val-loss checkpoint in output_dir.")
    ap.add_argument("--num-episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50, help="Euler integration steps.")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    seed_everything(int(cfg.seed))
    device = "cuda" if (torch.cuda.is_available() and str(cfg.device) != "cpu") else "cpu"

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    if args.checkpoint:
        ckpt_path, best_val = Path(args.checkpoint), None
    else:
        ckpt_path, best_val = find_best_checkpoint(ckpt_dir)
    tag = f" (val_loss={best_val:.5f})" if best_val is not None else ""
    print(f"Loading checkpoint: {ckpt_path}{tag}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # latent mode: decode latents -> pixels for the saved videos (MSE stays in latent space)
    decode_fn = None
    if bool(cfg.data.get("latent", False)):
        from train_flock_dit import build_decode_fn
        decode_fn = build_decode_fn(cfg)

    ds = FlockingDiTDataset(
        root=cfg.data.root,
        split="val",
        val_fraction=cfg.data.val_fraction,
        num_context_frames=cfg.data.num_context_frames,
        num_future_frames=cfg.data.num_future_frames,
        frame_stride=cfg.data.frame_stride,
        image_size=cfg.data.image_size,
        action_features=list(cfg.data.action_features),
        partial_agent_indices=cfg.data.get("partial_agent_indices", None),
        random_clip=False,
        split_seed=cfg.seed,
    )
    n = min(int(args.num_episodes), len(ds))
    out_dir = Path(cfg.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("rollout_*.mp4"):  # clear prior eval's videos so runs don't pile up
        old.unlink()
    fps = int(cfg.logger.get("video_fps", 8)) if cfg.get("logger") else 8

    roll_mses, base_mses = [], []
    print(f"\n{'idx':>4} {'episode':>8} {'agent':>5} {'rollout_mse':>12} {'baseline_mse':>13}")
    print("-" * 48)
    for i in range(n):
        s = ds[i]
        frames = s["frames"].unsqueeze(0).to(device)      # (1,F,C,H,W) in [-1,1]
        actions = s["actions"].unsqueeze(0).to(device)    # (1,F,A)
        ctx = int(s["context_len"])
        nf = frames.shape[1]
        clip = fm.euler_rollout(model, frames[:, :ctx], actions, nf - ctx, num_steps=int(args.steps))

        roll = ((clip[:, ctx:] - frames[:, ctx:]) ** 2).mean().item()
        last = frames[:, ctx - 1:ctx].expand(-1, nf - ctx, -1, -1, -1)  # repeat last context frame
        base = ((last - frames[:, ctx:]) ** 2).mean().item()
        roll_mses.append(roll)
        base_mses.append(base)

        # GT (top) over prediction (bottom); decode latents -> pixels in latent mode
        gt, pred = frames[0], clip[0]
        if decode_fn is not None:
            gt, pred = decode_fn(gt), decode_fn(pred)
        vid = torch.cat([gt, pred], dim=2).clamp(-1, 1)
        vid = ((vid + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        write_mp4(vid, out_dir / f"rollout_{i:02d}_ep{s['episode_id']}_a{s['agent_index']}.mp4", fps)
        print(f"{i:>4} {s['episode_id']:>8} {s['agent_index']:>5} {roll:>12.5f} {base:>13.5f}")

    mr, mb = float(np.mean(roll_mses)), float(np.mean(base_mses))
    print("-" * 48)
    print(f"\nmean rollout MSE   = {mr:.5f}")
    print(f"mean baseline MSE  = {mb:.5f}   (repeat last context frame)")
    print(f"improvement        = {mb / max(mr, 1e-9):.2f}x better than baseline")
    print(f"videos saved to    : {out_dir}")


if __name__ == "__main__":
    main()

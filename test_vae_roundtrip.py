"""Isolated VAE encode/decode round-trip test -- independent of the streaming eval
pipeline. Reads pre-recorded per-agent videos straight off disk (decord, same as
precompute_latents.py), encodes with the frozen VAE, decodes back, writes the
reconstruction next to nothing else -- no SimClipGenerator, no StreamingLatentEncoder,
no eval harness at all. Point of this: isolate whether a color-identity flip (a real
agent's rendered hue changing mid-episode) originates in the VAE itself, independent
of any of the new streaming eval code.

Usage:
  uv run python test_vae_roundtrip.py
  uv run python test_vae_roundtrip.py --videos data/recording/20260702160453/video_a3/00000.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from modeling.models.frozen_vae import FrozenVAE
from eval_flock_dit import write_mp4

DEFAULT_VIDEOS = [
    "data/recording/20260702160453/video_a1/00000.mp4",
    "data/recording/20260702160453/video_a2/00000.mp4",
    "data/recording/20260702160453/video_a5/00000.mp4",
    "data/recording/20260702160453/video_a9/00000.mp4",
]
VAE_CHECKPOINT = "pretrained/color_agent/checkpoint2/vae-081-0.0042.ckpt"
OUT_DIR = "output/vae_roundtrip_test"


def load_clip(path: str, img_size=(128, 128)) -> torch.Tensor:
    """video -> (1,3,T,H,W) float32 in [-1,1], exactly like precompute_latents.py."""
    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    n = len(reader)
    frames = torch.from_numpy(reader.get_batch(list(range(n))).asnumpy())  # (T,H,W,3) uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0                   # (T,3,H,W) [0,1]
    if (frames.shape[-2], frames.shape[-1]) != tuple(img_size):
        frames = F.interpolate(frames, size=img_size, mode="bilinear", align_corners=False)
    pixels = (frames * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)            # (1,3,T,H,W) [-1,1]
    return pixels


def to_uint8(pixels_bthw: torch.Tensor) -> np.ndarray:
    """(3,T,H,W) in [-1,1] -> (T,H,W,3) uint8."""
    x = ((pixels_bthw.clamp(-1, 1) + 1) * 0.5 * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Isolated VAE encode/decode round-trip test.")
    ap.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    ap.add_argument("--vae", default=VAE_CHECKPOINT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = FrozenVAE(args.vae, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_path in args.videos:
        name = Path(video_path).parent.name + "_" + Path(video_path).stem  # e.g. video_a2_00000
        pixels = load_clip(video_path)
        with torch.no_grad():
            z = vae.encode(pixels)              # (1, z_dim, T_lat, h, w)
            recon = vae.decode(z)               # (1, 3, T, H, W)
        orig_u8 = to_uint8(pixels[0])
        recon_u8 = to_uint8(recon[0].cpu())
        write_mp4(orig_u8, str(out_dir / f"{name}_original.mp4"), args.fps)
        write_mp4(recon_u8, str(out_dir / f"{name}_vae_recon.mp4"), args.fps)
        print(f"{name}: {pixels.shape[2]} frames -> wrote original + vae_recon -> {out_dir}")

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()

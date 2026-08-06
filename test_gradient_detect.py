"""Check the gradient-background boid detector (boid_detect.detect_boids with
background_size set) on real sim frames and their VAE round-trip reconstruction.

Renders a short episode directly from the sim with background_gradient=true
(no disk dataset needed -- same SimClipGenerator streaming training uses), so
we get exact GT visible-boid counts/positions for free (gt_project.visible_boids).
Compares detected counts against GT for:
  * raw sim frames, no background subtraction   (sanity: does the gradient alone
    already trip v_thresh, even with a perfect renderer?)
  * raw sim frames, with background subtraction  (does subtraction hurt when
    there's nothing to correct for?)
  * VAE round-trip frames, no background subtraction (does gradient + recon
    noise blow up false positives?)
  * VAE round-trip frames, with background subtraction (the proposed fix)
Also writes overlay mp4s (GT marks + detected rings) for visual inspection.

Usage:
  uv run python test_gradient_detect.py
"""
from __future__ import annotations

import numpy as np
import torch

from modeling.data.streaming_vae_dataset import SimClipGenerator, load_sim_cfg_container
from modeling.models.frozen_vae import FrozenVAE
from modeling.eval import gt_project as gp
from modeling.eval import overlay as ov
from modeling.eval.boid_detect import detect_boids
from eval_flock_dit import write_mp4

VAE_CHECKPOINT = "pretrained/color_bg/vae-3343-0.0046.ckpt"
SIM_CONFIG = "config/data_recording.yaml"
SIM_OVERRIDES = ["rendering.color_mode=agent_id", "rendering.background_gradient=true"]
NUM_PARTIAL_AGENTS = 4
NUM_FRAMES = 89          # 1 + 4*22: exact VAE round-trip length (causal 4x temporal downsample)
WARMUP_STEPS = 60
SEED = 12345
OUT_DIR = "output/gradient_detect_test"
BACKGROUND_SIZES = [None, 21, 31, 41, 61]  # None = old behavior (no subtraction)


def to_pixels(clip_u8: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (1,3,T,H,W) float32 in [-1,1]."""
    x = torch.from_numpy(clip_u8).permute(0, 3, 1, 2).float() / 255.0
    return (x * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    gen = SimClipGenerator(
        load_sim_cfg_container(SIM_CONFIG, SIM_OVERRIDES),
        num_frames=NUM_FRAMES, frame_stride=1, num_partial_agents=NUM_PARTIAL_AGENTS,
        windows_per_episode=1, warmup_steps=WARMUP_STEPS, threads=2,
        return_actions=True, return_gt_positions=True,
    )
    clips, _actions, gt_positions = gen.episode(SEED)   # clips (P,T,H,W,3); gt (1,T,N,2)
    gt = gt_positions[0]                                # (T, N, 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = FrozenVAE(VAE_CHECKPOINT, device=device)

    print(f"{'agent':>5} {'src':>6} {'bg_size':>7} {'mean_det':>9} {'mean_gt':>8} "
          f"{'over_frac':>9} {'under_frac':>10}")

    for k in range(NUM_PARTIAL_AGENTS):
        raw = clips[k]                                        # (T,128,128,3) uint8
        with torch.no_grad():
            z = vae.encode(to_pixels(raw))
            recon = vae.decode(z)
        recon_u8 = ov.to_uint8(recon[0].permute(1, 0, 2, 3).cpu())  # (T,128,128,3) uint8

        best_overlay = None
        for src_name, frames in [("raw", raw), ("vae", recon_u8)]:
            n = len(frames)
            gt_counts = np.array([len(gp.visible_boids(gt[t], k)[0]) for t in range(n)])
            for bg_size in BACKGROUND_SIZES:
                dets = [detect_boids(frames[t], background_size=bg_size) for t in range(n)]
                counts = np.array([len(d["centroids"]) for d in dets])
                over = float(np.mean(counts > gt_counts))
                under = float(np.mean(counts < gt_counts))
                print(f"{k:>5} {src_name:>6} {str(bg_size):>7} {counts.mean():>9.2f} "
                      f"{gt_counts.mean():>8.2f} {over:>9.2%} {under:>10.2%}")
                if src_name == "vae" and bg_size == 41:
                    best_overlay = (frames, dets)

        # overlay: GT marks (red=focal/green=others) + detected rings (yellow), on VAE recon
        frames, dets = best_overlay
        marked = ov.draw_gt_marks(ov.upscale(frames, 4), gt, agent_world_idx=k)
        marked = ov.draw_detections(marked, [d["centroids"] for d in dets])
        write_mp4(marked, f"{OUT_DIR}/agent{k}_vae_bg41_overlay.mp4", 30)

    print(f"\nwrote overlays -> {OUT_DIR}")


if __name__ == "__main__":
    main()

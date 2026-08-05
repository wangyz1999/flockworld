"""Cross-experiment comparison table for the streaming FlockDiT runs.

Detection rate / position error / detections-per-frame (Tier A, GT-anchored),
one column per experiment, plus a shared ``ceiling`` (GT latents decoded, no
rollout) and ``floor`` (independent single-agent model, in-distribution)
reference column -- built on the same held-out episodes for every column, so
the comparison is apples-to-apples. Also writes overlay rollout videos
(GT-projected boid marks over the decoded rollout) for a few episodes per
experiment, reusing the same decoded frames the metrics pass already computed.

Experiments are looked up by name in EXPERIMENTS below (checkpoint dirs under
flockdit_streaming_20260715/, see jobs/flockdit_streaming_20260715/flockdit_streaming.job
for the authoritative run-key -> config mapping). The two-stage runs (exp04/07/08)
come from the 2026-07-27 re-run, SLURM job 10621962; the columns are their stage-2
(multi-agent) checkpoints. exp05 (density) has never been run.

CAVEAT on exp04: its stage 2 landed on a faster node (b02-06) and did 547k steps in
its 35h50m, vs ~298k for exp07/exp08 and ~394k for the from-scratch runs in their
47h40m. Equal wall clock, unequal optimizer steps -- read its column with that in mind.

Usage:
  uv run python compare_experiments.py --include exp01_baseline exp02_tiled
  uv run python compare_experiments.py --include exp01_baseline exp02_tiled \
      exp03_diffusion_forcing exp06_tiled_df --episodes 6 --seconds 10
  # all seven finished experiments in one table:
  uv run python compare_experiments.py --include exp01_baseline exp02_tiled \
      exp03_diffusion_forcing exp04_two_stage exp06_tiled_df exp07_tiled_df_two_stage \
      exp08_tiled_two_stage --episodes 6 --seconds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling.eval import consistency as cs, overlay as ov, pair_consistency as pc
from eval_flock_dit import write_mp4
from eval_flock_multi import (
    _decode_detect,
    _gt_positions,
    _load_model,
    _num_camera_agents,
    _rollout_model,
    _rollout_single,
)

RUN_ROOT = "flockdit_streaming_20260715"

# run-key -> (config, output_dir); output_dir holds checkpoints/epoch_*.pt.
EXPERIMENTS = {
    "exp01_baseline": (
        "config/train_flockdit_latent_multi_stream.yaml",
        f"{RUN_ROOT}/exp01_baseline-10290187_0",
    ),
    "exp02_tiled": (
        "config/train_flockdit_latent_multi_stream_tiled.yaml",
        f"{RUN_ROOT}/exp02_tiled-10290187_1",
    ),
    "exp03_diffusion_forcing": (
        "config/train_flockdit_latent_multi_stream_df.yaml",
        f"{RUN_ROOT}/exp03_diffusion_forcing-10290187_2",
    ),
    "exp06_tiled_df": (
        "config/train_flockdit_latent_multi_stream_tiled_df.yaml",
        f"{RUN_ROOT}/exp06_tiled_df-10290187_4",
    ),
    # Two-stage runs: single-agent pretrain -> multi-agent fine-tune. The output_dir
    # holds the STAGE-2 checkpoints; stage1/ alongside it holds the pretrain (which is
    # also what FLOOR_OUTPUT_DIR below points at).
    "exp04_two_stage": (
        "config/train_flockdit_latent_multi_stream_twostage.yaml",
        f"{RUN_ROOT}/exp04_two_stage-10621962_3",
    ),
    "exp07_tiled_df_two_stage": (
        "config/train_flockdit_latent_multi_stream_triple.yaml",
        f"{RUN_ROOT}/exp07_tiled_df_two_stage-10621962_5",
    ),
    "exp08_tiled_two_stage": (
        "config/train_flockdit_latent_multi_stream_twostage_tiled.yaml",
        f"{RUN_ROOT}/exp08_tiled_two_stage-10621962_6",
    ),
}

# The floor: an independent single-agent model, rolled out per-camera-agent on the
# SAME episodes (in-distribution -- the "why not just run 10 single-agent models" bar).
# exp04's stage-1 pretrain is the plain single-agent config (exp07's stage 1 adds
# diffusion forcing, so it is NOT interchangeable here). Same config and same epoch 84
# as the pre-re-run floor, so this column stays comparable to the 2026-07-29 table.
FLOOR_CONFIG = "config/train_flockdit_latent_single_stream.yaml"
FLOOR_OUTPUT_DIR = f"{RUN_ROOT}/exp04_two_stage-10621962_3/stage1"

TIER_A_KEYS = ["detection_rate", "position_error", "mean_detected_per_frame"]
TIER_A_LABELS = {
    "detection_rate": "Detection rate (up)",
    "position_error": "Position error px (down)",
    "mean_detected_per_frame": "Detections / frame",
    "mean_gt_visible_per_frame": "GT boids visible / frame (volume)",
}

# Cross-view consistency (GT-FREE: do the agents agree with EACH OTHER, not with
# reality -- see modeling/eval/pair_consistency.py). Different question from Tier A.
CONSISTENCY_KEYS = ["reciprocity_rate", "displacement_error", "motion_error", "white_correspondence"]
PIXEL_KEYS = ["psnr", "ssim"]
CONSISTENCY_LABELS = {
    "reciprocity_rate": "Reciprocity (up)",
    "displacement_error": "Displacement error px (down)",
    "motion_error": "Motion error px/frame (down)",
    "white_correspondence": "White boid match (up)",
    "psnr": "PSNR dB (up)",
    "ssim": "SSIM (up)",
    "sightings_per_frame": "Sightings / pair-frame (volume)",
}


@torch.no_grad()
def _all_metrics(lat, decode_fn, pos, cam_idx, n_cam):
    frames, dets = _decode_detect(lat, decode_fn)
    cents = [[d["centroids"] for d in ag] for ag in dets]
    tier_a = cs.tier_a(cents, pos, cam_idx)
    pair = pc.pair_consistency(dets, cam_idx, n_cam)
    pix = pc.pixel_consistency(frames, dets, cam_idx, n_cam)
    pix.pop("events", None)
    return tier_a, pair, pix, frames


def _write_videos(name, out_dir, episodes, frames_per_episode, positions_per_episode,
                   agent_indices, sim_fps, upscale, grid_cols):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ep_id, frames_all, positions in zip(episodes, frames_per_episode, positions_per_episode):
        views = [
            ov.draw_gt_marks(ov.upscale(frames_all[j], upscale), positions, agent_indices[j] - 1)
            for j in range(len(frames_all))
        ]
        write_mp4(ov.tile(views, cols=grid_cols), str(out_dir / f"{name}_ep{ep_id}.mp4"), sim_fps)
    print(f"  wrote {len(episodes)} video(s) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Cross-experiment Tier A comparison table.")
    ap.add_argument("--include", nargs="+", required=True, choices=list(EXPERIMENTS),
                    help="Experiment keys to add as columns, in order.")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--no-floor", action="store_true", help="skip the single-agent floor column")
    ap.add_argument("--video-episodes", type=int, default=2,
                    help="how many of the --episodes to also render as overlay mp4s (0 disables).")
    ap.add_argument("--video-dir", default=f"{RUN_ROOT}/eval_videos")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--grid-cols", type=int, default=5,
                    help="columns in the tiled POV grid (10 agents -> 5 gives a clean 5x2; "
                         "0 falls back to near-square packing, which leaves a 2-cell hole).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # One shared streaming eval dataset (data.streaming.* is identical across the
    # multi configs) -> every column below sees the SAME held-out episodes and GT.
    # ds.full_episode() returns UN-normalized latents; each column below normalizes
    # with ITS OWN exactly-replayed training-time stats (fit_train_stats), since
    # streaming checkpoints don't persist their own stats and a naive shared refit
    # measured ~25% off -- see modeling/eval/streaming_eval_dataset.py's docstring.
    from modeling.eval.streaming_eval_dataset import (
        StreamingMultiEvalDataset, fit_train_stats, make_decode_fn, raw_decode_fn, normalize_frames,
    )
    from modeling.models.frozen_vae import FrozenVAE

    first_cfg = load_cfg(EXPERIMENTS[args.include[0]][0], [])
    vae = FrozenVAE(str(first_cfg.vae.checkpoint_path), device=device)
    ds = StreamingMultiEvalDataset(
        first_cfg, vae, device, num_episodes=int(args.episodes), seconds=float(args.seconds),
        sim_fps=int(args.sim_fps), eval_seed=int(args.eval_seed),
    )

    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(first_cfg.data.num_future_frames)
    n_ep = min(int(args.episodes), len(ds.episodes))
    n_vid = min(int(args.video_episodes), n_ep)
    print(f"{n_ep} held-out episodes, {args.seconds}s ({n_lat} latent frames) each, "
          f"eval_seed={args.eval_seed}")

    # Per-episode fixed context: ctx, T, cam_idx, GT positions -- shared by every column.
    n_cam = _num_camera_agents(first_cfg, first_cfg.data.num_agents)
    episode_ctx = []
    for e in range(n_ep):
        ep = ds.full_episode(e)
        ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        cam_idx = [ai - 1 for ai in ep["agent_indices"]]
        pos = _gt_positions(first_cfg, ds, ep["episode_id"])
        episode_ctx.append((ep, ctx, T, cam_idx, pos))

    columns_a: dict[str, list[dict]] = {}     # Tier A: per-view fidelity vs GT
    columns_b: dict[str, list[dict]] = {}     # cross-view consistency, GT-free
    columns_c: dict[str, list[dict]] = {}     # dense pixel similarity (PSNR/SSIM), GT-free
    column_frames: dict[str, list] = {}       # name -> list of per-episode frames_all (video episodes only)

    def run_column(name, model_fn, decode_fn, norm):
        """model_fn(frames_in, actions_in, ctx, T, wf, steps) -> lat; None for ceiling.
        norm: this column's own (mean, std) to normalize ep["frames"] before use, or
        None for ceiling (raw latents, decoded with a matching raw decode_fn)."""
        res_a, res_b, res_c, frames_kept = [], [], [], []
        for i, (ep, ctx, T, cam_idx, pos) in enumerate(episode_ctx):
            raw = ep["frames"]
            frames_for_use = raw if norm is None else normalize_frames(raw, *norm)
            if model_fn is None:
                lat = frames_for_use[:, :T]
            else:
                frames_in = frames_for_use.unsqueeze(0).to(device)
                actions_in = ep["actions"].unsqueeze(0).to(device)
                lat = model_fn(frames_in, actions_in, ctx, T, wf, int(args.steps))
            tier_a, pair, pix, frames_all = _all_metrics(lat, decode_fn, pos, cam_idx, n_cam)
            res_a.append(tier_a); res_b.append(pair); res_c.append(pix)
            if i < n_vid:
                frames_kept.append(frames_all)
            print(f"  [{name}] ep{ep['episode_id']} ({i + 1}/{n_ep})", flush=True)
        columns_a[name] = res_a
        columns_b[name] = res_b
        columns_c[name] = res_c
        column_frames[name] = frames_kept

    print("\n=== ceiling (GT latents decoded, no rollout) ===")
    run_column("ceiling", None, raw_decode_fn(vae, device), None)

    if not args.no_floor:
        print(f"\n=== floor: single-agent model ({FLOOR_OUTPUT_DIR}) ===")
        bcfg = load_cfg(FLOOR_CONFIG, [f"output_dir={FLOOR_OUTPUT_DIR}"])
        floor_model = _load_model(bcfg, bcfg.output_dir, device)
        floor_mean, floor_std = fit_train_stats(bcfg, vae, device)
        run_column("floor", lambda f, a, ctx, T, wf, steps: _rollout_single(
            floor_model, f, a, ctx, T, wf, steps),
            make_decode_fn(vae, device, floor_mean, floor_std), (floor_mean, floor_std))
        del floor_model
        torch.cuda.empty_cache()

    for key in args.include:
        cfg_path, out_dir = EXPERIMENTS[key]
        print(f"\n=== {key} ({out_dir}) ===")
        ecfg = load_cfg(cfg_path, [f"output_dir={out_dir}"])
        model = _load_model(ecfg, ecfg.output_dir, device)
        mean_e, std_e = fit_train_stats(ecfg, vae, device)
        run_column(key, lambda f, a, ctx, T, wf, steps, m=model: _rollout_model(
            m, f, a, ctx, T, wf, steps),
            make_decode_fn(vae, device, mean_e, std_e), (mean_e, std_e))
        del model
        torch.cuda.empty_cache()

    cols = (["ceiling"] + ([] if args.no_floor else ["floor"]) + list(args.include))

    def mean(acc, key):
        vals = [d[key] for d in acc if not (isinstance(d[key], float) and d[key] != d[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n=== Tier A -- per-view fidelity vs GT ({n_ep} episodes x {args.seconds}s, steps={args.steps}) ===")
    print(f"{'metric':>34}" + "".join(f"{c:>16}" for c in cols))
    for k in TIER_A_KEYS + ["mean_gt_visible_per_frame"]:
        print(f"{TIER_A_LABELS[k]:>34}" + "".join(f"{mean(columns_a[c], k):>16.3f}" for c in cols))
    print("\ndetection_rate: higher=better | position_error: lower=better (px) | "
          "detections/frame is volume, not accuracy on its own.")
    print("Compare rates at similar 'GT boids visible / frame' -- a near-empty rollout trivially scores high.")

    print(f"\n=== Cross-view consistency -- GT-FREE, agents vs EACH OTHER, not vs reality "
          f"({n_ep} episodes x {args.seconds}s) ===")
    print(f"{'metric':>34}" + "".join(f"{c:>16}" for c in cols))
    for k in CONSISTENCY_KEYS:
        print(f"{CONSISTENCY_LABELS[k]:>34}" + "".join(f"{mean(columns_b[c], k):>16.3f}" for c in cols))
    for k in PIXEL_KEYS:
        print(f"{CONSISTENCY_LABELS[k]:>34}" + "".join(f"{mean(columns_c[c], k):>16.3f}" for c in cols))
    print(f"{CONSISTENCY_LABELS['sightings_per_frame']:>34}"
          + "".join(f"{mean(columns_b[c], 'sightings_per_frame'):>16.4f}" for c in cols))
    print("\nreciprocity/white_correspondence/psnr/ssim: higher=better | displacement/motion: lower=better (px).")
    print("This is a DIFFERENT question from Tier A: two views can agree with each other while both being "
          "wrong about reality (or vice versa). Read the rates against 'sightings / pair-frame' -- a near-empty "
          "rollout trivially agrees with itself.")

    def total(acc, key):
        return int(sum(d[key] for d in acc))

    print("\nvolume -- RAW EVENT COUNTS each row above is averaged over (differs per column; a rate or mean "
          "computed over a handful of events is not comparable to one computed over hundreds):")
    print(f"{'':>34}" + "".join(f"{c:>16}" for c in cols))
    for acc, key, lab in [
        (columns_b, "n_sightings", "n_sightings"),
        (columns_b, "n_reciprocal", "n_reciprocal (recip./disp./motion.)"),
        (columns_b, "n_matched_white", "n_matched_white"),
        (columns_b, "n_overlap_white", "n_overlap_white (white_match denom)"),
        (columns_c, "n_pixel_events", "n_pixel_events (psnr/ssim)"),
    ]:
        print(f"{lab:>34}" + "".join(f"{total(acc[c], key):>16}" for c in cols))

    if n_vid > 0:
        print(f"\nwriting {n_vid} overlay video(s) per column -> {args.video_dir}")
        for name in cols:
            episodes = [episode_ctx[i][0]["episode_id"] for i in range(n_vid)]
            positions = [episode_ctx[i][4] for i in range(n_vid)]
            agent_indices = episode_ctx[0][0]["agent_indices"]
            _write_videos(name, args.video_dir, episodes, column_frames[name], positions,
                          agent_indices, int(args.sim_fps), int(args.upscale), int(args.grid_cols))


if __name__ == "__main__":
    main()

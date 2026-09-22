"""Multi-agent cross-view evaluation for FlockDiT.

Two modes (``--mode``):

* **overlay** (qualitative): roll out all P agents' views, decode, overlay
  GT-projected boid marks (red ring = focal agent, green = other boids), tile
  into one grid video. Sources: ``--source gt`` (decode REAL latents -> geometry
  sanity) or ``--source pred`` (roll out the model -> prediction overlay).

* **metrics** (quantitative): compute Tier A (per-view fidelity vs GT) and the
  GT-free cross-view consistency (``pair_consistency``) for the sources and print
  a bracketed table:
    - **ceiling**  = GT latents decoded (best achievable; VAE + detection noise).
    - **model**    = the multi-agent rollout (P agents together, cross-attention on).
    - **baseline** (``--baseline``): the consistency FLOOR.
        single = each agent rolled out by the INDEPENDENT single-agent model
                 (in-distribution, the "why not just run 10 single-agent models" floor);
        none   = skip.

  Dart color identity is smoothed along tracks by default (the VAE flashes colors
  for stretches of frames, which a per-frame hue read turns into lost sightings);
  ``--no-hue-smooth`` restores the per-frame read, ``--hue-smooth-window`` /
  ``--hue-link-dist`` tune it. See ``pair_consistency.smooth_identities``.

Usage:
  uv run python -m modeling.cli.eval_flock_multi --config config/train_flockdit_latent_multi.yaml \
    --mode metrics --episodes 6 --seconds 10 --baseline single \
    --baseline-config config/train_flockdit_latent.yaml \
    --baseline-output-dir output/flock_dit_latent_10k \
    data.root=data/recording/20260623_135735 \
    output_dir=output/flock_dit_multi_10kep_nocolor_nobg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from modeling.configs import load_cfg
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.models.decoding import build_decode_fn
from modeling.eval.multi import (
    _cfg_has_gradient,
    _load_model,
    run_metrics,
    run_overlay,
    write_metrics_csv,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Multi-agent eval: qualitative overlay or quantitative metrics.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["overlay", "metrics"], default="overlay")
    ap.add_argument("--source", choices=["gt", "pred"], default="pred", help="overlay mode only.")
    ap.add_argument("--baseline", choices=["single", "none"], default="single",
                    help="metrics floor: single=independent single-agent model (clean); none=skip.")
    ap.add_argument("--baseline-config", default=None,
                    help="defaults to the single-agent disk config, or the streaming single-agent "
                         "config when --config is a streaming config.")
    ap.add_argument("--baseline-output-dir", default=None,
                    help="output_dir of the single-agent model (for --baseline single).")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50, help="Euler steps per window.")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=0,
                    help="streaming only: base seed for the held-out eval episodes.")
    ap.add_argument("--background-size", type=int, default=None,
                    help="boid-detector background-subtraction kernel, gradient-background "
                         "setups only (see boid_detect.detect_boids). Auto-defaults to 31 when "
                         "the config's sim_overrides set rendering.background_gradient=true; "
                         "pass explicitly to override, or 0 to force it off.")
    ap.add_argument("--hue-smooth-window", type=int, default=0,
                    help="metrics mode: temporal color-identity smoothing width, in DECODED "
                         "frames (30 fps). 0 (default) = one identity vote per track, which is "
                         "what survives a VAE color flash lasting seconds; N>=3 = centered "
                         "N-frame rolling vote, which only outvotes flashes shorter than N/2.")
    ap.add_argument("--hue-link-dist", type=float, default=8.0,
                    help="metrics mode: max per-frame centroid motion (px) for the identity "
                         "tracker to link two detections. Default 8 = the sim's worst-case "
                         "boid-vs-crop relative motion (boids cap at 4 px/tick, crop rides the "
                         "camera agent). Lower = more broken tracks = less smoothing.")
    ap.add_argument("--no-hue-smooth", action="store_true",
                    help="metrics mode: disable temporal identity smoothing and read each boid's "
                         "color per frame (the behavior before smoothing was added).")
    ap.add_argument("--save-json", default=None,
                    help="metrics mode: write the structured results table to this JSON path.")
    ap.add_argument("--save-csv", default=None,
                    help="metrics mode: write per-episode (not just averaged) records to this CSV "
                         "path, one row per (column, episode, metric) -- so future plotting/stats "
                         "don't require a rerun.")
    ap.add_argument("--tag", default=None,
                    help="experiment name recorded in --save-json output; defaults to the "
                         "output_dir's basename.")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    if args.background_size is None and _cfg_has_gradient(cfg):
        args.background_size = 31
        print("[gradient background detected] background_size defaulting to 31 "
              "(pass --background-size to override, or --background-size 0 to force off)")
    if args.background_size == 0:
        args.background_size = None
    streaming = bool(OmegaConf.select(cfg, "data.streaming.enabled", default=False))
    if not streaming and not bool(cfg.data.get("latent", False)):
        raise SystemExit("eval_flock_multi requires latent mode (data.latent=true).")
    device = "cuda" if (torch.cuda.is_available() and str(cfg.device) != "cpu") else "cpu"

    if streaming:
        from modeling.eval.streaming_eval_dataset import (
            StreamingMultiEvalDataset, fit_train_stats, make_decode_fn, normalize_frames,
        )
        from modeling.models.frozen_vae import FrozenVAE
        vae = FrozenVAE(str(cfg.vae.checkpoint_path), device=device)
        ds = StreamingMultiEvalDataset(
            cfg, vae, device, num_episodes=int(args.episodes), seconds=float(args.seconds),
            sim_fps=int(args.sim_fps), eval_seed=int(args.eval_seed),
        )
        # ds.full_episode returns UN-normalized latents; normalize every episode's
        # frames to THIS config's own exactly-replayed training-time stats (see
        # streaming_eval_dataset.py docstring -- a naive eval-time refit measured
        # ~25% off, so this replays the real calibration instead of approximating it).
        mean, std = fit_train_stats(cfg, vae, device)
        decode_fn = make_decode_fn(vae, device, mean, std)
        for item in ds._items:
            item["frames"] = normalize_frames(item["frames"], mean, std)
        if args.baseline_config is None:
            args.baseline_config = "config/train_flockdit_latent_single_stream.yaml"
        if args.baseline == "single":
            print("[warn] streaming --baseline single: the baseline model is fed frames "
                  "normalized with the PRIMARY config's stats, not its own true training "
                  "stats (a documented approximation here -- see modeling/cli/compare_experiments.py "
                  "for a per-model-exact comparison).")
    else:
        decode_fn = build_decode_fn(cfg)
        ds = FlockingLatentMultiDataset(
            cache_dir=Path(cfg.data.root) / cfg.data.get("latent_cache_dir", "latent_cache"),
            split="val", val_fraction=cfg.data.val_fraction,
            num_context_frames=cfg.data.num_context_frames,
            num_future_frames=cfg.data.num_future_frames,
            random_clip=False, num_agents=int(cfg.data.num_agents),
        )
        if args.baseline_config is None:
            args.baseline_config = "config/train_flockdit_latent.yaml"

    model = baseline_model = None
    model_ckpt = baseline_ckpt = None
    if args.mode == "metrics" or args.source == "pred":
        print("multi-agent model:")
        model, ckpt_path, val_loss = _load_model(cfg, cfg.output_dir, device, return_info=True)
        model_ckpt = {"path": ckpt_path, "val_loss": val_loss}
    if args.mode == "metrics" and args.baseline == "single":
        bov = [f"output_dir={args.baseline_output_dir}"] if args.baseline_output_dir else []
        bcfg = load_cfg(args.baseline_config, bov)
        print("single-agent baseline model:")
        baseline_model, bckpt_path, bval_loss = _load_model(bcfg, bcfg.output_dir, device, return_info=True)
        baseline_ckpt = {"path": bckpt_path, "val_loss": bval_loss}

    if args.mode == "metrics":
        results = run_metrics(cfg, model, baseline_model, ds, decode_fn, device, args)
        if args.save_json:
            import json
            payload = {
                "experiment": args.tag or Path(cfg.output_dir).name,
                "config": args.config,
                "overrides": args.overrides,
                "output_dir": str(cfg.output_dir),
                "streaming": streaming,
                "num_agents": int(cfg.data.num_agents),
                "background_size": args.background_size,
                "hue_smooth": (None if args.no_hue_smooth else
                               {"window": args.hue_smooth_window, "link_dist": args.hue_link_dist}),
                "eval_args": {
                    "episodes": args.episodes, "seconds": args.seconds, "sim_fps": args.sim_fps,
                    "steps": args.steps, "eval_seed": args.eval_seed, "baseline": args.baseline,
                },
                "model_checkpoint": model_ckpt,
                "baseline_config": args.baseline_config if args.baseline == "single" else None,
                "baseline_output_dir": args.baseline_output_dir if args.baseline == "single" else None,
                "baseline_checkpoint": baseline_ckpt,
                **results,
            }
            out_path = Path(args.save_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"\nwrote metrics JSON -> {out_path}")
        if args.save_csv:
            write_metrics_csv(args.save_csv, args.tag or Path(cfg.output_dir).name, results)
    else:
        run_overlay(cfg, model, ds, decode_fn, device, args)


if __name__ == "__main__":
    main()

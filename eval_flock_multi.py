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

Usage:
  uv run python eval_flock_multi.py --config config/train_flockdit_latent_multi.yaml \
    --mode metrics --episodes 6 --seconds 10 --baseline single \
    --baseline-config config/train_flockdit_latent.yaml \
    --baseline-output-dir output/flock_dit_latent_10k \
    data.root=data/recording/20260623_135735 \
    output_dir=output/flock_dit_multi_10kep_nocolor_nobg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from modeling.configs import load_cfg
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import gt_project as gp, overlay as ov, consistency as cs, pair_consistency as pc
from modeling.eval.boid_detect import detect_boids
from eval_flock_dit import find_best_checkpoint, build_model, write_mp4
from train_flock_dit import build_decode_fn

NUM_SIM_BOIDS = 100  # all boids are in the parquet; any can be a GT mark


def _gt_positions(cfg, ds, episode_id):
    if hasattr(ds, "gt_positions"):          # streaming: positions come straight from the sim
        return ds.gt_positions(episode_id)
    return gp.load_gt_positions(
        Path(cfg.data.root) / "state_action" / f"{episode_id}.parquet", NUM_SIM_BOIDS)


def _num_camera_agents(cfg, default: int) -> int:
    """Hue period for identity colors = the camera-agent count at generation time.

    Read from the dataset's saved ``settings.yaml`` (colored agent k -> hue
    k/n_cam), falling back to ``default`` (the rolled-out agent count).
    Streaming configs have no ``data.root`` at all -- always use ``default``.
    """
    from omegaconf import OmegaConf
    root = cfg.data.get("root", None)
    if root is None:
        return int(default)
    p = Path(root) / "settings.yaml"
    if p.exists():
        s = OmegaConf.load(p)
        for key in ("collection.partial_agents", "partial_agents", "num_camera_agents"):
            v = OmegaConf.select(s, key)
            if v is not None:
                return int(v)
    return int(default)


def _load_model(cfg, output_dir, device, return_info=False):
    ckpt, val = find_best_checkpoint(Path(output_dir) / "checkpoints")
    print(f"checkpoint: {ckpt} (val_loss={val:.5f})")
    m = build_model(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    m = m.eval()
    return (m, str(ckpt), val) if return_info else m


def _decode_detect(lat, decode_fn, background_size=None):
    """lat (P, T, z, h, w) -> (frames[agent] (Tp,H,W,3) uint8, dets[agent][frame] dict).

    Decode each agent's latents ONCE; keep the pixel frames (for pixel_consistency)
    and run the dart detector on them (centroids+hue for tier_a / pair_consistency).
    ``background_size``: gradient-background setups only -- see boid_detect.detect_boids.
    """
    frames_all, dets_all = [], []
    for j in range(lat.shape[0]):
        frames = ov.to_uint8(decode_fn(lat[j]).detach().cpu())
        frames_all.append(frames)
        dets_all.append([detect_boids(frames[t], background_size=background_size)
                          for t in range(len(frames))])
    return frames_all, dets_all


def _cfg_has_gradient(cfg) -> bool:
    """True if this config's sim overrides turn on the gradient background."""
    overrides = OmegaConf.select(cfg, "data.streaming.sim_overrides", default=None) or []
    return any(str(o).strip() == "rendering.background_gradient=true" for o in overrides)


def _rollout_model(model, frames, actions, ctx, T, wf, steps):
    """Full P-agent rollout (cross-attention on) -> (P, T, z, h, w)."""
    return fm.multi_autoregressive_rollout(
        model, frames[:, :, :ctx], actions, T, window_future=wf, num_steps=steps)[0]


def _rollout_single(single_model, frames, actions, ctx, T, wf, steps):
    """Each agent rolled out by the INDEPENDENT single-agent model (5-D, in-distribution)."""
    solos = []
    for k in range(frames.shape[1]):
        ctx_k = frames[0, k, :ctx].unsqueeze(0)        # (1, ctx, z, h, w)
        act_k = actions[0, k].unsqueeze(0)             # (1, T, A)
        ck = fm.autoregressive_rollout(single_model, ctx_k, act_k, T, window_future=wf, num_steps=steps)
        solos.append(ck[0])                            # (T, z, h, w)
    return torch.stack(solos, 0)


@torch.no_grad()
def run_metrics(cfg, model, baseline_model, ds, decode_fn, device, args):
    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(cfg.data.num_future_frames)
    steps = int(args.steps)
    n_ep = min(int(args.episodes), len(ds.episodes))
    cols = ["ceiling", "model"] + ([] if args.baseline == "none" else ["baseline"])
    A = {s: [] for s in cols}; B = {s: [] for s in cols}; C = {s: [] for s in cols}
    ev = {s: [] for s in cols}                             # (overlap_frac, psnr, ssim) per pixel event
    n_cam = _num_camera_agents(cfg, cfg.data.num_agents)   # hue period for identity colors
    print(f"metrics: {args.seconds}s = {n_lat} latent frames, {n_ep} episodes "
          f"(baseline={args.baseline}, n_cam={n_cam})")
    for e in range(n_ep):
        ep = ds.full_episode(e); ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        cam_idx = [ai - 1 for ai in ep["agent_indices"]]
        pos = _gt_positions(cfg, ds, ep["episode_id"])
        frames = ep["frames"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        lat = {
            "ceiling": ep["frames"][:, :T],
            "model": _rollout_model(model, frames, actions, ctx, T, wf, steps),
        }
        if args.baseline == "single":
            lat["baseline"] = _rollout_single(baseline_model, frames, actions, ctx, T, wf, steps)
        for s in cols:
            frames, dets = _decode_detect(lat[s], decode_fn, background_size=args.background_size)
            cents = [[d["centroids"] for d in ag] for ag in dets]   # tier_a needs centroids only
            A[s].append(cs.tier_a(cents, pos, cam_idx))
            B[s].append(pc.pair_consistency(dets, cam_idx, n_cam))   # GT-free (no pos)
            pm = pc.pixel_consistency(frames, dets, cam_idx, n_cam)  # GT-free dense (warped overlap)
            ev[s].extend(pm.pop("events")); C[s].append(pm)
        print(f"  ep{ep['episode_id']} ({e + 1}/{n_ep})", flush=True)

    def mean(acc, key):
        vals = [d[key] for d in acc if not (isinstance(d[key], float) and d[key] != d[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def table(title, acc, keys):
        print(f"\n=== {title} ===")
        print(f"{'metric':>26}" + "".join(f"{c:>10}" for c in cols))
        for k in keys:
            print(f"{k:>26}" + "".join(f"{mean(acc[c], k):>10.3f}" for c in cols))

    table("Tier A — per-view fidelity vs GT", A,
          ["detection_rate", "position_error", "mean_detected_per_frame", "mean_gt_visible_per_frame"])
    table("Cross-view consistency — GT-FREE (ceiling=best | baseline=floor)", B,
          ["reciprocity_rate", "displacement_error", "motion_error",
           "white_correspondence", "white_count_error", "sightings_per_frame"])
    table("Pixel similarity — warped overlap, GT-FREE (higher=better)", C,
          ["psnr", "ssim", "mean_overlap_frac"])

    def total(acc, key):
        return int(sum(d[key] for d in acc))

    # A self-triggered metric: the rates above are only meaningful against the VOLUME the model
    # actually rendered (a near-empty world trivially "agrees"). These counts are the sample sizes.
    print("\nvolume (self-generated, differs per column — READ THE RATES AGAINST THIS):")
    print(f"{'':>26}" + "".join(f"{c:>10}" for c in cols))
    for acc, key, lab in [(B, "n_sightings", "sightings"), (B, "n_reciprocal", "reciprocal"),
                          (B, "n_matched_white", "white_match"), (B, "n_overlap_white", "white_chances"),
                          (C, "n_pixel_events", "pixel_events")]:
        print(f"{lab:>26}" + "".join(f"{total(acc[c], key):>10}" for c in cols))

    # PSNR stratified by overlap fraction (the "harder when bigger" check): does dense
    # agreement decay as the shared region grows? All GT-free.
    print("\nPSNR by overlap fraction:")
    print(f"{'overlap frac':>26}" + "".join(f"{c:>10}" for c in cols))
    for lo, hi in [(0.0, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        cells = []
        for c in cols:
            v = [p for (f, p, _s) in ev[c] if lo <= f < hi]
            cells.append(f"{np.mean(v):>10.2f}" if v else f"{'-':>10}")
        print(f"{f'[{lo:.2f},{hi:.2f})':>26}" + "".join(cells))

    print("\nreciprocity/white_correspondence/psnr/ssim: higher=better | displacement/motion/count: lower=better")
    print("NB: a sparse world scores high on rates at near-zero volume — compare columns at similar volume.")
    print("model should beat the single-agent baseline on agreement AND approach the ceiling (detector/hue floor).")

    # Structured results, mirroring the printed tables, for --save-json (see main()).
    results = {"n_ep": n_ep, "columns": {}}
    for s in cols:
        results["columns"][s] = {
            "tier_a": {k: mean(A[s], k) for k in
                       ["detection_rate", "position_error", "mean_detected_per_frame",
                        "mean_gt_visible_per_frame"]},
            "consistency": {k: mean(B[s], k) for k in
                            ["reciprocity_rate", "displacement_error", "motion_error",
                             "white_correspondence", "white_count_error", "sightings_per_frame"]},
            "pixel": {k: mean(C[s], k) for k in ["psnr", "ssim", "mean_overlap_frac"]},
            "volume": {
                "n_sightings": total(B[s], "n_sightings"),
                "n_reciprocal": total(B[s], "n_reciprocal"),
                "n_matched_white": total(B[s], "n_matched_white"),
                "n_overlap_white": total(B[s], "n_overlap_white"),
                "n_pixel_events": total(C[s], "n_pixel_events"),
            },
        }
    results["psnr_by_overlap"] = {}
    for lo, hi in [(0.0, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        key = f"{lo:.2f}-{hi:.2f}"
        results["psnr_by_overlap"][key] = {}
        for c in cols:
            v = [p for (f, p, _s) in ev[c] if lo <= f < hi]
            results["psnr_by_overlap"][key][c] = float(np.mean(v)) if v else None
    return results


@torch.no_grad()
def run_overlay(cfg, model, ds, decode_fn, device, args):
    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(cfg.data.num_future_frames)
    out = Path(cfg.output_dir) / "eval" / f"{args.source}_overlay_{args.seconds:g}s"
    out.mkdir(parents=True, exist_ok=True)
    n_ep = min(int(args.episodes), len(ds.episodes))
    print(f"{args.source} overlay: {args.seconds}s = {n_lat} latent frames, {n_ep} episodes -> {out}")
    for e in range(n_ep):
        ep = ds.full_episode(e); ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        if args.source == "gt":
            lat = ep["frames"][:, :T]
        else:
            frames = ep["frames"].unsqueeze(0).to(device)
            actions = ep["actions"].unsqueeze(0).to(device)
            lat = _rollout_model(model, frames, actions, ctx, T, wf, int(args.steps))
        positions = _gt_positions(cfg, ds, ep["episode_id"])
        views = [ov.draw_gt_marks(ov.upscale(ov.to_uint8(decode_fn(lat[j]).detach().cpu()), int(args.upscale)),
                                  positions, ep["agent_indices"][j] - 1) for j in range(lat.shape[0])]
        write_mp4(ov.tile(views), str(out / f"ep{ep['episode_id']}_{args.source}.mp4"), int(args.sim_fps))
        print(f"  wrote ep{ep['episode_id']} ({e + 1}/{n_ep})")
    print("done ->", out)


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
                    help="dart-detector background-subtraction kernel, gradient-background "
                         "setups only (see boid_detect.detect_boids). Auto-defaults to 31 when "
                         "the config's sim_overrides set rendering.background_gradient=true; "
                         "pass explicitly to override, or 0 to force it off.")
    ap.add_argument("--save-json", default=None,
                    help="metrics mode: write the structured results table to this JSON path.")
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
                  "stats (a documented approximation here -- see compare_experiments.py "
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
    else:
        run_overlay(cfg, model, ds, decode_fn, device, args)


if __name__ == "__main__":
    main()

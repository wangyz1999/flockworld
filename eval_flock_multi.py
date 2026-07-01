"""Multi-agent cross-view evaluation for FlockDiT.

Two modes (``--mode``):

* **overlay** (qualitative): roll out all P agents' views, decode, overlay
  GT-projected boid marks (red ring = focal agent, green = other boids), tile
  into one grid video. Sources: ``--source gt`` (decode REAL latents -> geometry
  sanity) or ``--source pred`` (roll out the model -> prediction overlay).

* **metrics** (quantitative): compute Tier A (per-view fidelity vs GT) and Tier B
  (cross-view consistency) for the sources and print a bracketed table:
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

from modeling.configs import load_cfg
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import gt_project as gp, overlay as ov, consistency as cs
from modeling.eval.boid_detect import detect_boids
from eval_flock_dit import find_best_checkpoint, build_model, write_mp4
from train_flock_dit import build_decode_fn

NUM_SIM_BOIDS = 100  # all boids are in the parquet; any can be a GT mark


def _gt_positions(cfg, episode_id):
    return gp.load_gt_positions(
        Path(cfg.data.root) / "state_action" / f"{episode_id}.parquet", NUM_SIM_BOIDS)


def _load_model(cfg, output_dir, device):
    ckpt, val = find_best_checkpoint(Path(output_dir) / "checkpoints")
    print(f"checkpoint: {ckpt} (val_loss={val:.5f})")
    m = build_model(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return m.eval()


def _detect_episode(lat, decode_fn):
    """lat (P, T, z, h, w) -> dets[agent][frame] = (N,2) detected centroids."""
    dets = []
    for j in range(lat.shape[0]):
        frames = ov.to_uint8(decode_fn(lat[j]).detach().cpu())
        dets.append([detect_boids(frames[t])["centroids"] for t in range(len(frames))])
    return dets


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
    A = {s: [] for s in cols}; B = {s: [] for s in cols}
    print(f"metrics: {args.seconds}s = {n_lat} latent frames, {n_ep} episodes (baseline={args.baseline})")
    for e in range(n_ep):
        ep = ds.full_episode(e); ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        cam_idx = [ai - 1 for ai in ep["agent_indices"]]
        pos = _gt_positions(cfg, ep["episode_id"])
        frames = ep["frames"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        lat = {
            "ceiling": ep["frames"][:, :T],
            "model": _rollout_model(model, frames, actions, ctx, T, wf, steps),
        }
        if args.baseline == "single":
            lat["baseline"] = _rollout_single(baseline_model, frames, actions, ctx, T, wf, steps)
        for s in cols:
            dets = _detect_episode(lat[s], decode_fn)
            A[s].append(cs.tier_a(dets, pos, cam_idx))
            B[s].append(cs.tier_b(dets, pos, cam_idx))
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
    table("Tier B — cross-view consistency (ceiling=best | baseline=floor)", B,
          ["correspondence", "positional_consistency", "temporal_std",
           "exclusion_rate", "covisibility_rate", "mean_covisible_per_frame"])
    print("\ncorrespondence/detection: higher=better | positional/temporal/exclusion/position_error: lower=better")
    print("model should beat the baseline on Tier B (interleaving buying cross-view agreement) and approach ceiling.")


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
        positions = _gt_positions(cfg, ep["episode_id"])
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
    ap.add_argument("--baseline-config", default="config/train_flockdit_latent.yaml")
    ap.add_argument("--baseline-output-dir", default=None,
                    help="output_dir of the single-agent model (for --baseline single).")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50, help="Euler steps per window.")
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

    model = baseline_model = None
    if args.mode == "metrics" or args.source == "pred":
        print("multi-agent model:")
        model = _load_model(cfg, cfg.output_dir, device)
    if args.mode == "metrics" and args.baseline == "single":
        bov = [f"output_dir={args.baseline_output_dir}"] if args.baseline_output_dir else []
        bcfg = load_cfg(args.baseline_config, bov)
        print("single-agent baseline model:")
        baseline_model = _load_model(bcfg, bcfg.output_dir, device)

    if args.mode == "metrics":
        run_metrics(cfg, model, baseline_model, ds, decode_fn, device, args)
    else:
        run_overlay(cfg, model, ds, decode_fn, device, args)


if __name__ == "__main__":
    main()

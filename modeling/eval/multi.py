"""Reusable multi-agent rollouts, detection, metrics, and overlay generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from modeling import flow_matching as fm
from modeling.eval import gt_project as gp, overlay as ov, consistency as cs, pair_consistency as pc
from modeling.eval.boid_detect import detect_boids
from modeling.eval.common import build_model, find_best_checkpoint, write_mp4


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


def _decode_detect(lat, decode_fn, background_size=None, n_cam=None,
                   hue_smooth_window=0, hue_link_dist=8.0):
    """lat (P, T, z, h, w) -> (frames[agent] (Tp,H,W,3) uint8, dets[agent][frame] dict, stats).

    Decode each agent's latents ONCE; keep the pixel frames (for pixel_consistency)
    and run the boid detector on them (centroids+hue for tier_a / pair_consistency).
    ``background_size``: gradient-background setups only -- see boid_detect.detect_boids.

    ``n_cam`` (not None) additionally runs ``pair_consistency.smooth_identities``:
    the VAE flashes boid colors for stretches of frames, so each boid's identity is
    voted along its track rather than read per frame. Pass ``n_cam=None`` for the
    raw per-frame behavior. ``stats`` is ``None`` when smoothing is off.
    """
    frames_all, dets_all = [], []
    for j in range(lat.shape[0]):
        frames = ov.to_uint8(decode_fn(lat[j]).detach().cpu())
        frames_all.append(frames)
        dets_all.append([detect_boids(frames[t], background_size=background_size)
                          for t in range(len(frames))])
    stats = None
    if n_cam is not None:
        dets_all, stats = pc.smooth_identities(dets_all, n_cam, window=int(hue_smooth_window),
                                               max_link=float(hue_link_dist))
    return frames_all, dets_all, stats


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
    S = {s: [] for s in cols}                              # hue-smoothing stats per episode
    n_cam = _num_camera_agents(cfg, cfg.data.num_agents)   # hue period for identity colors
    smooth_cam = None if args.no_hue_smooth else n_cam     # None -> per-frame hue, no smoothing
    episode_ids = []                                       # per-instance record, same order as A/B/C/S
    print(f"metrics: {args.seconds}s = {n_lat} latent frames, {n_ep} episodes "
          f"(baseline={args.baseline}, n_cam={n_cam})")
    print("hue smoothing: OFF (per-frame identity)" if args.no_hue_smooth else
          f"hue smoothing: ON (window={args.hue_smooth_window or 'whole track'}, "
          f"link={args.hue_link_dist:g}px)")
    for e in range(n_ep):
        ep = ds.full_episode(e); ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        episode_ids.append(ep["episode_id"])
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
            frames, dets, sm = _decode_detect(
                lat[s], decode_fn, background_size=args.background_size, n_cam=smooth_cam,
                hue_smooth_window=args.hue_smooth_window, hue_link_dist=args.hue_link_dist)
            if sm is not None:
                S[s].append(sm)
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

    # How unstable the per-frame color read was: the fraction of detections whose
    # identity the track vote overrode. The ceiling column is the VAE-only flash
    # floor (GT latents in), so model-minus-ceiling is the model's own color drift.
    if any(S[c] for c in cols):
        print("\nhue-flash diagnostics (fraction of boids relabeled by the track vote):")
        print(f"{'':>26}" + "".join(f"{c:>10}" for c in cols))
        for key, lab in [("relabeled_frac", "relabeled_frac"), ("mean_track_len", "mean_track_len")]:
            print(f"{lab:>26}" + "".join(f"{mean(S[c], key):>10.3f}" for c in cols))

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
            "hue_smoothing": ({k: mean(S[s], k) for k in
                               ["relabeled_frac", "n_tracks", "mean_track_len"]} if S[s] else None),
        }
    results["psnr_by_overlap"] = {}
    for lo, hi in [(0.0, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        key = f"{lo:.2f}-{hi:.2f}"
        results["psnr_by_overlap"][key] = {}
        for c in cols:
            v = [p for (f, p, _s) in ev[c] if lo <= f < hi]
            results["psnr_by_overlap"][key][c] = float(np.mean(v)) if v else None

    # Raw per-episode records (not just the means above) -- lets --save-csv reproduce
    # every number in this run (including error bars / distributions) without a rerun.
    results["episode_ids"] = episode_ids
    results["per_episode"] = {
        s: {"tier_a": A[s], "consistency": B[s], "pixel": C[s], "hue_smoothing": S[s]}
        for s in cols
    }
    return results


def write_metrics_csv(path, tag, results):
    """One row per (column, episode, metric) -- long/tidy format, safe to concat across runs."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_ids = results["episode_ids"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "column", "episode_index", "episode_id", "family", "metric", "value"])
        for col, families in results["per_episode"].items():
            for family, records in families.items():
                for i, rec in enumerate(records):
                    eid = episode_ids[i] if i < len(episode_ids) else ""
                    for k, v in rec.items():
                        if isinstance(v, (list, dict, tuple)):
                            continue  # non-scalar (e.g. raw event lists) -- not tidy-CSV shaped
                        w.writerow([tag, col, i, eid, family, k, v])
    print(f"wrote per-episode CSV -> {path}")


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

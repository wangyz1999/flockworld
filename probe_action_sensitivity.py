"""Diagnostic 1 — does the action channel actually drive the rollout?

The conditioning action in this sim is ENDOGENOUS: the focal agent's acceleration is
``alignment*w + cohesion*w + separation*w`` over the neighbours inside ``vision``
(flockworld/core/boids.py), i.e. a function of exactly what the egocentric crop shows.
So the model can learn to read the acceleration off the neighbours and ignore the
action channel. If it did, then feeding stale GT actions to a drifted rollout is
harmless -- but the model is also not action-controllable.

This probe rolls the SAME context out under several action streams and asks how far
the generated video moves. Only actions from ``ctx`` onward are altered; the context
window keeps its true actions, so we are measuring "what happens when the command
after the context is wrong", which is exactly the rollout situation.

Conditions:
  gt          true actions (the reference every other condition is compared against)
  gt_seed2    true actions, different sampling noise  -> STOCHASTIC NOISE FLOOR
  zero        actions clamped to 0 after ctx
  neg         actions negated after ctx (same magnitude, opposite command)
  perm        actions time-shuffled after ctx (same marginal, wrong timing)
  other       actions taken from a different held-out episode
  agentswap   agent p gets agent p+1's actions (multi-agent identity binding)

Every condition except ``gt_seed2`` reuses ``gt``'s sampler noise draw (same seed,
identical randn call sequence), so condition-vs-gt is a PAIRED comparison under common
random numbers. ``gt_seed2`` breaks that coupling and measures how far two samples of
the same command drift apart -- the stochastic noise floor.

The headline number is the Action Sensitivity Index

    ASI(c) = divergence(c, gt) / divergence(gt_seed2, gt)

ASI ~ 1 means changing the action moved the rollout no more than resampling the
sampler's noise did: the action channel is inert. ASI >> 1 means the model obeys.

READ ASI AT THE EARLY HORIZON. Late-horizon divergence saturates: after a few seconds
both rollouts are effectively independent samples from the model's marginal, so every
ASI drifts toward 1.0 by construction and says nothing about the action. The first
generated window is where the action's local effect is measurable.

The consistency columns say whether a wrong command also *costs* cross-view
consistency (outcome b) or leaves it untouched (outcome a).

NB ``zero`` zeroes ACCELERATION, not velocity -- a boid with zero acceleration keeps
coasting, so ``motion`` (apparent world motion) is not expected to collapse under it.
``motion`` is reported as a sanity channel, not an adherence test; adherence proper is
diagnostic 2.

Usage:
  uv run python probe_action_sensitivity.py --include exp08_tiled_two_stage \
      --episodes 4 --seconds 10
  uv run python probe_action_sensitivity.py --include exp01_baseline exp06_tiled_df \
      exp08_tiled_two_stage --episodes 6 --seconds 10 --out probe_out/asi.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling.eval import pair_consistency as pc
from eval_flock_multi import _decode_detect, _gt_positions, _load_model, _num_camera_agents
from compare_experiments import EXPERIMENTS, RUN_ROOT
from modeling import flow_matching as fm

CONDITIONS = ["gt", "gt_seed2", "zero", "neg", "perm", "other", "agentswap"]


def make_actions(cond, acts, ctx, other_acts, gen):
    """acts (P, L, A) -> a modified copy; only indices >= ctx change."""
    a = acts.clone()
    P, L, _ = a.shape
    if cond in ("gt", "gt_seed2"):
        return a
    if cond == "zero":
        a[:, ctx:] = 0.0
        return a
    if cond == "neg":
        a[:, ctx:] = -a[:, ctx:]
        return a
    if cond == "perm":
        n = L - ctx
        for p in range(P):                        # independent shuffle per agent
            idx = torch.randperm(n, generator=gen)
            a[p, ctx:] = a[p, ctx:][idx]
        return a
    if cond == "other":
        n = min(L, other_acts.shape[1])
        a[:, ctx:n] = other_acts[:, ctx:n]
        if n < L:                                 # shorter donor: hold its last action
            a[:, n:] = other_acts[:, -1:].expand(-1, L - n, -1)
        return a
    if cond == "agentswap":
        a[:, ctx:] = acts.roll(-1, dims=0)[:, ctx:]
        return a
    raise ValueError(cond)


@torch.no_grad()
def rollout(model, frames, acts, ctx, T, wf, steps, seed):
    torch.manual_seed(seed)                       # sampler noise is torch.randn in multi_euler_rollout
    return fm.multi_autoregressive_rollout(
        model, frames[:, :, :ctx], acts, T, window_future=wf, num_steps=steps)[0]


def psnr_seq(a, b):
    """Per-frame PSNR between two decoded uint8 frame lists (per agent), averaged over agents."""
    out = []
    for t in range(min(len(a[0]), len(b[0]))):
        vals = []
        for j in range(len(a)):
            x = a[j][t].astype(np.float64); y = b[j][t].astype(np.float64)
            mse = float(np.mean((x - y) ** 2))
            vals.append(99.0 if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse))
        out.append(float(np.mean(vals)))
    return out


def apparent_motion(frames_all):
    """Mean |frame_t - frame_{t-1}| per agent-frame -- a crude proxy for how much the
    world appears to move (in an egocentric crop this is dominated by ego motion)."""
    vals = []
    for ag in frames_all:
        d = [float(np.mean(np.abs(ag[t].astype(np.float64) - ag[t - 1].astype(np.float64))))
             for t in range(1, len(ag))]
        vals.append(float(np.mean(d)) if d else float("nan"))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description="Action-sensitivity probe (diagnostic 1).")
    ap.add_argument("--include", nargs="+", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--rollout-seed", type=int, default=1234)
    ap.add_argument("--out", default=f"{RUN_ROOT}/eval_logs/action_sensitivity.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conds = [c for c in CONDITIONS if c in args.conditions]
    if "gt" not in conds or "gt_seed2" not in conds:
        raise SystemExit("gt and gt_seed2 are both required (gt_seed2 is the noise floor).")

    from modeling.eval.streaming_eval_dataset import (
        StreamingMultiEvalDataset, fit_train_stats, make_decode_fn, normalize_frames,
    )
    from modeling.models.frozen_vae import FrozenVAE

    first_cfg = load_cfg(EXPERIMENTS[args.include[0]][0], [])
    vae = FrozenVAE(str(first_cfg.vae.checkpoint_path), device=device)
    # +1 episode so "other" has a donor that is never the episode being rolled out
    ds = StreamingMultiEvalDataset(
        first_cfg, vae, device, num_episodes=int(args.episodes) + 1,
        seconds=float(args.seconds), sim_fps=int(args.sim_fps), eval_seed=int(args.eval_seed),
    )

    pix_frames = round(float(args.seconds) * int(args.sim_fps))
    n_lat = 1 + (pix_frames - 1) // 4
    wf = int(first_cfg.data.num_future_frames)
    n_ep = min(int(args.episodes), len(ds.episodes) - 1)
    n_cam = _num_camera_agents(first_cfg, first_cfg.data.num_agents)
    print(f"{n_ep} episodes, {args.seconds}s ({n_lat} latent frames), conditions={conds}")

    episode_ctx = []
    for e in range(n_ep):
        ep = ds.full_episode(e)
        donor = ds.full_episode(e + 1)            # next held-out episode supplies "other"
        ctx = int(ep["context_len"]); T = min(n_lat, ep["frames"].shape[1])
        episode_ctx.append((ep, donor, ctx, T,
                            [ai - 1 for ai in ep["agent_indices"]],
                            _gt_positions(first_cfg, ds, ep["episode_id"])))

    results = {}
    for key in args.include:
        cfg_path, out_dir = EXPERIMENTS[key]
        print(f"\n=== {key} ({out_dir}) ===", flush=True)
        ecfg = load_cfg(cfg_path, [f"output_dir={out_dir}"])
        model = _load_model(ecfg, ecfg.output_dir, device)
        mean_e, std_e = fit_train_stats(ecfg, vae, device)
        decode_fn = make_decode_fn(vae, device, mean_e, std_e)

        per = {c: {"motion": [], "pair": [], "lat_rmse_t": [], "psnr_t": []} for c in conds}
        first_gen_pix = None
        for i, (ep, donor, ctx, T, cam_idx, pos) in enumerate(episode_ctx):
            first_gen_pix = 1 + 4 * (ctx - 1)     # pixel index of the first GENERATED frame
            frames_in = normalize_frames(ep["frames"], mean_e, std_e).unsqueeze(0).to(device)
            acts = ep["actions"].to(device)                       # (P, L, A)
            donor_acts = donor["actions"].to(device)
            gen = torch.Generator().manual_seed(args.rollout_seed + i)

            ref_lat = ref_frames = None
            for c in conds:
                a = make_actions(c, acts, ctx, donor_acts, gen).unsqueeze(0)
                seed = args.rollout_seed + i + (10_000 if c == "gt_seed2" else 0)
                lat = rollout(model, frames_in, a, ctx, T, wf, int(args.steps), seed)
                frames_all, dets = _decode_detect(lat, decode_fn)
                if c == "gt":
                    ref_lat, ref_frames = lat, frames_all
                # divergence vs the gt-action rollout, restricted to GENERATED frames
                rmse_t = (lat[:, ctx:] - ref_lat[:, ctx:]).pow(2).mean((0, 2, 3, 4)).sqrt()
                p_t = psnr_seq(frames_all, ref_frames)
                per[c]["lat_rmse_t"].append([float(v) for v in rmse_t])
                per[c]["psnr_t"].append(p_t)
                per[c]["motion"].append(apparent_motion(frames_all))
                per[c]["pair"].append(pc.pair_consistency(dets, cam_idx, n_cam))
                print(f"  [{key}] ep{ep['episode_id']} {c:>10}  "
                      f"latRMSE[first]={rmse_t[0]:.4f}  latRMSE[last]={rmse_t[-1]:.4f}  "
                      f"PSNR[first_gen]={p_t[first_gen_pix]:.2f}", flush=True)
            print(f"  [{key}] ep{ep['episode_id']} done ({i + 1}/{n_ep})", flush=True)

        def curve(c, k):
            m = min(len(x) for x in per[c][k])
            return np.mean(np.array([x[:m] for x in per[c][k]]), axis=0)

        results[key] = {"first_gen_pix": int(first_gen_pix), "conditions": {}}
        for c in conds:
            lr, ps = curve(c, "lat_rmse_t"), curve(c, "psnr_t")
            g = ps[first_gen_pix:]                          # generated pixel frames only
            results[key]["conditions"][c] = {
                "lat_rmse_first": float(lr[0]),
                "lat_rmse_win": float(lr[:wf].mean()),      # first generated window
                "lat_rmse_late": float(lr[len(lr) // 2:].mean()),
                "psnr_first": float(g[0]),
                "psnr_1s": float(g[:int(args.sim_fps)].mean()),
                "psnr_late": float(g[len(g) // 2:].mean()),
                "motion": float(np.mean(per[c]["motion"])),
                "lat_rmse_t": lr.tolist(),
                "psnr_t": ps.tolist(),
                **{k: float(np.nanmean([d[k] for d in per[c]["pair"]]))
                   for k in ("reciprocity_rate", "displacement_error", "motion_error",
                             "white_correspondence", "sightings_per_frame")},
            }
        del model
        torch.cuda.empty_cache()

    # ---- report -------------------------------------------------------------
    for key, entry in results.items():
        r = entry["conditions"]
        fl = r["gt_seed2"]
        print(f"\n########## {key} ##########")
        print("ACTION SENSITIVITY  (paired: every row shares gt's sampler noise except gt_seed2)")
        print(f"{'condition':>10}" + "".join(f"{h:>11}" for h in
              ["rmse_1st", "ASI_1st", "rmse_win", "ASI_win", "rmse_late", "ASI_late"]))
        for c in r:
            d = r[c]
            def asi(k):
                return "-" if c == "gt" else (f"{d[k] / fl[k]:.2f}" if fl[k] > 0 else "inf")
            print(f"{c:>10}{d['lat_rmse_first']:>11.4f}{asi('lat_rmse_first'):>11}"
                  f"{d['lat_rmse_win']:>11.4f}{asi('lat_rmse_win'):>11}"
                  f"{d['lat_rmse_late']:>11.4f}{asi('lat_rmse_late'):>11}")
        print("\nPIXEL DIVERGENCE FROM THE gt-ACTION ROLLOUT (dB; HIGHER = the action changed less)")
        print(f"{'condition':>10}" + "".join(f"{h:>11}" for h in
              ["psnr_1st", "psnr_1s", "psnr_late", "motion"]))
        for c in r:
            d = r[c]
            print(f"{c:>10}{d['psnr_first']:>11.2f}{d['psnr_1s']:>11.2f}"
                  f"{d['psnr_late']:>11.2f}{d['motion']:>11.2f}")
        print("\nCROSS-VIEW CONSISTENCY UNDER EACH COMMAND (does a wrong action cost consistency?)")
        print(f"{'condition':>10}" + "".join(f"{h:>11}" for h in
              ["recip", "displ", "motionE", "white", "sight/f"]))
        for c in r:
            d = r[c]
            print(f"{c:>10}{d['reciprocity_rate']:>11.3f}{d['displacement_error']:>11.2f}"
                  f"{d['motion_error']:>11.2f}{d['white_correspondence']:>11.3f}"
                  f"{d['sightings_per_frame']:>11.2f}")
        print("\nASI = rmse(cond)/rmse(gt_seed2). ~1.0 => the action moved the rollout no more "
              "than resampling noise did (inert channel); >>1 => the model obeys.")
        print("READ ASI_1st / ASI_win. ASI_late is chaos-saturated and drifts to 1.0 regardless.")
        print("motion = mean |frame_t - frame_{t-1}| (sanity channel; 'zero' zeroes accel, "
              "not velocity, so coasting motion is expected to persist).")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

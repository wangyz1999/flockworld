"""Where in the network does the action signal go? -- the "attention check" analogue.

There is NO attention over actions in this architecture, so a literal action-attention
map does not exist. Actions never become tokens: they enter only through AdaLN, as
``e = time_embedding(t) + action_embedding(a)`` -> ``SiLU+Linear(dim, 6*dim)`` -> one
``e0`` shared by every block, added to that block's learned ``modulation`` parameter and
used as scale/shift/gate (modeling/models/flock_dit.py:343-358). Attention runs over
video tokens only, ``(B, heads, F*P*S, F*P*S)``.

So this probe measures action INFLUENCE instead, two ways:

Part A -- modulation variance share. Across a real batch, decompose the variance of the
  AdaLN signal into the part induced by the action distribution and the part induced by
  the timestep distribution, per block. Norms are the wrong statistic here: a
  near-constant action offset can have a large norm while carrying no information, so we
  hold one input at its batch mean and vary the other. Reported relative to each block's
  own ``modulation`` parameter, which is what the action has to compete with.

Part B -- action -> agent influence matrix. ``J[p][q] = || d(agent p's output) /
  d(agent q's action) ||``, normalised by the empirical per-agent action std so the units
  are "output change per real action unit". A model that binds actions to agent identity
  is DIAGONAL; a model whose action conditioning is agent-anonymous is FLAT. Summary
  statistic: diagonal dominance = mean(diag) / mean(off-diag), ~1.0 meaning no binding.

Both are diagnostic, not evaluative: they say where the signal is (or isn't), not whether
the model obeys the command. Output effect is probe_action_sensitivity.py; command
appropriateness is probe_command_coherence.py.

Usage:
  uv run python probe_action_influence.py --include exp01_baseline exp08_tiled_two_stage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modeling.configs import load_cfg
from modeling import flow_matching as fm
from eval_flock_multi import _load_model
from compare_experiments import EXPERIMENTS, RUN_ROOT


@torch.no_grad()
def modulation_share(model, t, actions):
    """Per-block action-vs-timestep variance share of the AdaLN signal.

    t (B,P,F), actions (B,P,F,A). Varies one input across the batch while the other is
    pinned to its batch mean, so each share reflects information carried, not offset size.
    """
    b, p, f = t.shape
    t_mean = t.mean(0, keepdim=True).expand_as(t).contiguous()
    a_mean = actions.mean(0, keepdim=True).expand_as(actions).contiguous()

    e_full, _ = model._modulation(t, actions, b, p, f)          # (B,F,P,6,dim)
    e_act, _ = model._modulation(t_mean, actions, b, p, f)      # only actions vary
    e_tim, _ = model._modulation(t, a_mean, b, p, f)            # only timesteps vary

    def var(x):                                                 # variance across the batch
        return float(x.float().var(dim=0, unbiased=False).sum())

    v_full, v_act, v_tim = var(e_full), var(e_act), var(e_tim)
    rows = []
    for i, blk in enumerate(model.blocks):
        m = blk.modulation.float()                              # (1,1,1,6,dim)
        # action-induced std of the AdaLN signal vs the block's own learned modulation
        act_std = float(e_act.float().var(dim=0, unbiased=False).sum().sqrt())
        rows.append({"block": i,
                     "action_std_over_modulation_norm": act_std / float(m.norm()),
                     "modulation_norm": float(m.norm())})
    return {"var_full": v_full, "var_action_only": v_act, "var_timestep_only": v_tim,
            "action_share": v_act / (v_act + v_tim) if (v_act + v_tim) > 0 else float("nan"),
            "per_block": rows}


def influence_matrix(model, x, t, actions, n_dirs=4, seed=0):
    """J[p][q] = ||d(output_p)/d(action_q)||, averaged over random output directions.

    Random projections avoid privileging any particular output channel; averaging over
    ``n_dirs`` keeps the estimate from riding on one draw. Columns are scaled by the
    empirical per-agent action std, so entries read as "output change per real action unit".
    """
    P = actions.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_std = actions.float().std(dim=(0, 2)).mean(-1).cpu().numpy()      # (P,) per-agent scale
    J = np.zeros((P, P))
    for d in range(n_dirs):
        for pt in range(P):
            a = actions.clone().requires_grad_(True)
            out = model(x, t, a)                                       # (B,P,F,C,H,W)
            op = out[:, pt]
            u = torch.randn(op.shape, generator=g).to(op.device, op.dtype)
            u = u / u.norm()
            s = (op * u).sum()
            grad, = torch.autograd.grad(s, a, retain_graph=False)
            gn = grad.float().pow(2).sum(dim=(0, 2, 3)).sqrt().cpu().numpy()   # (P,) per source
            J[pt] += gn / n_dirs
            del a, out, op, grad
    J = J * a_std[None, :]                                             # per real action unit
    diag = np.diag(J)
    off = J[~np.eye(P, dtype=bool)]
    return {"J": J.tolist(),
            "diag_mean": float(diag.mean()), "offdiag_mean": float(off.mean()),
            "diagonal_dominance": float(diag.mean() / off.mean()) if off.mean() > 0 else float("inf"),
            "row_normalized_diag": float(np.mean(diag / np.maximum(J.sum(1), 1e-12)))}


def main():
    ap = argparse.ArgumentParser(description="Action influence probe (AdaLN share + agent binding).")
    ap.add_argument("--include", nargs="+", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--sim-fps", type=int, default=30)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--dirs", type=int, default=4, help="random output directions to average.")
    ap.add_argument("--out", default=f"{RUN_ROOT}/eval_logs/action_influence.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from modeling.eval.streaming_eval_dataset import (
        StreamingMultiEvalDataset, fit_train_stats, normalize_frames)
    from modeling.models.frozen_vae import FrozenVAE

    first_cfg = load_cfg(EXPERIMENTS[args.include[0]][0], [])
    vae = FrozenVAE(str(first_cfg.vae.checkpoint_path), device=device)
    ds = StreamingMultiEvalDataset(
        first_cfg, vae, device, num_episodes=int(args.episodes), seconds=float(args.seconds),
        sim_fps=int(args.sim_fps), eval_seed=int(args.eval_seed))
    eps = [ds.full_episode(e) for e in range(min(int(args.episodes), len(ds.episodes)))]
    F = int(first_cfg.data.num_context_frames) + int(first_cfg.data.num_future_frames)
    ctx = int(first_cfg.data.num_context_frames)
    print(f"{len(eps)} episodes, one {F}-frame training-shaped window each")

    results = {}
    for key in args.include:
        cfg_path, out_dir = EXPERIMENTS[key]
        print(f"\n=== {key} ===", flush=True)
        ecfg = load_cfg(cfg_path, [f"output_dir={out_dir}"])
        model = _load_model(ecfg, ecfg.output_dir, device)
        mean_e, std_e = fit_train_stats(ecfg, vae, device)

        # one training-shaped batch: (B=episodes, P, F, z, h, w) + matching actions
        xs = [normalize_frames(ep["frames"], mean_e, std_e)[:, :F] for ep in eps]
        acts = [ep["actions"][:, :F] for ep in eps]
        x = torch.stack(xs).to(device)
        a = torch.stack(acts).to(device)
        b, P = x.shape[0], x.shape[1]
        t = fm.sample_timesteps((b, P), F, ctx, device)
        xt, _ = fm.add_noise(x, t)

        try:
            share = modulation_share(model, t, a)
            infl = influence_matrix(model, xt, t, a, n_dirs=int(args.dirs))
        except torch.OutOfMemoryError as e:
            # A co-tenant process on the GPU shouldn't cost us the columns already done.
            print(f"  [SKIPPED] {key}: CUDA OOM ({e.__class__.__name__}); "
                  f"other columns are still written.", flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        results[key] = {"modulation": share, "influence": infl,
                        "num_agents": int(P), "broadcast_actions": bool(
                            getattr(model, "broadcast_actions", False))}
        print(f"  broadcast_actions={results[key]['broadcast_actions']}")
        print(f"  AdaLN variance share: action={share['action_share']:.4f}  "
              f"(timestep={1 - share['action_share']:.4f})")
        print(f"  action-induced std / block modulation norm: "
              f"{np.mean([r['action_std_over_modulation_norm'] for r in share['per_block']]):.4f}")
        print(f"  influence diagonal dominance = {infl['diagonal_dominance']:.3f}   "
              f"(1.0 => agent-anonymous, >>1 => bound to identity)")
        del model
        torch.cuda.empty_cache()

    print("\n########## ACTION INFLUENCE ##########")
    print(f"{'experiment':>26}{'bcast':>7}{'AdaLN act share':>18}{'diag dominance':>16}")
    for k, r in results.items():
        print(f"{k:>26}{str(r['broadcast_actions']):>7}"
              f"{r['modulation']['action_share']:>18.4f}"
              f"{r['influence']['diagonal_dominance']:>16.3f}")
    print("\nAdaLN act share = fraction of AdaLN-signal batch variance induced by the action "
          "distribution rather than the timestep distribution.")
    print("diag dominance = mean(diag J)/mean(offdiag J); 1.0 means agent p's own action is "
          "no more influential on agent p's view than any other agent's.")
    print("Both are DIAGNOSTIC (where the signal is), not evaluative (whether the model obeys).")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

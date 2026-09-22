"""Cross-view attention interpretability for a multi-agent FlockDiT model.

For one query agent's last (generated) frame, shows -- per OTHER view, arranged
in the model's own tile layout (or a square-ish grid for non-tiled models) --
WHERE in that view's 8x8 latent grid the query attends (heatmap, overlaid on the
view's decoded context frame) and HOW MUCH total attention mass lands on that
view vs. the others (annotated % + printed table). Averaged over all heads and
layers. Shared extraction logic lives in modeling/eval/attention_probe.py.

Caveat (read before interpreting): a single forward pass noises every agent's
future frames at once (like one step of Euler rollout), so "same generated
frame, other agent" keys are noise, not real content -- there's nothing
meaningful to overlay a heatmap on for that portion. This script therefore
aggregates/visualizes attention onto CONTEXT frames only (real, clean, decodable)
and separately reports what fraction of the query's total attention mass that
context-only view actually accounts for (context vs. same-frame-live vs. own
past), so you know how much of the total the picture represents.

    uv run python -m scripts.figures.gen_attention_heatmap --config <cfg> --output-dir <dir> \
        --episode 0 --query-agent 0 --sigma 1.0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from modeling.eval import overlay as ov
from modeling.eval.attention_probe import corner_heaviness, load_model_and_data, extract_dist


def tile_fixed(views, rows, cols, pad=4, bg=20) -> np.ndarray:
    """List of (H, W, 3) uint8 (same shape) -> one (gridH, gridW, 3) grid, exact rows x cols."""
    H, W, _ = views[0].shape
    gh, gw = rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad
    grid = np.full((gh, gw, 3), bg, np.uint8)
    for i, v in enumerate(views):
        r, c = divmod(i, cols)
        y0, x0 = pad + r * (H + pad), pad + c * (W + pad)
        grid[y0:y0 + H, x0:x0 + W] = v
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train_flockdit_latent_multi_stream_tiled.yaml")
    ap.add_argument("--output-dir", default="output/flock_dit_multi_tiled")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--query-agent", type=int, default=0,
                    help="local index (0..P-1) of the view generating a frame")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="noise level of future frames at extraction time (1.0 = first Euler step)")
    ap.add_argument("--upscale", type=int, default=16, help="8x8 heatmap -> 128x128 for overlay")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, ds, decode_fn, ck, val = load_model_and_data(args.config, args.output_dir, device)
    print(f"ckpt {ck} val {val:.4f}")

    dist, ep, ctx, f, p, s, h, w = extract_dist(
        model, cfg, ds, args.episode, args.query_agent, args.sigma, device)
    qf, qp = f - 1, args.query_agent
    tile_grid = cfg.model.get("tile_grid", None)
    if tile_grid is not None:
        rows, cols = tuple(tile_grid)          # tiled model's own layout
    else:
        cols = int(np.ceil(np.sqrt(p)))        # interleaved: no native layout, use a square-ish grid
        rows = int(np.ceil(p / cols))

    total_mass = dist.sum().item()  # should be ~1.0 (softmax rows sum to 1)
    own_past = dist[:ctx, qp].sum().item()
    own_live = dist[ctx:, qp].sum().item()   # same/other future frames, own agent (noise, not renderable)
    other_ctx = dist[:ctx, [j for j in range(p) if j != qp]].sum().item()
    other_live = dist[ctx:, [j for j in range(p) if j != qp]].sum().item()
    print(f"\nquery: agent {qp} (world idx {ep['agent_indices'][qp]}), frame {qf} (generated), sigma={args.sigma}")
    print(f"total attention mass: {total_mass:.4f} (should be ~1.0)")
    print(f"  own past (context, real):        {own_past / total_mass:6.2%}")
    print(f"  own live (same/future, noise):    {own_live / total_mass:6.2%}")
    print(f"  OTHER agents' context (real):     {other_ctx / total_mass:6.2%}  <- visualized below")
    print(f"  OTHER agents' live (noise):       {other_live / total_mass:6.2%}")

    # per-agent context-only spatial map (renderable: real clean latents)
    ctx_dist = dist[:ctx]  # (ctx, p, sk)
    per_agent_ctx = ctx_dist.sum(dim=0)  # (p, sk) -- sum over context frames
    per_agent_share = per_agent_ctx.sum(dim=-1) / max(other_ctx + own_past, 1e-9)  # share of context-mass, incl self
    print(f"  corner/center heaviness (other agents' context maps): "
          f"{corner_heaviness(per_agent_ctx, qp):.2f}x")

    tiles = []
    last_ctx_frame = ep["frames"][:, ctx - 1]  # (P, C, H, W), real clean latents
    for j in range(p):
        img = ov.to_uint8(decode_fn(last_ctx_frame[j:j + 1]))[0]  # (H_img, W_img, 3)
        img = ov.upscale(img[None], args.upscale)[0]
        heat = per_agent_ctx[j].reshape(h, w).cpu().numpy()
        heat = heat / (heat.max() + 1e-9)  # per-view normalize -- shows WHERE within this view
        heat_img = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat_img = np.clip(heat_img, 0, 1)
        heat_color = cv2.applyColorMap((heat_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        blend = (0.45 * img + 0.55 * heat_color).astype(np.uint8)
        label = f"a{ep['agent_indices'][j]}" + (" (query)" if j == qp else f" {per_agent_share[j]:.1%}")
        cv2.putText(blend, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0) if j == qp else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blend, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0) if j == qp else (255, 255, 255), 1, cv2.LINE_AA)
        if j == qp:
            cv2.rectangle(blend, (0, 0), (blend.shape[1] - 1, blend.shape[0] - 1), (255, 0, 0), 3)
        tiles.append(blend)

    grid = tile_fixed(tiles, rows, cols)
    out_dir = Path(args.output_dir) / "eval" / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ep{ep['episode_id']}_q{qp}_f{qf}_sigma{args.sigma:g}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {out_path} ({rows}x{cols} grid, heatmap = share of context-only attention mass, "
          f"per-tile %% = share of {qp}'s CONTEXT-mass total, query outlined in red)")


if __name__ == "__main__":
    main()

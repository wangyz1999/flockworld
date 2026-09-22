"""Border-artifact ablation: is the corner attention hotspot driven by the
rendered white border (border_width=2), or is it content-driven?

Decodes the real context frames, zeros the outer border_px pixels to the
background color, re-encodes through the SAME frozen VAE, substitutes that as
context, and re-runs the same query -- comparing corner_heaviness and the
own/other split before vs. after. If the hotspot is the border, it should
collapse toward 1.0x (no longer disproportionate) once the border is gone.

    uv run python -m scripts.diagnostics.border_attention
"""
import torch

from modeling.eval.attention_probe import (
    corner_heaviness, extract_dist, load_model_and_data, mask_border_and_reencode,
)

RUNS = [
    ("tiled", "config/train_flockdit_latent_multi_stream_tiled.yaml", "output/flock_dit_multi_tiled"),
    ("interleaved", "config/train_flockdit_latent_multi_stream.yaml", "output/flock_dit_multi_interleaved"),
]
EPISODES = [0, 1, 2]
QUERY_AGENTS = [0, 3]
SIGMA = 1.0


def summarize(dist, ctx, qp):
    total = dist.sum().item()
    own = dist[:, qp].sum().item()
    per_agent_ctx = dist[:ctx].sum(dim=0)
    return (total - own) / total, corner_heaviness(per_agent_ctx, qp)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, config, output_dir in RUNS:
        cfg, model, ds, decode_fn, ck, val = load_model_and_data(config, output_dir, device)
        print(f"\n=== {name} (ckpt {ck.name}, val {val:.4f}) ===")
        for e in EPISODES:
            ep = ds.full_episode(e)
            ctx = int(ep["context_len"])
            ctx_masked = mask_border_and_reencode(ep, ctx, cfg, device)
            for qp in QUERY_AGENTS:
                dist_base, *_ = extract_dist(model, cfg, ds, e, qp, SIGMA, device)
                dist_mask, *_ = extract_dist(model, cfg, ds, e, qp, SIGMA, device, ctx_override=ctx_masked)
                other_b, corner_b = summarize(dist_base, ctx, qp)
                other_m, corner_m = summarize(dist_mask, ctx, qp)
                print(f"  ep{ep['episode_id']:>5} q{qp}: "
                      f"baseline other={other_b:6.2%} corner={corner_b:5.2f}x  |  "
                      f"border-masked other={other_m:6.2%} corner={corner_m:5.2f}x")


if __name__ == "__main__":
    main()

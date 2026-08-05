"""Clean control for the border-artifact question: render a FEW FRESH episodes
straight from the sim with rendering.border_width=0 (genuinely in-distribution,
no post-hoc masking edge), and compare corner/ring heaviness against the real
(border=2) recorded episodes. Complements test_border_attention.py's masking
ablation, which can't fully rule out "the masking edge itself is the anchor."

    uv run python test_border0_clean.py
"""
import torch

from modeling.eval.attention_probe import (
    corner_heaviness, extract_dist, extract_dist_from_ep, load_model_and_data,
    ring_heaviness, synth_border0_episode,
)

RUNS = [
    ("tiled", "config/train_flockdit_latent_multi_stream_tiled.yaml", "output/flock_dit_multi_tiled"),
    ("interleaved", "config/train_flockdit_latent_multi_stream.yaml", "output/flock_dit_multi_interleaved"),
]
SEEDS = [1001, 1002, 1003]   # fresh seeds, not among the 100 recorded episodes
QUERY_AGENTS = [0, 3]
SIGMA = 1.0


def summarize(dist, ctx, qp):
    total = dist.sum().item()
    own = dist[:, qp].sum().item()
    per_agent_ctx = dist[:ctx].sum(dim=0)
    return (total - own) / total, corner_heaviness(per_agent_ctx, qp), ring_heaviness(per_agent_ctx, qp)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, config, output_dir in RUNS:
        cfg, model, ds, decode_fn, ck, val = load_model_and_data(config, output_dir, device)
        print(f"\n=== {name} (ckpt {ck.name}, val {val:.4f}) ===")
        ctx = int(cfg.data.num_context_frames)
        wf = int(cfg.data.num_future_frames)
        for seed in SEEDS:
            print(f"  rendering fresh seed={seed}, border_width=0 ...")
            ep = synth_border0_episode(cfg, seed, ctx, wf, device)
            for qp in QUERY_AGENTS:
                dist, *_ = extract_dist_from_ep(model, cfg, ep, qp, SIGMA, device)
                other, corner, ring = summarize(dist, ctx, qp)
                print(f"  seed{seed:>5} q{qp}: border0-clean  other={other:6.2%}  "
                      f"corner={corner:5.2f}x  ring={ring:5.2f}x")
        # baseline for direct comparison: same models' real (border=2) recorded episodes
        print("  -- baseline (real, border=2) for comparison --")
        for e in [0, 1]:
            for qp in QUERY_AGENTS:
                dist, ep2, ctx2, *_ = extract_dist(model, cfg, ds, e, qp, SIGMA, device)
                other, corner, ring = summarize(dist, ctx2, qp)
                print(f"  ep{ep2['episode_id']:>5} q{qp}: baseline        other={other:6.2%}  "
                      f"corner={corner:5.2f}x  ring={ring:5.2f}x")


if __name__ == "__main__":
    main()

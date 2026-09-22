"""Robustness sweep: does the "own vs. other" split and the corner-anchor pattern
hold up across multiple episodes/query agents, for both the tiled and interleaved
models? Loads each model once, loops over (episode, query_agent) pairs, prints
per-run and aggregate (mean +/- std) numbers -- no images, just the summary stats.

    uv run python -m scripts.analysis.analyze_attention_sweep
"""
import numpy as np
import torch

from modeling.eval.attention_probe import corner_heaviness, extract_dist, load_model_and_data

RUNS = [
    ("tiled", "config/train_flockdit_latent_multi_stream_tiled.yaml", "output/flock_dit_multi_tiled"),
    ("interleaved", "config/train_flockdit_latent_multi_stream.yaml", "output/flock_dit_multi_interleaved"),
]
EPISODES = [0, 1, 2, 3]
QUERY_AGENTS = [0, 3, 6]
SIGMA = 1.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, config, output_dir in RUNS:
        cfg, model, ds, decode_fn, ck, val = load_model_and_data(config, output_dir, device)
        print(f"\n=== {name} (ckpt {ck.name}, val {val:.4f}) ===")
        other_shares, heaviness = [], []
        for e in EPISODES:
            for qp in QUERY_AGENTS:
                dist, ep, ctx, f, p, s, h, w = extract_dist(model, cfg, ds, e, qp, SIGMA, device)
                total = dist.sum().item()
                own = dist[:, qp].sum().item()
                other = total - own
                per_agent_ctx = dist[:ctx].sum(dim=0)  # (p, sk)
                ch = corner_heaviness(per_agent_ctx, qp)
                other_shares.append(other / total)
                heaviness.append(ch)
                print(f"  ep{ep['episode_id']:>5} q{qp}: other_share={other / total:6.2%}  corner_heaviness={ch:5.2f}x")
        other_shares, heaviness = np.array(other_shares), np.array(heaviness)
        print(f"  -- {name} aggregate over {len(other_shares)} runs --")
        print(f"  other_share:      {other_shares.mean():6.2%} +/- {other_shares.std():.2%}")
        print(f"  corner_heaviness: {heaviness.mean():5.2f}x +/- {heaviness.std():.2f}")


if __name__ == "__main__":
    main()

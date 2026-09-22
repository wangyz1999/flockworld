"""Is it specifically corners, or the whole boundary? Three-way breakdown of
attention density: corner (4 cells) vs edge-only (24 border-but-not-corner
cells) vs interior (36 cells), all normalized per-cell so magnitudes are
directly comparable (not two overlapping ratios).

    uv run python -m scripts.analysis.analyze_corner_vs_edge
"""
import numpy as np
import torch

from modeling.eval.attention_probe import corner_edge_interior, extract_dist, load_model_and_data

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
        corners, edges, interiors = [], [], []
        for e in EPISODES:
            for qp in QUERY_AGENTS:
                dist, ep, ctx, f, p, s, h, w = extract_dist(model, cfg, ds, e, qp, SIGMA, device)
                per_agent_ctx = dist[:ctx].sum(dim=0)
                c, ed, i = corner_edge_interior(per_agent_ctx, qp)
                corners.append(c); edges.append(ed); interiors.append(i)
                print(f"  ep{ep['episode_id']:>5} q{qp}: corner={c:.5f}  edge={ed:.5f}  interior={i:.5f}  "
                      f"(corner/edge={c / ed:.2f}x, edge/interior={ed / i:.2f}x)")
        corners, edges, interiors = np.array(corners), np.array(edges), np.array(interiors)
        print(f"  -- {name} aggregate (per-cell density, mean +/- std) --")
        print(f"  corner:   {corners.mean():.5f} +/- {corners.std():.5f}")
        print(f"  edge:     {edges.mean():.5f} +/- {edges.std():.5f}")
        print(f"  interior: {interiors.mean():.5f} +/- {interiors.std():.5f}")
        print(f"  corner/edge:     {(corners / edges).mean():.2f}x")
        print(f"  edge/interior:   {(edges / interiors).mean():.2f}x")
        print(f"  corner/interior: {(corners / interiors).mean():.2f}x")


if __name__ == "__main__":
    main()

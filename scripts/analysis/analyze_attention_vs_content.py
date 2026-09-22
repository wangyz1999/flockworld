"""Does the edge/corner attention hotspot exceed what real boid density there
would justify? For each (episode, query_agent), compares:

  - ring_heaviness:        attention mass on the outer 8x8-grid ring vs interior
  - boid_density_ring_ratio: REAL boid count (from GT parquet positions) on the
                             same ring vs interior, at the matching context frame

If attention's ratio is well above the content ratio, the model is looking at
edges more than boid density alone explains -- a genuine edge/boundary effect
(e.g. boids visually clipped/distorted as they exit the crop), not just "more
boids happen to be near edges."

    uv run python -m scripts.analysis.analyze_attention_vs_content
"""
from pathlib import Path

import numpy as np
import torch

from modeling.eval.attention_probe import (
    ROOT, boid_density_ring_ratio, extract_dist, load_model_and_data, ring_heaviness,
)

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
        attn_ratios, content_ratios = [], []
        for e in EPISODES:
            for qp in QUERY_AGENTS:
                dist, ep, ctx, f, p, s, h, w = extract_dist(model, cfg, ds, e, qp, SIGMA, device)
                per_agent_ctx = dist[:ctx].sum(dim=0)  # (p, 64)
                rh = ring_heaviness(per_agent_ctx, qp)
                parquet_path = Path(ROOT) / "state_action" / f"{ep['episode_id']}.parquet"
                bd = boid_density_ring_ratio(ep, ctx, ep["agent_indices"], qp, parquet_path)
                attn_ratios.append(rh); content_ratios.append(bd)
                excess = rh / (bd + 1e-9)
                print(f"  ep{ep['episode_id']:>5} q{qp}: attn_ring={rh:5.2f}x  "
                      f"content_ring={bd:5.2f}x  excess={excess:5.2f}x")
        attn_ratios, content_ratios = np.array(attn_ratios), np.array(content_ratios)
        print(f"  -- {name} aggregate over {len(attn_ratios)} runs --")
        print(f"  attn_ring:    {attn_ratios.mean():5.2f}x +/- {attn_ratios.std():.2f}")
        print(f"  content_ring: {content_ratios.mean():5.2f}x +/- {content_ratios.std():.2f}")
        print(f"  excess (attn / content): {(attn_ratios / content_ratios).mean():5.2f}x")


if __name__ == "__main__":
    main()

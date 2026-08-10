"""Reshape compare_experiments.py's combined --save-json output into the one-file-per-
experiment format plot_eval_sweep.py expects under results-dir/streaming_arch/.

compare_experiments.py evaluates every streaming-arch experiment (+ceiling +floor +any
extra informational column) in ONE process, sharing the held-out episodes, and writes
ONE json with a "columns" dict keyed by column name (ceiling/floor/exp01_baseline/...).
plot_eval_sweep.py (written against eval_flock_multi.py's per-experiment --save-json)
instead expects one file per experiment, each with columns exactly {ceiling, model,
baseline}. This splits the former into the latter, so the existing plotting/table code
runs unmodified. The combined file (and the per-episode CSV next to it) stays as the
source of truth; these are just a compatibility view.

Usage:
  uv run python jobs/eval_sweep_20260809/adapt_streaming_arch_json.py \
    eval_results/20260809_full_fidelity/results/streaming_arch/comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _nanmean(vals):
    arr = np.asarray([v for v in vals if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _augment(col_summary: dict, per_episode_col: dict) -> dict:
    """compare_experiments.py's own --save-json summary only aggregates the console-table
    keys (TIER_A_KEYS/CONSISTENCY_KEYS/PIXEL_KEYS); backfill 'volume' + sightings_per_frame
    /white_count_error from the raw per-episode records (also in comparison.json) so this
    matches eval_flock_multi.py's schema (plot_eval_sweep.py's VOLUME_METRICS rows)."""
    consistency = per_episode_col.get("consistency", [])
    pixel = per_episode_col.get("pixel", [])
    out = {**col_summary}
    out["consistency"] = {
        **out.get("consistency", {}),
        "sightings_per_frame": _nanmean([d.get("sightings_per_frame") for d in consistency]),
        "white_count_error": _nanmean([d.get("white_count_error") for d in consistency]),
    }
    out["volume"] = {
        "n_sightings": int(sum(d.get("n_sightings", 0) for d in consistency)),
        "n_reciprocal": int(sum(d.get("n_reciprocal", 0) for d in consistency)),
        "n_matched_white": int(sum(d.get("n_matched_white", 0) for d in consistency)),
        "n_overlap_white": int(sum(d.get("n_overlap_white", 0) for d in consistency)),
        "n_pixel_events": int(sum(d.get("n_pixel_events", 0) for d in pixel)),
    }
    return out


def main():
    src = Path(sys.argv[1])
    comparison = json.loads(src.read_text())
    out_dir = src.parent
    cols = comparison["columns"]
    per_episode = comparison.get("per_episode", {})
    ceiling = _augment(cols["ceiling"], per_episode.get("ceiling", {})) if "ceiling" in cols else None
    floor = _augment(cols["floor"], per_episode.get("floor", {})) if "floor" in cols else None
    extra = comparison.get("extra_single_agent")
    model_keys = list(comparison["include"])
    if extra and extra.get("name"):
        model_keys.append(extra["name"])

    written = []
    for key in model_keys:
        if key not in cols:
            continue
        entry = {
            "experiment": key,
            "config": None,
            "output_dir": None,
            "num_agents": 10,
            "n_ep": comparison["n_ep"],
            "eval_args": comparison["eval_args"],
            "columns": {
                "ceiling": ceiling,
                "model": _augment(cols[key], per_episode.get(key, {})),
                **({"baseline": floor} if floor is not None else {}),
            },
            "source_comparison_json": str(src),
        }
        out_path = out_dir / f"{key}.json"
        out_path.write_text(json.dumps(entry, indent=2))
        written.append(out_path)
    print(f"wrote {len(written)} per-experiment json(s) -> {out_dir}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()

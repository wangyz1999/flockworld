"""Plots and markdown tables over a directory of eval_flock_multi --save-json outputs.

Reads three optional group subdirectories under --results-dir:

  agentcount/*.json     one file per num_agents value -- line plots vs. camera-agent count
  streaming_arch/*.json one file per streaming-architecture experiment -- bar charts
  color_bg/*.json       single gradient-background run, no baseline column -- table only

Any group directory that doesn't exist is skipped, not an error (e.g. only
agentcount/ may be populated so far). No GPU/torch dependency -- reads JSON, writes
PNG + markdown, safe to run on a login node.

Usage:
  uv run python plot_eval_sweep.py --results-dir <dir> --plots-dir <dir> --tables-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_CEILING = "#2a78d6"
COLOR_MODEL = "#eb6834"
COLOR_BASELINE = "#1baf7a"
COLOR_MODEL_BAR = "#2a78d6"   # bars in streaming_arch charts: sequential hue, not identity -- see task spec
COLOR_REF_CEILING = "#898781"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_AXIS = "#52514e"
COLOR_GRID = "#e1e0d9"

METRICS = ["detection_rate", "position_error", "reciprocity_rate", "displacement_error", "psnr", "ssim"]
METRIC_SECTION = {
    "detection_rate": "tier_a", "position_error": "tier_a",
    "reciprocity_rate": "consistency", "displacement_error": "consistency",
    "psnr": "pixel", "ssim": "pixel",
}
METRIC_TITLE = {
    "detection_rate": "Detection rate", "position_error": "Position error",
    "reciprocity_rate": "Reciprocity rate", "displacement_error": "Displacement error",
    "psnr": "PSNR", "ssim": "SSIM",
}
METRIC_UNIT = {
    "detection_rate": "rate 0-1", "position_error": "px",
    "reciprocity_rate": "rate 0-1", "displacement_error": "px",
    "psnr": "dB", "ssim": "rate 0-1",
}
# labels mirror compare_experiments.py's console-table convention (metric name + direction)
QUALITY_LABEL = {
    "detection_rate": "Detection rate (up)", "position_error": "Position error px (down)",
    "reciprocity_rate": "Reciprocity rate (up)", "displacement_error": "Displacement error px (down)",
    "psnr": "PSNR dB (up)", "ssim": "SSIM (up)",
}
# volume/context: never imply "higher is better" -- these are event counts / rates that
# scale with how much the rollout has to report on, not with fidelity.
VOLUME_METRICS = [
    ("mean_detected_per_frame", "tier_a", "Detections / frame (volume)"),
    ("mean_gt_visible_per_frame", "tier_a", "GT boids visible / frame (volume)"),
    ("sightings_per_frame", "consistency", "Sightings / pair-frame (volume)"),
    ("n_sightings", "volume", "n_sightings (volume)"),
    ("n_reciprocal", "volume", "n_reciprocal (volume)"),
    ("n_matched_white", "volume", "n_matched_white (volume)"),
    ("n_overlap_white", "volume", "n_overlap_white (volume)"),
    ("n_pixel_events", "volume", "n_pixel_events (volume)"),
]

STREAMING_ORDER = [
    "exp01_baseline", "exp02_tiled", "exp03_diffusion_forcing", "exp04_two_stage",
    "exp06_tiled_df", "exp07_tiled_df_two_stage", "exp08_tiled_two_stage",
]

COLUMN_ORDER = ["ceiling", "model", "baseline"]
COLUMN_STYLE = {
    "ceiling": (COLOR_CEILING, "ceiling (GT decoded)"),
    "model": (COLOR_MODEL, "model"),
    "baseline": (COLOR_BASELINE, "baseline (single-agent)"),
}


def load_entries(group_dir: Path) -> list[dict]:
    entries = []
    for p in sorted(group_dir.glob("*.json")):
        try:
            with open(p) as f:
                entries.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {p}: {e}")
    return entries


def get(entry: dict, column: str, section: str, key: str) -> float:
    col = (entry.get("columns") or {}).get(column)
    if col is None:
        return float("nan")
    val = (col.get(section) or {}).get(key)
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def nanmean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def fmt(v, is_int: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{int(round(v))}" if is_int else f"{v:.3f}"


def group_columns(entries: list[dict]) -> list[str]:
    return [c for c in COLUMN_ORDER if any(c in (e.get("columns") or {}) for e in entries)]


def new_fig():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    return fig, ax


def style_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_AXIS, labelsize=9)
    ax.set_title(title, color=COLOR_TEXT, fontsize=11)
    ax.set_xlabel(xlabel, color=COLOR_AXIS, fontsize=9)
    ax.set_ylabel(ylabel, color=COLOR_AXIS, fontsize=9)


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agentcount(entries: list[dict], plots_dir: Path):
    entries = sorted(entries, key=lambda e: e["num_agents"])
    xs = [e["num_agents"] for e in entries]
    for metric in METRICS:
        section = METRIC_SECTION[metric]
        fig, ax = new_fig()
        drawn = False
        for col in COLUMN_ORDER:
            ys = [get(e, col, section, metric) for e in entries]
            if all(math.isnan(y) for y in ys):
                continue
            drawn = True
            color, label = COLUMN_STYLE[col]
            # NaN in the y array is enough -- matplotlib breaks the line at that point
            # and reconnects around it, so a single missing checkpoint doesn't sever
            # the whole series.
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label, zorder=3)
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        style_axes(ax, f"{METRIC_TITLE[metric]} vs. camera-agent count",
                   "Camera agents (count)", f"{METRIC_TITLE[metric]} ({METRIC_UNIT[metric]})")
        ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_TEXT)
        savefig(fig, plots_dir / f"agentcount_{metric}.png")


def plot_streaming_arch(entries: list[dict], plots_dir: Path):
    by_name = {e["experiment"]: e for e in entries}
    names = [n for n in STREAMING_ORDER if n in by_name]
    if not names:
        return
    xs = np.arange(len(names))
    for metric in METRICS:
        section = METRIC_SECTION[metric]
        model_vals = np.array([get(by_name[n], "model", section, metric) for n in names])
        if np.all(np.isnan(model_vals)):
            continue
        # each experiment's own ceiling/floor are estimates of the SAME underlying
        # quantity (shared held-out episodes, shared floor model) -- averaging across
        # experiments smooths eval-episode sampling noise, not a real per-experiment effect.
        ceiling_mean = nanmean([get(by_name[n], "ceiling", section, metric) for n in names])
        baseline_mean = nanmean([get(by_name[n], "baseline", section, metric) for n in names])

        fig, ax = new_fig()
        valid = ~np.isnan(model_vals)
        ax.bar(xs[valid], model_vals[valid], color=COLOR_MODEL_BAR, width=0.6, zorder=3, label="model")
        for i in np.where(~valid)[0]:
            ax.text(xs[i], 0, "–", ha="center", va="bottom", color=COLOR_AXIS)

        if not math.isnan(ceiling_mean):
            ax.axhline(ceiling_mean, color=COLOR_REF_CEILING, linestyle="--", linewidth=2,
                       label="ceiling (GT decoded)", zorder=4)
        if not math.isnan(baseline_mean):
            ax.axhline(baseline_mean, color=COLOR_BASELINE, linestyle=":", linewidth=2,
                       label="floor (single-agent)", zorder=4)

        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        style_axes(ax, f"{METRIC_TITLE[metric]} across streaming architectures",
                   "", f"{METRIC_TITLE[metric]} ({METRIC_UNIT[metric]})")
        # headroom so the legend doesn't sit on top of the ceiling reference line
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.18)
        ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_TEXT, loc="upper right")
        savefig(fig, plots_dir / f"streaming_arch_{metric}.png")


def make_md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def entry_block(entry: dict, cols: list[str], label: str) -> str:
    header = ["metric"] + cols
    quality_rows = [
        [QUALITY_LABEL[k]] + [fmt(get(entry, c, METRIC_SECTION[k], k)) for c in cols]
        for k in METRICS
    ]
    volume_rows = [
        [label3] + [fmt(get(entry, c, section, key), is_int=(section == "volume")) for c in cols]
        for key, section, label3 in VOLUME_METRICS
    ]
    return (
        f"### {label}\n\n"
        f"**Quality metrics**\n\n{make_md_table(header, quality_rows)}\n\n"
        f"**Volume / context (not a quality metric)**\n\n{make_md_table(header, volume_rows)}\n"
    )


def write_table(entries: list[dict], out_path: Path, title: str, label_fn, sort_key=None, fixed_order=None):
    if fixed_order is not None:
        by_name = {e["experiment"]: e for e in entries}
        entries = [by_name[n] for n in fixed_order if n in by_name]
    elif sort_key is not None:
        entries = sorted(entries, key=sort_key)
    cols = group_columns(entries)
    blocks = [entry_block(e, cols, label_fn(e)) for e in entries]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# {title}\n\n" + "\n".join(blocks))
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plots and markdown tables over eval_flock_multi JSON sweep results.")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--plots-dir", required=True)
    ap.add_argument("--tables-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    tables_dir = Path(args.tables_dir)

    agentcount_dir = results_dir / "agentcount"
    if agentcount_dir.is_dir():
        entries = load_entries(agentcount_dir)
        if entries:
            print(f"agentcount: {len(entries)} file(s)")
            plot_agentcount(entries, plots_dir)
            write_table(entries, tables_dir / "agentcount.md", "agentcount",
                        lambda e: f"num_agents = {e['num_agents']}", sort_key=lambda e: e["num_agents"])

    streaming_dir = results_dir / "streaming_arch"
    if streaming_dir.is_dir():
        entries = load_entries(streaming_dir)
        if entries:
            print(f"streaming_arch: {len(entries)} file(s)")
            plot_streaming_arch(entries, plots_dir)
            write_table(entries, tables_dir / "streaming_arch.md", "streaming_arch",
                        lambda e: e["experiment"], fixed_order=STREAMING_ORDER)

    colorbg_dir = results_dir / "color_bg"
    if colorbg_dir.is_dir():
        entries = load_entries(colorbg_dir)
        if entries:
            print(f"color_bg: {len(entries)} file(s) (table only -- n=1 is not a chart)")
            write_table(entries, tables_dir / "color_bg.md", "color_bg",
                        lambda e: e["experiment"], sort_key=lambda e: e["experiment"])


if __name__ == "__main__":
    main()

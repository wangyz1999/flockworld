"""Compose the teaser figure: one arena, ten egocentric views, one model.

The left panel is the global arena of a held-out evaluation episode, regenerated
from the simulator with short motion trails, with a coloured box marking each
camera agent's 128px crop. The right panel is that same instant as the world
model generated it -- the ten views tiled, each bordered in its agent's own hue,
so a box on the left and a tile on the right that share a colour are the same
agent.

Both panels show the same frame of the same episode. Evaluation episode ids are
the simulator seed in hex (``stream_eval_<seed:08x>``), and the streamed episode
warms up 60 steps and steps once more before rendering its first frame, so
video frame ``t`` is simulator step ``t + 61``. This offset is verified by
correlating regenerated crops against the ceiling video, not assumed.

Usage
-----
    python gen_hero_teaser.py                                  # the README teaser
    python gen_hero_teaser.py frame=210                        # a different instant
    python gen_hero_teaser.py video=docs/media/rollout_floor.mp4
    python gen_hero_teaser.py trail=0 out=output/figures/teaser_flat
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console

from flockworld.runtime import configure_jax_platform
from flockworld.utils import load_config

console = Console()

SCRIPT_KEYS = ("episode", "frame", "video", "out", "trail", "decay", "dpi", "grid")

# Streamed episodes warm up 60 steps, then the chunk steps once before its first
# render. Verified against the ceiling video by cross-correlation (r > 0.97).
SIM_OFFSET = 61

CAPTION_L = "One arena: 100 boids, 10 carrying cameras"
CAPTION_R = "Ten egocentric views, generated jointly by one model"
SUBCAPTION = ("Each box marks a camera agent's 128px crop; the tile bordered in the same colour is "
              "that agent's generated view of this instant.")
QUESTION = "The model never sees the arena — only the ten views. Can it imagine the whole flock?"

# Point sizes. The right title carries a much wider panel than the left, so it
# is set larger to read at the same weight.
FS_TITLE_L = 12.0
FS_TITLE_R = 15.0
FS_QUESTION = 16.0
FS_SUBCAPTION = 12.0
FS_ARROW = 13.5

# First installed family wins. The tail is the fallback for machines without the
# Windows UI fonts, so the figure still renders sanely on the cluster.
FONT_STACK = ["Segoe UI", "Inter", "Source Sans Pro", "Open Sans", "DejaVu Sans"]


def agent_hue(k: int, p: int) -> tuple[float, float, float]:
    """The renderer's own identity colour for camera agent ``k``."""
    return colorsys.hsv_to_rgb(k / p, 1.0, 1.0)


def read_video_frame(path: Path, index: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {path}")
    return frame[..., ::-1].copy()


def main():
    cli = list(sys.argv[1:])
    extra = OmegaConf.from_dotlist([a for a in cli if a.split("=")[0] in SCRIPT_KEYS])
    cfg = load_config([a for a in cli if a.split("=")[0] not in SCRIPT_KEYS])
    configure_jax_platform(cfg.get("device", "auto"))

    import jax
    from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
    from flockworld.rendering.renderer import build_uv_grid

    episode = str(extra.get("episode", "65e69251"))
    frame_idx = int(extra.get("frame", 240))
    video = Path(extra.get("video", "docs/media/rollout_flockworld_df.mp4"))
    out_stem = Path(extra.get("out", "output/figures/teaser"))
    trail = int(extra.get("trail", 24))
    decay = float(extra.get("decay", 0.86))
    dpi = int(extra.get("dpi", 300))
    grid_shape = tuple(int(v) for v in str(extra.get("grid", "2,5")).split(","))
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    seed = int(episode, 16)
    sim_step = frame_idx + SIM_OFFSET

    # The reported campaign's condition: identity-coloured agents on black.
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([
        "rendering.background_gradient=false",
        "rendering.color_mode=agent_id",
    ]))
    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)
    uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)
    crop = int(cfg.canvas.partial)

    rows, cols = grid_shape
    n_cam = rows * cols

    console.print(
        f"[bold]Teaser[/bold] | episode {episode} (seed 0x{seed:08x}) | "
        f"video frame {frame_idx} = sim step {sim_step} | {n_cam} camera agents | "
        f"trail {trail} frames"
    )

    state = reset(jax.random.PRNGKey(seed), ec)
    zero = jax.numpy.float32(0.0)
    for _ in range(sim_step - trail):
        state, _, _, _ = step(state, zero, params)

    # Short exposure into the featured frame, so the flock reads as moving.
    acc = np.zeros((ec.canvas_h, ec.canvas_w, 3), dtype=np.float32)
    for _ in range(max(trail, 1)):
        state, _, _, _ = step(state, zero, params)
        frame = np.asarray(render(state, params, uv_grid), dtype=np.float32)
        acc = np.maximum(acc * decay, frame)
    # Dim the exposure and lay the featured frame back on top at full strength,
    # so the trails stay a hint of motion and the agents keep their colours.
    arena = (np.clip(np.maximum(acc * 0.42, frame), 0, 1) * 255).astype(np.uint8)
    positions = np.asarray(state.boids.positions)[:n_cam]

    tiles = read_video_frame(video, frame_idx)

    _compose(arena, positions, tiles, crop, rows, cols, out_stem, dpi)
    console.print(f"wrote [bold]{out_stem.with_suffix('.png')}[/bold]")


def _compose(arena, positions, tiles, crop, rows, cols, out_stem, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    stack = os.environ.get("TEASER_FONT")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ([stack] if stack else []) + FONT_STACK

    n_cam = rows * cols
    h_arena = arena.shape[0]
    th, tw = tiles.shape[:2]

    # One row: arena, arrow, tiles. Widths in figure units track pixel aspect so
    # neither panel is stretched.
    fig_w, fig_h = 13.0, 5.6
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    ax_a = fig.add_axes([0.015, 0.12, 0.315, 0.80])
    ax_m = fig.add_axes([0.345, 0.12, 0.055, 0.80])
    ax_t = fig.add_axes([0.405, 0.12, 0.585, 0.80])
    for ax in (ax_a, ax_m, ax_t):
        ax.set_facecolor("black")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax_a.imshow(arena, interpolation="nearest")
    for k, (x, y) in enumerate(positions):
        color = agent_hue(k, n_cam)
        ax_a.add_patch(Rectangle(
            (x - crop / 2, y - crop / 2), crop, crop,
            fill=False, edgecolor=color, linewidth=1.4, alpha=0.95,
        ))
    ax_a.set_xlim(0, arena.shape[1]), ax_a.set_ylim(h_arena, 0)

    ax_m.axis("off")
    ax_m.annotate(
        "", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="white", linewidth=1.8,
                        mutation_scale=18),
    )
    ax_m.text(0.5, 0.545, "FlockWorld", color="white", fontsize=FS_ARROW,
              ha="center", va="bottom", transform=ax_m.transAxes)

    ax_t.imshow(tiles, interpolation="nearest")
    # Tile geometry: equal gutters between tiles and around the border.
    tile_h = (th - (rows + 1) * _gutter(th, rows)) / rows
    tile_w = (tw - (cols + 1) * _gutter(tw, cols)) / cols
    gy, gx = _gutter(th, rows), _gutter(tw, cols)
    for k in range(n_cam):
        r, c = divmod(k, cols)
        x0 = gx + c * (tile_w + gx)
        y0 = gy + r * (tile_h + gy)
        ax_t.add_patch(Rectangle(
            (x0 - 0.5, y0 - 0.5), tile_w, tile_h,
            fill=False, edgecolor=agent_hue(k, n_cam), linewidth=1.8,
        ))
    ax_t.set_xlim(0, tw), ax_t.set_ylim(th, 0)

    # Titles as figure text, so both sit on one baseline regardless of how each
    # image letterboxes inside its axes.
    for ax, caption, size in ((ax_a, CAPTION_L, FS_TITLE_L), (ax_t, CAPTION_R, FS_TITLE_R)):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.945, caption, color="white",
                 fontsize=size, ha="center", va="center")
    fig.text(0.5, 0.072, QUESTION, color="white", fontsize=FS_QUESTION,
             ha="center", va="center")
    fig.text(0.5, 0.024, SUBCAPTION, color="#9a9a9a", fontsize=FS_SUBCAPTION,
             ha="center", va="center")

    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, facecolor="black",
                pad_inches=0.0)
    plt.close(fig)


def _gutter(total: int, n: int) -> float:
    """Recover the gutter width the tiler used, assuming equal inner/outer gutters."""
    # total = n * tile + (n + 1) * gutter, with tile a multiple of 2 (2x upscaled 128).
    for g in range(1, 8):
        tile = (total - (n + 1) * g) / n
        if abs(tile - round(tile)) < 1e-6 and round(tile) % 2 == 0:
            return float(g)
    return 2.0


if __name__ == "__main__":
    main()

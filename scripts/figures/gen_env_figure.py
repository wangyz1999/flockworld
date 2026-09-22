"""Compose the paper figure for the FlockWorld observation conditions.

Renders one frozen simulation state under all four background/agent-color
combinations, tiles them 2x2 with row/column headers, and pairs each condition
with the 128px egocentric crop the model actually consumes. The top-left panel
carries the two otherwise-invisible parameters (vision radius, crop extent).

Boid geometry is identical across all panels by construction: one EnvState is
rendered through four EnvParams, and rendering never touches dynamics.

Usage
-----
    python -m scripts.figures.gen_env_figure step=240 focal=9         # the paper figure
    python -m scripts.figures.gen_env_figure frustum=none             # crop boxes, no callout lines
    python -m scripts.figures.gen_env_figure mark_future_work=true    # tag the gradient row as untrained
    python -m scripts.figures.gen_env_figure out=paper/figures/env_conditions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console

from flockworld.runtime import configure_jax_platform
from flockworld.utils import load_config, _video_warmup_steps
from flockworld.video.recorder import crop_partial

console = Console()

SCRIPT_KEYS = ("step", "focal", "out", "mark_future_work", "dpi", "frustum", "omit")

# (key, background_gradient, color_mode)
CONDITIONS = [
    ("plain",          False, "fixed"),
    ("color",          False, "agent_id"),
    ("gradient",       True,  "fixed"),
    ("gradient_color", True,  "agent_id"),
]

COL_HEADERS = ["uniform agents", "identity-coloured agents"]
ROW_HEADERS = ["black background", "gradient background"]

# ── figure geometry, in inches ───────────────────────────────────────────
# Each condition is a pair: the global panel and, immediately right of it, the
# crop the model sees. Adjacency carries the correspondence, so no leader line
# has to cross the figure.
PANEL = 1.90        # global panel edge
POV = 1.18          # POV crop edge
PAIR_GAP = 0.045    # global -> its own POV
COL_GAP = 0.24      # pair -> pair
ROW_GAP = 0.07      # gutter between rows; walls supply the rest of the seam
ROW_LAB = 0.34      # left strip for rotated row headers
COL_HDR = 0.46      # top strip: column header + per-panel mini-labels
FOOT = 0.12         # bottom margin; grows if the future-work tag is switched on
EDGE = 0.08         # right margin

ANNOT = "#FFFFFF"


def main():
    cli = list(sys.argv[1:])
    extra = OmegaConf.from_dotlist([a for a in cli if a.split("=")[0] in SCRIPT_KEYS])
    cfg = load_config([a for a in cli if a.split("=")[0] not in SCRIPT_KEYS])
    configure_jax_platform(cfg.get("device", "auto"))

    import jax
    from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
    from flockworld.rendering.renderer import build_uv_grid

    step_index = int(extra.get("step", 660))
    out_stem = Path(extra.get("out", "output/figures/env_conditions"))
    mark_future = bool(extra.get("mark_future_work", False))
    dpi = int(extra.get("dpi", 400))
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)
    uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)

    total_steps = _video_warmup_steps(cfg) + step_index
    console.print(
        f"[bold]Figure[/bold] | seed={cfg.seed} | {ec.num_agents} agents | "
        f"advancing {total_steps} steps"
    )

    state = reset(jax.random.PRNGKey(cfg.seed), ec)
    dummy_action = jax.numpy.float32(0.0)
    for _ in range(total_steps):
        state, _, _, _ = step(state, dummy_action, params)

    positions = np.asarray(state.boids.positions)
    focal = (
        int(extra["focal"]) if extra.get("focal", None) is not None
        else _pick_focal(positions, ec)
    )
    focal_pos = positions[focal]
    console.print(
        f"focal agent [bold]{focal}[/bold] at "
        f"({focal_pos[0]:.0f}, {focal_pos[1]:.0f}) | crop {cfg.canvas.partial}px | "
        f"vision {ec.vision:.0f}px"
    )

    globals_, povs = {}, {}
    for name, gradient, color_mode in CONDITIONS:
        variant_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([
            f"rendering.background_gradient={str(gradient).lower()}",
            f"rendering.color_mode={color_mode}",
        ]))
        variant_params = EnvParams(env_config_from_omega(variant_cfg))
        frame = np.asarray(render(state, variant_params, uv_grid))
        frame_u8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
        globals_[name] = frame_u8
        povs[name] = crop_partial(frame_u8, focal_pos, int(cfg.canvas.partial))

    omit = tuple(
        k.strip() for k in str(extra.get("omit", "")).split(",") if k.strip()
    )
    unknown = [k for k in omit if k not in {c[0] for c in CONDITIONS}]
    if unknown:
        raise ValueError(f"omit={unknown} is not a condition key; pick from "
                         f"{[c[0] for c in CONDITIONS]}")
    if omit:
        console.print(f"omitting condition(s): [bold]{', '.join(omit)}[/bold]")

    _compose(
        globals_, povs, focal_pos,
        vision=float(ec.vision),
        crop=int(cfg.canvas.partial),
        out_stem=out_stem,
        mark_future=mark_future,
        dpi=dpi,
        frustum=str(extra.get("frustum", "all")),
        omit=omit,
    )
    _write_caption(out_stem, cfg, ec, focal, mark_future, omit)


def _pick_focal(positions: np.ndarray, ec) -> int:
    """Pick the camera agent with the busiest neighbourhood, away from the walls.

    A focal agent hugging a wall or sitting alone gives a POV crop that shows
    nothing about the flock, which is the one thing the crop is there to show.
    """
    n_cam = min(10, positions.shape[0])
    margin = 90.0
    best, best_count = 0, -1
    for i in range(n_cam):
        x, y = positions[i]
        if not (margin < x < ec.canvas_w - margin and margin < y < ec.canvas_h - margin):
            continue
        d = np.linalg.norm(positions - positions[i], axis=1)
        count = int(((d > 0) & (d < 90.0)).sum())
        if count > best_count:
            best, best_count = i, count
    return best


def _compose(globals_, povs, focal_pos, *, vision, crop, out_stem, mark_future, dpi,
             frustum="all", omit=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    foot = 0.36 if mark_future else FOOT
    pair_w = PANEL + PAIR_GAP + POV
    content_w = 2 * pair_w + COL_GAP
    fig_w = ROW_LAB + content_w + EDGE
    fig_h = foot + 2 * PANEL + ROW_GAP + COL_HDR

    fig = plt.figure(figsize=(fig_w, fig_h))

    def rect(x, y, w, h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    pair_x = [ROW_LAB, ROW_LAB + pair_w + COL_GAP]
    pov_x = [x + PANEL + PAIR_GAP for x in pair_x]
    row_y = [foot + PANEL + ROW_GAP, foot]            # top row first
    pov_dy = (PANEL - POV) / 2
    top = foot + 2 * PANEL + ROW_GAP

    order = [["plain", "color"], ["gradient", "gradient_color"]]

    for r, names in enumerate(order):
        for c, name in enumerate(names):
            if name in omit:
                continue          # condition not in the study; cell left blank
            ax = fig.add_axes(rect(pair_x[c], row_y[r], PANEL, PANEL))
            ax.imshow(globals_[name])
            ax.set_xticks([]); ax.set_yticks([])
            # The canvas has a white wall border, which would otherwise vanish
            # into the white page; a hairline frame gives it an edge to read against.
            for s in ax.spines.values():
                s.set_edgecolor("#AAAAAA")
                s.set_linewidth(0.5)

            # crop box on every panel: ties each global view to the POV beside it
            ax.add_patch(Rectangle(
                (focal_pos[0] - crop / 2, focal_pos[1] - crop / 2), crop, crop,
                fill=False, edgecolor=ANNOT, linewidth=0.6,
                alpha=1.0 if (r, c) == (0, 0) else 0.5,
            ))

            if (r, c) == (0, 0):
                ax.add_patch(Circle(
                    (focal_pos[0], focal_pos[1]), vision,
                    fill=False, edgecolor=ANNOT, linewidth=0.6,
                    linestyle=(0, (2.2, 1.8)),
                ))
                _annotate(ax, focal_pos, vision, crop, globals_[name].shape[1])

            # the crop the model consumes, immediately right of its own panel
            pax = fig.add_axes(rect(pov_x[c], row_y[r] + pov_dy, POV, POV))
            pax.imshow(povs[name], interpolation="nearest")
            pax.set_xticks([]); pax.set_yticks([])
            for s in pax.spines.values():
                s.set_edgecolor("#AAAAAA")
                s.set_linewidth(0.5)

            if frustum == "all" or (frustum == "top" and r == 0):
                _draw_frustum(
                    fig, Line2D, fig_w, fig_h,
                    gx=pair_x[c], gy=row_y[r],
                    px=pov_x[c], py=row_y[r] + pov_dy,
                    focal_pos=focal_pos, crop=crop,
                    canvas=globals_[name].shape[0],
                )

    # ── headers ──────────────────────────────────────────────────────
    for c, text in enumerate(COL_HEADERS):
        fig.text((pair_x[c] + pair_w / 2) / fig_w, (top + 0.235) / fig_h, text,
                 ha="center", va="bottom", fontsize=9)
        fig.text((pair_x[c] + PANEL / 2) / fig_w, (top + 0.055) / fig_h,
                 "global view", ha="center", va="bottom",
                 fontsize=7.6, color="#555555", style="italic")
        fig.text((pov_x[c] + POV / 2) / fig_w, (top + 0.055) / fig_h,
                 f"{crop}px agent POV", ha="center", va="bottom",
                 fontsize=7.6, color="#555555", style="italic")

    for r, text in enumerate(ROW_HEADERS):
        grey = mark_future and r == 1
        fig.text((ROW_LAB - 0.11) / fig_w, (row_y[r] + PANEL / 2) / fig_h, text,
                 ha="center", va="center", rotation=90, fontsize=9,
                 color="#6A6A6A" if grey else "black")

    # ── optional future-work marking on the gradient row ─────────────
    if mark_future:
        pad = 0.055
        fig.patches.append(Rectangle(
            ((ROW_LAB - 0.25) / fig_w, (foot - pad) / fig_h),
            (content_w + 0.25 + pad) / fig_w,
            (PANEL + 2 * pad) / fig_h,
            transform=fig.transFigure, fill=False,
            edgecolor="#A0A0A0", linewidth=0.6, linestyle=(0, (3, 2)),
        ))
        fig.text((ROW_LAB + content_w / 2) / fig_w, (foot - pad - 0.115) / fig_h,
                 "not trained \u2014 future work", ha="center", va="center",
                 fontsize=7, color="#666666")

    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    console.print(f"[green]Wrote[/green] {png}  ({fig_w:.2f}\u00d7{fig_h:.2f} in)")
    console.print(f"[green]Wrote[/green] {pdf}")


def _draw_frustum(fig, Line2D, fig_w, fig_h, *, gx, gy, px, py, focal_pos, crop, canvas):
    """Magnified-callout lines: crop-box corners out to the POV panel corners.

    Drawn in two collinear segments so each stays legible on its own ground —
    white while crossing the dark canvas, grey once out on the white page. The
    top-right corner goes to the POV's top-left and the bottom-right to its
    bottom-left, so the pair fans open and never crosses.
    """
    half = crop / 2

    def to_fig(dx, dy):
        return (gx + dx / canvas * PANEL, gy + PANEL - dy / canvas * PANEL)

    pairs = [
        (to_fig(focal_pos[0] + half, focal_pos[1] - half), (px, py + POV)),
        (to_fig(focal_pos[0] + half, focal_pos[1] + half), (px, py)),
    ]
    edge_x = gx + PANEL

    for (sx, sy), (ex, ey) in pairs:
        t = (edge_x - sx) / (ex - sx)
        mx, my = edge_x, sy + t * (ey - sy)
        for (ax0, ay0), (ax1, ay1), colour, alpha in (
            ((sx, sy), (mx, my), ANNOT, 0.75),
            ((mx, my), (ex, ey), "#B4B4B4", 1.0),
        ):
            fig.add_artist(Line2D(
                [ax0 / fig_w, ax1 / fig_w], [ay0 / fig_h, ay1 / fig_h],
                transform=fig.transFigure, color=colour, alpha=alpha,
                linewidth=0.5, zorder=5,
            ))


def _annotate(ax, focal_pos, vision, crop, canvas):
    """Label the vision radius and crop extent on the top-left panel only."""
    x, y = float(focal_pos[0]), float(focal_pos[1])
    half = crop / 2
    arrow = dict(arrowstyle="-", color=ANNOT, linewidth=0.5, shrinkA=1, shrinkB=1)

    # Labels go on whichever side of the box has room. With the callout lines
    # leaving the box's right corners, the left side is also the uncluttered one
    # whenever the focal agent sits right of centre.
    side = 1 if (canvas - (x + half)) > 170 else -1
    ha = "left" if side > 0 else "right"
    tx = x + side * (half + 26)

    ax.annotate(
        f"vision r = {vision:.0f} px",
        xy=(x + side * vision * 0.71, y - vision * 0.71),
        xytext=(tx, y - half - 14),
        arrowprops=arrow, color=ANNOT, fontsize=5.8, ha=ha, va="center",
    )
    ax.annotate(
        f"{crop} px crop",
        xy=(x + side * half, y + half),
        xytext=(tx, y + half + 20),
        arrowprops=arrow, color=ANNOT, fontsize=5.8, ha=ha, va="center",
    )


def _write_caption(out_stem: Path, cfg, ec, focal: int, mark_future: bool, omit=()):
    n_cam = min(int(cfg.collection.partial_agents), int(ec.num_agents))
    blank = (
        " The uniform-agent gradient cell is left blank: that combination is not "
        "one of the studied conditions."
        if "gradient" in omit else ""
    )
    future = (
        " Gradient conditions are rendered for comparison but are left to future "
        "work; all reported results use the black-background conditions."
        if mark_future else ""
    )
    caption = (
        "\\caption{\n"
        "  \\textbf{FlockWorld observation conditions.}\n"
        f"  {ec.num_agents} boids on a {ec.canvas_w}$\\times${ec.canvas_h} canvas with "
        "reflecting walls; every panel renders the \\emph{same} simulation state, so\n"
        "  agent positions and headings are identical and only the observation encoding\n"
        "  differs. \\textbf{Columns:} every agent rendered white (left) versus the\n"
        f"  {n_cam} camera agents given evenly spaced identity hues, the remaining\n"
        f"  {int(ec.num_agents) - n_cam} left white (right). \\textbf{{Rows:}} a black background (top)\n"
        "  versus a position-encoding background in which the red channel encodes $x/W$\n"
        f"  and the green channel $y/H$ at brightness {ec.gradient_brightness} (bottom), giving an agent\n"
        "  an absolute positional cue it cannot otherwise recover. \\textbf{Right:} the\n"
        f"  {cfg.canvas.partial}$\\times${cfg.canvas.partial} egocentric crop the model actually consumes, for the\n"
        f"  focal agent boxed in each panel; the model never observes the global view. The\n"
        f"  dashed circle marks the {ec.vision:.0f}\\,px vision radius that makes flocking local."
        f"{blank}{future}\n"
        "}\n"
        f"\\label{{fig:env-conditions}}\n"
    )
    path = out_stem.with_suffix(".caption.tex")
    path.write_text(caption)
    console.print(f"[green]Wrote[/green] {path}")


if __name__ == "__main__":
    main()

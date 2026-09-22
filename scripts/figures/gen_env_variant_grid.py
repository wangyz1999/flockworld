"""Render the same simulation state under all 4 background/color combinations.

One rollout, one state, four renders — so boid positions and headings are
pixel-identical across the four images and only the rendering differs.

Usage
-----
    python -m scripts.figures.gen_env_variant_grid                       # frame 300 after warmup
    python -m scripts.figures.gen_env_variant_grid step=500 seed=7       # pick a different moment
    python -m scripts.figures.gen_env_variant_grid out_dir=output/figs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console

from flockworld.runtime import configure_jax_platform
from flockworld.utils import load_config, _video_warmup_steps

console = Console()

VARIANTS = [
    ("plain",             False, "fixed"),
    ("color",             False, "agent_id"),
    ("gradient",          True,  "fixed"),
    ("gradient_color",    True,  "agent_id"),
]


def main():
    cli = list(sys.argv[1:])
    extra = OmegaConf.from_dotlist(
        [a for a in cli if a.split("=")[0] in ("step", "out_dir", "gutter")]
    )
    cfg = load_config(
        [a for a in cli if a.split("=")[0] not in ("step", "out_dir", "gutter")]
    )
    configure_jax_platform(cfg.get("device", "auto"))

    import jax
    from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
    from flockworld.rendering.renderer import build_uv_grid

    step_index = int(extra.get("step", 300))
    out_dir = Path(extra.get("out_dir", "output/env_variants"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dynamics are independent of the rendering config, so step once with the
    # base params and reuse the resulting state for every variant.
    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)
    uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)

    total_steps = _video_warmup_steps(cfg) + step_index
    console.print(
        f"[bold]Variant grid[/bold] | seed={cfg.seed} | {ec.num_agents} agents | "
        f"advancing {total_steps} steps (warmup {_video_warmup_steps(cfg)} + {step_index})"
    )

    state = reset(jax.random.PRNGKey(cfg.seed), ec)
    dummy_action = jax.numpy.float32(0.0)
    for _ in range(total_steps):
        state, _, _, _ = step(state, dummy_action, params)

    saved = []
    panels = {}
    for name, gradient, color_mode in VARIANTS:
        variant_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.from_dotlist([
                f"rendering.background_gradient={str(gradient).lower()}",
                f"rendering.color_mode={color_mode}",
            ]),
        )
        variant_params = EnvParams(env_config_from_omega(variant_cfg))
        frame = np.asarray(render(state, variant_params, uv_grid))
        frame_u8 = np.clip(frame * 255, 0, 255).astype(np.uint8)

        path = out_dir / f"{name}.png"
        _write_png(path, frame_u8)
        panels[name] = frame_u8
        saved.append(str(path))

    # 2x2 grid: left->right adds agent color, top->bottom adds the gradient.
    grid_path = out_dir / "grid_2x2.png"
    # gutter=0: the panels' own wall borders butt together and form the divider,
    # so the seam stays as thin as the environment's own boundary allows.
    _write_png(grid_path, _tile_2x2(panels, gutter=int(extra.get("gutter", 0))))
    saved.append(str(grid_path))

    console.print(f"[green]Done.[/green] Wrote:\n  " + "\n  ".join(saved))


def _tile_2x2(panels: dict, gutter: int) -> np.ndarray:
    """Tile the four variants with white gutters, matching the POV-figure style."""
    layout = [["plain", "color"], ["gradient", "gradient_color"]]
    h, w = panels["plain"].shape[:2]
    grid_h = h * 2 + gutter
    grid_w = w * 2 + gutter
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for row, names in enumerate(layout):
        for col, name in enumerate(names):
            y = row * (h + gutter)
            x = col * (w + gutter)
            grid[y:y + h, x:x + w] = panels[name]
    return grid


def _write_png(path: Path, frame_u8: np.ndarray):
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()

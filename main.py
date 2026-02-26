"""FlockWorld CLI — run the boid simulation and record videos.

Usage
-----
    python main.py                              # headless video recording
    python main.py env.render=true              # live window, no video
    python main.py boids.num_agents=100         # override via CLI
    python main.py video.duration=5 seed=123    # multiple overrides
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
from flockworld.rendering.renderer import build_uv_grid
from flockworld.video.recorder import VideoRecorder

console = Console()


def load_config(cli_args: list[str] | None = None):
    """Load ``config/default.yaml`` and merge CLI overrides."""
    base_path = Path(__file__).resolve().parent / "config" / "default.yaml"
    base_cfg = OmegaConf.load(base_path)
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg
    return cfg


def _warmup(state, params, uv_grid):
    """Run one step + render to trigger JIT compilation."""
    console.print("[dim]Compiling JAX kernels...[/dim]", end=" ")
    t0 = time.time()
    dummy_action = jnp.float32(0.0)
    state, _, _, _ = step(state, dummy_action, params)
    _ = render(state, params, uv_grid).block_until_ready()
    console.print(f"[dim]done in {time.time() - t0:.1f}s[/dim]")


def _run_render(cfg, ec, params, uv_grid, state):
    """Live window mode — display frames with cv2, no video written."""
    import cv2

    fps = cfg.video.fps
    frame_delay = max(1, int(1000 / fps))
    window_name = "FlockWorld"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    console.print("[bold green]Render mode[/bold green] — press Q or ESC to quit")

    while True:
        action = state.boids.headings[0]
        state, _, done, _ = step(state, action, params)

        frame_f = render(state, params, uv_grid)
        frame_u8 = np.clip(np.asarray(frame_f) * 255, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, frame_bgr)
        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if done:
            break

    cv2.destroyAllWindows()
    console.print("[green]Done.[/green]")


def _run_headless(cfg, ec, params, uv_grid, state):
    """Headless mode — record video to disk."""
    fps = cfg.video.fps
    duration = cfg.video.duration
    total_frames = int(fps * duration)

    full_path = None if cfg.video.partial_only else cfg.video.full_obs_path
    partial_path = None if cfg.video.full_obs_only else cfg.video.partial_obs_path

    with VideoRecorder(
        full_obs_path=full_path,
        partial_obs_path=partial_path,
        partial_obs_size=cfg.canvas.partial,
        fps=fps,
        canvas_w=ec.canvas_w,
        canvas_h=ec.canvas_h,
    ) as recorder:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.completed}/{task.total} frames"),
        ) as progress:
            task = progress.add_task("Recording", total=total_frames)

            for _ in range(total_frames):
                action = state.boids.headings[0]
                state, _, done, _ = step(state, action, params)

                frame_f = render(state, params, uv_grid)
                frame_u8 = np.clip(np.asarray(frame_f) * 255, 0, 255).astype(np.uint8)

                controlled_pos = np.asarray(state.boids.positions[0])
                recorder.record(frame_u8, controlled_pos)

                progress.advance(task)

                if done:
                    break

    saved = [p for p in (full_path, partial_path) if p is not None]
    console.print(f"[green]Done.[/green] Videos saved to: {', '.join(saved)}")


def main():
    cfg = load_config(sys.argv[1:])
    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)

    mode = "render" if cfg.env.render else "headless"
    console.print(
        f"[bold]FlockWorld[/bold]  |  {ec.num_agents} agents  |  "
        f"{ec.canvas_w}x{ec.canvas_h}  |  boundary={ec.boundary}  |  "
        f"mode={mode}  |  device={jax.devices()[0]}"
    )

    uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)

    key = jax.random.PRNGKey(cfg.seed)
    state = reset(key, ec)

    _warmup(state, params, uv_grid)
    state = reset(key, ec)

    if cfg.env.render:
        _run_render(cfg, ec, params, uv_grid, state)
    else:
        _run_headless(cfg, ec, params, uv_grid, state)


if __name__ == "__main__":
    main()

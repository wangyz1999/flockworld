"""FlockWorld CLI — run the boid simulation and record videos.

Usage
-----
    python main.py                              # defaults
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


def main():
    cfg = load_config(sys.argv[1:])
    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)

    console.print(
        f"[bold]FlockWorld[/bold]  |  {ec.num_agents} agents  |  "
        f"{ec.canvas_w}x{ec.canvas_h}  |  boundary={ec.boundary}  |  "
        f"device={jax.devices()[0]}"
    )

    uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)

    key = jax.random.PRNGKey(cfg.seed)
    state = reset(key, ec)

    fps = cfg.video.fps
    duration = cfg.video.duration
    total_frames = int(fps * duration)

    # Warmup JIT compilation
    console.print("[dim]Compiling JAX kernels...[/dim]", end=" ")
    t0 = time.time()
    dummy_action = jnp.float32(0.0)
    state, _, _, _ = step(state, dummy_action, params)
    _ = render(state, params, uv_grid).block_until_ready()
    console.print(f"[dim]done in {time.time() - t0:.1f}s[/dim]")

    # Re-reset after warmup
    state = reset(key, ec)

    with VideoRecorder(
        full_obs_path=cfg.video.full_obs_path,
        partial_obs_path=cfg.video.partial_obs_path,
        partial_obs_size=cfg.video.partial_obs_size,
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

            for i in range(total_frames):
                # Default policy: keep current heading (stays on device)
                action = state.boids.headings[0]

                state, reward, done, info = step(state, action, params)

                frame_f = render(state, params, uv_grid)
                frame_u8 = np.clip(np.asarray(frame_f) * 255, 0, 255).astype(np.uint8)

                controlled_pos = np.asarray(state.boids.positions[0])
                recorder.record(frame_u8, controlled_pos)

                progress.advance(task)

                if done:
                    break

    console.print(
        f"[green]Done.[/green] Videos saved to: "
        f"{cfg.video.full_obs_path}, {cfg.video.partial_obs_path}"
    )


if __name__ == "__main__":
    main()

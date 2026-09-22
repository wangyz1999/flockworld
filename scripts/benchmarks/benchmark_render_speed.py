"""Benchmark FlockWorld render throughput across CPU/GPU and agent counts.

Examples
--------
    python -m scripts.benchmarks.benchmark_render_speed --frames 120
    python -m scripts.benchmarks.benchmark_render_speed --devices cpu,gpu --agents 100,500,1500
    python -m scripts.benchmarks.benchmark_render_speed --frames 60 canvas.width=512 canvas.height=512
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "data_recording.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "benchmarks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("agent counts must be positive")
    return parsed


def _parse_csv_strings(value: str) -> list[str]:
    from flockworld.runtime import normalize_device

    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one device")
    try:
        return [normalize_device(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_config(config_path: Path, overrides: list[str]):
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _benchmark_worker(args: argparse.Namespace) -> None:
    # Keep JAX imports inside the worker after JAX_PLATFORM_NAME has been set by
    # the parent process.
    import jax
    import jax.numpy as jnp

    from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
    from flockworld.rendering.renderer import build_uv_grid

    results: list[dict[str, Any]] = []
    base_cfg = _load_config(args.config, args.overrides)

    for agents in args.worker_agents:
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.device = args.worker_device
        cfg.boids.num_agents = agents
        if args.include_step:
            cfg.env.max_steps = max(int(cfg.env.max_steps), args.frames + 2)

        ec = env_config_from_omega(cfg)
        params = EnvParams(ec)
        uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)
        key = jax.random.PRNGKey(int(cfg.seed))
        state = reset(key, ec)
        action = jnp.float32(0.0)

        compile_t0 = time.perf_counter()
        if args.include_step:
            state, _, _, _ = step(state, action, params)
        render(state, params, uv_grid).block_until_ready()
        compile_s = time.perf_counter() - compile_t0

        timed_t0 = time.perf_counter()
        for _ in range(args.frames):
            if args.include_step:
                state, _, _, _ = step(state, action, params)
            render(state, params, uv_grid).block_until_ready()
        elapsed_s = time.perf_counter() - timed_t0

        fps = args.frames / elapsed_s if elapsed_s > 0.0 else math.inf
        results.append(
            {
                "device": args.worker_device,
                "jax_backend": jax.default_backend(),
                "jax_device": str(jax.devices()[0]),
                "agents": agents,
                "frames": args.frames,
                "canvas_width": ec.canvas_w,
                "canvas_height": ec.canvas_h,
                "color_mode": ec.color_mode,
                "include_step": args.include_step,
                "compile_s": compile_s,
                "elapsed_s": elapsed_s,
                "fps": fps,
                "ms_per_frame": 1000.0 / fps if fps > 0.0 else math.inf,
            }
        )

        jax.clear_caches()

    print(json.dumps(results))


def _run_device_worker(
    device: str,
    agents: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str | None]:
    env = os.environ.copy()
    if device != "auto":
        platform = "gpu" if device in {"gpu", "cuda"} else device
        jax_platforms = "cuda" if platform == "gpu" else platform
        env["JAX_PLATFORM_NAME"] = platform
        env["JAX_PLATFORMS"] = jax_platforms
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-device",
        device,
        "--worker-agents",
        ",".join(str(item) for item in agents),
        "--frames",
        str(args.frames),
        "--config",
        str(args.config),
    ]
    if args.include_step:
        cmd.append("--include-step")
    cmd.extend(args.overrides)

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return [], message

    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return [], f"Worker for {device!r} did not return JSON: {exc}\n{completed.stdout}"


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "device",
        "jax_backend",
        "jax_device",
        "agents",
        "frames",
        "canvas_width",
        "canvas_height",
        "color_mode",
        "include_step",
        "compile_s",
        "elapsed_s",
        "fps",
        "ms_per_frame",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg_plot(path: Path, rows: list[dict[str, Any]], *, log_x: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"520\" />\n")
        return

    width, height = 960, 560
    left, right, top, bottom = 82, 32, 54, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    agents = sorted({int(row["agents"]) for row in rows})
    max_fps = max(float(row["fps"]) for row in rows)
    y_max = max(1.0, max_fps * 1.12)
    x_min, x_max = min(agents), max(agents)
    if x_min == x_max:
        x_min = max(0, x_min - 1)
        x_max += 1
    if log_x:
        log_x_min = math.log10(max(1, x_min))
        log_x_max = math.log10(max(1, x_max))
        if log_x_min == log_x_max:
            log_x_min = max(0.0, log_x_min - 0.1)
            log_x_max += 0.1

    def sx(agent_count: int) -> float:
        if log_x:
            value = math.log10(max(1, agent_count))
            return left + (value - log_x_min) / (log_x_max - log_x_min) * plot_w
        return left + (agent_count - x_min) / (x_max - x_min) * plot_w

    def sy(fps: float) -> float:
        return top + plot_h - (fps / y_max) * plot_h

    colors = {
        "cpu": "#2f6fbb",
        "gpu": "#d65f2e",
        "cuda": "#d65f2e",
    }
    devices = sorted({str(row["device"]) for row in rows})
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#333;stroke-width:1.4} .tick{font-size:12px} .label{font-size:14px;font-weight:600} .title{font-size:20px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="30">FlockWorld render speed{" (log x)" if log_x else ""}</text>',
    ]

    for i in range(6):
        fps = y_max * i / 5
        y = sy(fps)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{fps:.1f}</text>')

    for agent_count in agents:
        x = sx(agent_count)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle">{agent_count}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    x_label = "Agents (log scale)" if log_x else "Agents"
    lines.append(f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle">{x_label}</text>')
    lines.append(f'<text class="label" x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle">Frames per second</text>')

    legend_x = left + plot_w - 136
    legend_y = top + 12
    for idx, device in enumerate(devices):
        device_rows = sorted(
            [row for row in rows if row["device"] == device],
            key=lambda row: int(row["agents"]),
        )
        color = colors.get(device, "#4b7f52")
        points = " ".join(
            f'{sx(int(row["agents"])):.1f},{sy(float(row["fps"])):.1f}'
            for row in device_rows
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        for row in device_rows:
            x = sx(int(row["agents"]))
            y = sy(float(row["fps"]))
            label = f'{float(row["fps"]):.1f} fps'
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"><title>{_svg_escape(device)} {row["agents"]} agents: {label}</title></circle>')
        y = legend_y + idx * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="2.6"/>')
        lines.append(f'<text class="tick" x="{legend_x + 34}" y="{y + 4}">{_svg_escape(device.upper())}</text>')

    first = rows[0]
    subtitle = (
        f'{first["frames"]} timed frames, {first["canvas_width"]}x{first["canvas_height"]}, '
        f'include_step={first["include_step"]}'
    )
    lines.append(f'<text class="tick" x="{left}" y="{height - 6}">{_svg_escape(subtitle)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No benchmark results were produced.")
        return
    print("device  agents  fps     ms/frame  compile_s  backend")
    print("------  ------  ------  --------  ---------  -------")
    for row in sorted(rows, key=lambda item: (str(item["device"]), int(item["agents"]))):
        print(
            f'{row["device"]:<6}  '
            f'{row["agents"]:>6}  '
            f'{row["fps"]:>6.2f}  '
            f'{row["ms_per_frame"]:>8.2f}  '
            f'{row["compile_s"]:>9.2f}  '
            f'{row["jax_backend"]}'
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides, e.g. canvas.width=512")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a FlockWorld YAML config")
    parser.add_argument("--agents", type=_parse_csv_ints, default=[100, 250, 500, 1000, 1500], help="Comma-separated agent counts")
    parser.add_argument("--devices", type=_parse_csv_strings, default=["cpu", "gpu"], help="Comma-separated devices: cpu,gpu")
    parser.add_argument("--frames", type=int, default=120, help="Timed frames per device/count")
    parser.add_argument("--include-step", action="store_true", help="Time simulation step + render instead of render only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for benchmark outputs")
    parser.add_argument("--json", type=Path, default=None, help="JSON output path")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--plot", type=Path, default=None, help="SVG plot output path")
    parser.add_argument("--log-plot", type=Path, default=None, help="Log-x SVG plot output path")
    parser.add_argument("--worker-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-agents", type=_parse_csv_ints, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    if args.worker_device:
        if not args.worker_agents:
            parser.error("--worker-agents is required with --worker-device")
        _benchmark_worker(args)
        return

    output_dir = args.output_dir
    json_path = args.json or (output_dir / "render_speed.json")
    csv_path = args.csv or (output_dir / "render_speed.csv")
    plot_path = args.plot or (output_dir / "render_speed.svg")
    log_plot_path = args.log_plot or (output_dir / "render_speed_logx.svg")

    all_rows: list[dict[str, Any]] = []
    for device in args.devices:
        rows, error = _run_device_worker(device, args.agents, args)
        if error:
            print(f"Skipping {device}: {error}", file=sys.stderr)
            continue
        all_rows.extend(rows)

    all_rows.sort(key=lambda row: (str(row["device"]), int(row["agents"])))
    _write_json(json_path, all_rows)
    _write_csv(csv_path, all_rows)
    _write_svg_plot(plot_path, all_rows)
    _write_svg_plot(log_plot_path, all_rows, log_x=True)
    _print_summary(all_rows)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {log_plot_path}")


if __name__ == "__main__":
    main()

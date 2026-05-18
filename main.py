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

import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from flockworld.runtime import configure_jax_platform
from flockworld.utils import (
    load_config,
    _generation_num_envs,
    _generation_seeds,
    _video_warmup_steps,
    _output_paths_for_env,
    _trajectory_path_for_env,
    _trajectory_enabled,
    _video_recorder_cls,
    _tree_index,
    _make_headless_warmup_fn,
    _make_headless_multi_warmup_fn,
    _make_batched_step_fn,
    _make_headless_chunk_fn,
    _make_headless_multi_chunk_fn,
    _make_batched_step_render_fn,
    _trajectory_snapshot_np,
    _append_trajectory_chunk,
    _append_multi_trajectory_chunk,
    _append_multi_trajectory_frame,
    _record_multi_chunk,
    _save_trajectory,
)

console = Console()


def _warmup(state, params, uv_grid, jnp, step_fn, render_fn):
    """Run one step + render to trigger JIT compilation."""
    console.print("[dim]Compiling JAX kernels...[/dim]", end=" ")
    t0 = time.time()
    dummy_action = jnp.float32(0.0)
    state, _, _, _ = step_fn(state, dummy_action, params)
    _ = render_fn(state, params, uv_grid).block_until_ready()
    console.print(f"[dim]done in {time.time() - t0:.1f}s[/dim]")


def _run_render(cfg, ec, params, uv_grid, state, policy_fn, policy_state, step_fn, render_fn):
    """Live window mode — display frames with cv2, no video written."""
    import cv2

    fps = cfg.video.fps
    frame_delay = max(1, int(1000 / fps))
    window_name = "FlockWorld"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    console.print("[bold green]Render mode[/bold green] — press Q or ESC to quit")

    while True:
        action, policy_state = policy_fn(state, params, policy_state, policy_state["key"])
        state, _, done, _ = step_fn(state, action, params)

        frame_f = render_fn(state, params, uv_grid)
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


def _run_headless(
    cfg, ec, params, uv_grid, state, policy_fn, policy_state,
    step_fn, render_fn, recorder_cls, jax, jnp,
):
    """Headless mode — record video to disk."""
    fps = cfg.video.fps
    duration = cfg.video.duration
    total_frames = int(fps * duration)
    warmup_steps = min(_video_warmup_steps(cfg), ec.max_steps)
    max_frames = min(total_frames, max(0, ec.max_steps - warmup_steps))
    chunk_size = max(1, int(cfg.video.get("chunk_size", 1)))

    full_path = None if cfg.video.partial_only else cfg.video.full_obs_path
    partial_path = None if cfg.video.full_obs_only else cfg.video.partial_obs_path
    save_trajectory = _trajectory_enabled(cfg)
    trajectory_chunks = []
    can_batch = (not ec.controlled_agent) or cfg.env.agent_policy == "straight"
    batch_fn = (
        _make_headless_chunk_fn(jax, jnp, step_fn, render_fn)
        if chunk_size > 1 and can_batch
        else None
    )

    if chunk_size > 1 and not can_batch:
        console.print(
            "[yellow]video.chunk_size ignored for controlled non-straight policies; "
            "falling back to exact per-frame policy recording.[/yellow]"
        )

    state, policy_state = _run_headless_warmup(
        cfg, state, params, policy_fn, policy_state, step_fn, jax, jnp,
        warmup_steps, chunk_size, can_batch,
    )

    with recorder_cls(
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
            task = progress.add_task("Recording", total=max_frames)

            recorded = 0
            if batch_fn is not None:
                use_straight_policy = bool(ec.controlled_agent)
                while recorded < max_frames:
                    n = min(chunk_size, max_frames - recorded)
                    state, chunk_outputs = batch_fn(
                        state, params, uv_grid, n, use_straight_policy,
                    )
                    frames, controlled_positions, trajectory = chunk_outputs
                    if save_trajectory:
                        _append_trajectory_chunk(trajectory_chunks, trajectory, n)

                    if hasattr(recorder, "record_jax_chunk"):
                        recorder.record_jax_chunk(frames, controlled_positions, n)
                    else:
                        frames_np = np.asarray(frames)
                        positions_np = np.asarray(controlled_positions)
                        for frame_u8, controlled_pos in zip(frames_np[:n], positions_np[:n]):
                            recorder.record(frame_u8, controlled_pos)

                    recorded += n
                    progress.advance(task, n)
            else:
                while recorded < max_frames:
                    action, policy_state = policy_fn(
                        state, params, policy_state, policy_state["key"],
                    )
                    state, _, done, _ = step_fn(state, action, params)

                    frame_f = render_fn(state, params, uv_grid)
                    frame_u8 = np.clip(np.asarray(frame_f) * 255, 0, 255).astype(np.uint8)

                    controlled_pos = np.asarray(state.boids.positions[0])
                    recorder.record(frame_u8, controlled_pos)
                    if save_trajectory:
                        trajectory_chunks.append(_trajectory_snapshot_np(state))

                    recorded += 1
                    progress.advance(task)

                    if done:
                        break

    saved = [p for p in (full_path, partial_path) if p is not None]
    if save_trajectory:
        trajectory_path = _trajectory_path_for_env(cfg, env_index=0, seed=int(cfg.seed))
        _save_trajectory(cfg, trajectory_path, int(cfg.seed), trajectory_chunks)
        saved.append(trajectory_path)
    console.print(f"[green]Done.[/green] Videos saved to: {', '.join(saved)}")


def _run_headless_multi(
    cfg, ec, params, uv_grid, states, policy_fn, policy_states, seeds,
    step_fn, render_fn, recorder_cls, jax, jnp,
):
    """Headless mode for homogeneous batched generation across environments."""
    fps = cfg.video.fps
    duration = cfg.video.duration
    total_frames = int(fps * duration)
    warmup_steps = min(_video_warmup_steps(cfg), ec.max_steps)
    max_frames = min(total_frames, max(0, ec.max_steps - warmup_steps))
    chunk_size = max(1, int(cfg.video.get("chunk_size", 1)))
    num_envs = len(seeds)
    save_trajectory = _trajectory_enabled(cfg)
    trajectory_chunks = [[] for _ in range(num_envs)]

    can_chunk = (not ec.controlled_agent) or cfg.env.agent_policy == "straight"
    chunk_fn = (
        _make_headless_multi_chunk_fn(jax, jnp, step_fn, render_fn)
        if chunk_size > 1 and can_chunk
        else None
    )
    step_render_fn = (
        None
        if chunk_fn is not None
        else _make_batched_step_render_fn(jax, jnp, step_fn, render_fn)
    )

    if chunk_size > 1 and not can_chunk:
        console.print(
            "[yellow]video.chunk_size ignored for controlled non-straight policies; "
            "falling back to batched per-frame policy recording.[/yellow]"
        )

    states, policy_states = _run_headless_multi_warmup(
        cfg, states, params, policy_fn, policy_states, step_fn, jax, jnp,
        warmup_steps, chunk_size, can_chunk,
    )

    recorders = []
    saved_paths = []
    try:
        for env_index, seed in enumerate(seeds):
            full_path, partial_path = _output_paths_for_env(cfg, env_index, seed)
            saved_paths.extend([p for p in (full_path, partial_path) if p is not None])
            recorders.append(
                recorder_cls(
                    full_obs_path=full_path,
                    partial_obs_path=partial_path,
                    partial_obs_size=cfg.canvas.partial,
                    fps=fps,
                    canvas_w=ec.canvas_w,
                    canvas_h=ec.canvas_h,
                )
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.completed}/{task.total} env-frames"),
        ) as progress:
            task = progress.add_task(f"Recording {num_envs} envs", total=max_frames * num_envs)

            recorded = 0
            if chunk_fn is not None:
                use_straight_policy = bool(ec.controlled_agent)
                while recorded < max_frames:
                    n = min(chunk_size, max_frames - recorded)
                    states, chunk_outputs = chunk_fn(
                        states, params, uv_grid, n, use_straight_policy,
                    )
                    frames, controlled_positions, trajectory = chunk_outputs
                    if save_trajectory:
                        _append_multi_trajectory_chunk(trajectory_chunks, trajectory, n)
                    _record_multi_chunk(recorders, frames, controlled_positions, n)
                    recorded += n
                    progress.advance(task, n * num_envs)
            else:
                while recorded < max_frames:
                    actions = []
                    for env_index in range(num_envs):
                        state_i = _tree_index(jax, states, env_index)
                        action, policy_states[env_index] = policy_fn(
                            state_i,
                            params,
                            policy_states[env_index],
                            policy_states[env_index]["key"],
                        )
                        actions.append(action)

                    action_array = jnp.asarray(actions, dtype=jnp.float32)
                    states, frame_outputs = step_render_fn(states, action_array, params, uv_grid)
                    frames, controlled_positions, trajectory = frame_outputs
                    frames_np = np.asarray(frames)
                    positions_np = np.asarray(controlled_positions)
                    if save_trajectory:
                        _append_multi_trajectory_frame(trajectory_chunks, trajectory)
                    for env_index, recorder in enumerate(recorders):
                        recorder.record(frames_np[env_index], positions_np[env_index])

                    recorded += 1
                    progress.advance(task, num_envs)
    finally:
        for recorder in recorders:
            recorder.close()

    if save_trajectory:
        for env_index, seed in enumerate(seeds):
            trajectory_path = _trajectory_path_for_env(cfg, env_index, seed)
            _save_trajectory(cfg, trajectory_path, seed, trajectory_chunks[env_index])
            saved_paths.append(trajectory_path)

    console.print(f"[green]Done.[/green] Videos saved to: {', '.join(saved_paths)}")


def _run_headless_warmup(
    cfg, state, params, policy_fn, policy_state, step_fn, jax, jnp,
    warmup_steps: int, chunk_size: int, can_batch: bool,
):
    """Advance one environment without rendering, recording, or saving trajectory."""
    if warmup_steps <= 0:
        return state, policy_state

    warmup_fn = (
        _make_headless_warmup_fn(jax, jnp, step_fn)
        if chunk_size > 1 and can_batch
        else None
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("{task.completed}/{task.total} steps"),
    ) as progress:
        task = progress.add_task("Warming up", total=warmup_steps)
        completed = 0
        if warmup_fn is not None:
            use_straight_policy = bool(params.ec.controlled_agent)
            while completed < warmup_steps:
                n = min(chunk_size, warmup_steps - completed)
                state = warmup_fn(state, params, n, use_straight_policy)
                completed += n
                progress.advance(task, n)
        else:
            while completed < warmup_steps:
                action, policy_state = policy_fn(
                    state, params, policy_state, policy_state["key"],
                )
                state, _, done, _ = step_fn(state, action, params)
                completed += 1
                progress.advance(task)
                if done:
                    break

    return state, policy_state


def _run_headless_multi_warmup(
    cfg, states, params, policy_fn, policy_states, step_fn, jax, jnp,
    warmup_steps: int, chunk_size: int, can_chunk: bool,
):
    """Advance batched environments without rendering, recording, or saving trajectory."""
    if warmup_steps <= 0:
        return states, policy_states

    warmup_fn = (
        _make_headless_multi_warmup_fn(jax, jnp, step_fn)
        if chunk_size > 1 and can_chunk
        else None
    )
    step_fn_batched = None if warmup_fn is not None else _make_batched_step_fn(jax, step_fn)
    num_envs = len(policy_states)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("{task.completed}/{task.total} env-steps"),
    ) as progress:
        task = progress.add_task("Warming up", total=warmup_steps * num_envs)
        completed = 0
        if warmup_fn is not None:
            use_straight_policy = bool(params.ec.controlled_agent)
            while completed < warmup_steps:
                n = min(chunk_size, warmup_steps - completed)
                states = warmup_fn(states, params, n, use_straight_policy)
                completed += n
                progress.advance(task, n * num_envs)
        else:
            while completed < warmup_steps:
                actions = []
                for env_index in range(num_envs):
                    state_i = _tree_index(jax, states, env_index)
                    action, policy_states[env_index] = policy_fn(
                        state_i,
                        params,
                        policy_states[env_index],
                        policy_states[env_index]["key"],
                    )
                    actions.append(action)

                action_array = jnp.asarray(actions, dtype=jnp.float32)
                states = step_fn_batched(states, action_array, params)
                completed += 1
                progress.advance(task, num_envs)

    return states, policy_states


def main():
    cfg = load_config(sys.argv[1:])
    configure_jax_platform(cfg.get("device", "auto"))

    import jax
    import jax.numpy as jnp

    from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
    from flockworld.policies import get_policy, init_policy
    from flockworld.rendering.renderer import build_uv_grid
    from flockworld.video.recorder import DlpackNvencVideoRecorder, VideoRecorder

    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)

    policy_name = cfg.env.agent_policy
    policy_fn = get_policy(policy_name)

    mode = "render" if cfg.env.render else "headless"
    console.print(
        f"[bold]FlockWorld[/bold]  |  {ec.num_agents} agents  |  "
        f"{ec.canvas_w}x{ec.canvas_h}  |  boundary={ec.boundary}  |  "
        f"policy={policy_name}  |  mode={mode}  |  "
        f"envs={_generation_num_envs(cfg)}  |  device={jax.devices()[0]}"
    )

    uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)

    key = jax.random.PRNGKey(cfg.seed)
    state = reset(key, ec)

    _warmup(state, params, uv_grid, jnp, step, render)
    state = reset(key, ec)

    _, policy_key = jax.random.split(key)
    policy_state = init_policy(policy_name, policy_key)

    num_envs = _generation_num_envs(cfg)
    if cfg.env.render and num_envs > 1:
        raise ValueError("generation.num_envs > 1 is only supported for headless video recording.")

    if cfg.env.render:
        _run_render(
            cfg, ec, params, uv_grid, state, policy_fn, policy_state,
            step, render,
        )
    else:
        recorder_cls = _video_recorder_cls(cfg, VideoRecorder, DlpackNvencVideoRecorder)
        if str(cfg.video.get("backend", "opencv")).lower() in {"pynv", "dlpack", "nvenc"}:
            chunk_size = max(1, int(cfg.video.get("chunk_size", 1)))
            can_chunk = (not ec.controlled_agent) or cfg.env.agent_policy == "straight"
            if chunk_size <= 1 or not can_chunk:
                raise ValueError(
                    "video.backend=pynv requires video.chunk_size > 1 and an uncontrolled "
                    "or straight controlled-agent policy so frames stay batched on GPU."
                )
        if num_envs == 1:
            _run_headless(
                cfg, ec, params, uv_grid, state, policy_fn, policy_state,
                step, render, recorder_cls, jax, jnp,
            )
        else:
            seeds = _generation_seeds(cfg)
            keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
            states = jax.vmap(lambda k: reset(k, ec))(keys)
            policy_states = [
                init_policy(policy_name, jax.random.split(jax.random.PRNGKey(seed))[1])
                for seed in seeds
            ]
            _run_headless_multi(
                cfg, ec, params, uv_grid, states, policy_fn, policy_states, seeds,
                step, render, recorder_cls, jax, jnp,
            )


if __name__ == "__main__":
    main()

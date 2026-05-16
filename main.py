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
from functools import partial
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from flockworld.runtime import configure_jax_platform

console = Console()

TRAJECTORY_KEYS = (
    "positions",
    "velocities",
    "accelerations",
    "headings",
    "actions",
    "step_count",
)


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
                    frames_np = np.asarray(frames)
                    positions_np = np.asarray(controlled_positions)
                    if save_trajectory:
                        _append_trajectory_chunk(trajectory_chunks, trajectory, n)

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
                    frames_np = np.asarray(frames[:n])
                    positions_np = np.asarray(controlled_positions[:n])
                    if save_trajectory:
                        _append_multi_trajectory_chunk(trajectory_chunks, trajectory, n)
                    _record_multi_chunk(recorders, frames_np, positions_np)
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


def _make_headless_warmup_fn(jax, jnp, step_fn):
    """Build a jitted no-render warmup function for one environment."""

    @partial(jax.jit, static_argnames=("params", "warmup_steps", "use_straight_policy"))
    def _warmup_scan(state, params, warmup_steps: int, use_straight_policy: bool):
        def _one_step(carry, _):
            action = (
                carry.boids.headings[0]
                if use_straight_policy
                else jnp.float32(0.0)
            )
            next_state, _, _, _ = step_fn(carry, action, params)
            return next_state, None

        state, _ = jax.lax.scan(_one_step, state, None, length=warmup_steps)
        return state

    return _warmup_scan


def _make_headless_multi_warmup_fn(jax, jnp, step_fn):
    """Build a jitted no-render warmup function for batched environments."""

    @partial(jax.jit, static_argnames=("params", "warmup_steps", "use_straight_policy"))
    def _warmup_scan(states, params, warmup_steps: int, use_straight_policy: bool):
        def _one_env(state):
            action = (
                state.boids.headings[0]
                if use_straight_policy
                else jnp.float32(0.0)
            )
            next_state, _, _, _ = step_fn(state, action, params)
            return next_state

        def _one_step(carry, _):
            return jax.vmap(_one_env)(carry), None

        states, _ = jax.lax.scan(_one_step, states, None, length=warmup_steps)
        return states

    return _warmup_scan


def _make_batched_step_fn(jax, step_fn):
    """Build a jitted batched one-step function without rendering."""

    @partial(jax.jit, static_argnames=("params",))
    def _step(states, actions, params):
        def _one_env(state, action):
            next_state, _, _, _ = step_fn(state, action, params)
            return next_state

        return jax.vmap(_one_env)(states, actions)

    return _step


def _make_headless_chunk_fn(jax, jnp, step_fn, render_fn):
    """Build a jitted function that steps, renders, and uint8-converts chunks."""

    @partial(jax.jit, static_argnames=("params", "chunk_size", "use_straight_policy"))
    def _generate_chunk(state, params, uv_grid, chunk_size: int, use_straight_policy: bool):
        def _one_frame(carry, _):
            action = (
                carry.boids.headings[0]
                if use_straight_policy
                else jnp.float32(0.0)
            )
            next_state, _, _, _ = step_fn(carry, action, params)
            frame_f = render_fn(next_state, params, uv_grid)
            frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
            controlled_pos = next_state.boids.positions[0]
            trajectory = _trajectory_snapshot_jnp(next_state, jnp)
            return next_state, (frame_u8, controlled_pos, trajectory)

        return jax.lax.scan(_one_frame, state, None, length=chunk_size)

    return _generate_chunk


def _make_headless_multi_chunk_fn(jax, jnp, step_fn, render_fn):
    """Build a jitted chunk generator with axes (time, env, ...)."""

    @partial(jax.jit, static_argnames=("params", "chunk_size", "use_straight_policy"))
    def _generate_chunk(states, params, uv_grid, chunk_size: int, use_straight_policy: bool):
        def _one_env(state):
            action = (
                state.boids.headings[0]
                if use_straight_policy
                else jnp.float32(0.0)
            )
            next_state, _, _, _ = step_fn(state, action, params)
            frame_f = render_fn(next_state, params, uv_grid)
            frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
            controlled_pos = next_state.boids.positions[0]
            trajectory = _trajectory_snapshot_jnp(next_state, jnp)
            return next_state, (frame_u8, controlled_pos, trajectory)

        def _one_frame(carry, _):
            return jax.vmap(_one_env)(carry)

        return jax.lax.scan(_one_frame, states, None, length=chunk_size)

    return _generate_chunk


def _make_batched_step_render_fn(jax, jnp, step_fn, render_fn):
    """Build a jitted one-frame generator for batched policy actions."""

    @partial(jax.jit, static_argnames=("params",))
    def _step_render(states, actions, params, uv_grid):
        def _one_env(state, action):
            next_state, _, _, _ = step_fn(state, action, params)
            frame_f = render_fn(next_state, params, uv_grid)
            frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
            controlled_pos = next_state.boids.positions[0]
            trajectory = _trajectory_snapshot_jnp(next_state, jnp)
            return next_state, (frame_u8, controlled_pos, trajectory)

        return jax.vmap(_one_env)(states, actions)

    return _step_render


def _tree_index(jax, tree, index: int):
    return jax.tree_util.tree_map(lambda x: x[index], tree)


def _generation_num_envs(cfg) -> int:
    generation = cfg.get("generation", {})
    return max(1, int(generation.get("num_envs", 1)))


def _generation_seeds(cfg) -> list[int]:
    generation = cfg.get("generation", {})
    num_envs = _generation_num_envs(cfg)
    stride = int(generation.get("seed_stride", 1))
    base_seed = int(cfg.seed)
    return [base_seed + i * stride for i in range(num_envs)]


def _video_warmup_steps(cfg) -> int:
    return max(0, int(cfg.video.get("warmup", 0)))


def _output_paths_for_env(cfg, env_index: int, seed: int):
    if _generation_num_envs(cfg) == 1:
        full_path = None if cfg.video.partial_only else cfg.video.full_obs_path
        partial_path = None if cfg.video.full_obs_only else cfg.video.partial_obs_path
        return full_path, partial_path

    generation = cfg.get("generation", {})
    values = {"env": env_index, "seed": seed}
    full_template = generation.get(
        "full_obs_path_template",
        "output/env_{env:04d}_seed_{seed}_full_obs.mp4",
    )
    partial_template = generation.get(
        "partial_obs_path_template",
        "output/env_{env:04d}_seed_{seed}_partial_obs.mp4",
    )
    full_path = None if cfg.video.partial_only else full_template.format(**values)
    partial_path = None if cfg.video.full_obs_only else partial_template.format(**values)
    return full_path, partial_path


def _record_multi_chunk(recorders, frames_np: np.ndarray, positions_np: np.ndarray):
    for env_index, recorder in enumerate(recorders):
        for frame_u8, controlled_pos in zip(frames_np[:, env_index], positions_np[:, env_index]):
            recorder.record(frame_u8, controlled_pos)


def _trajectory_enabled(cfg) -> bool:
    trajectory = cfg.get("trajectory", {})
    return bool(trajectory.get("enabled", False))


def _trajectory_snapshot_jnp(state, jnp):
    return {
        "positions": state.boids.positions,
        "velocities": state.boids.velocities,
        "accelerations": state.boids.accelerations,
        "headings": state.boids.headings,
        "actions": state.boids.headings,
        "step_count": jnp.asarray(state.step_count, dtype=jnp.int32),
    }


def _trajectory_snapshot_np(state):
    return {
        "positions": np.asarray(state.boids.positions),
        "velocities": np.asarray(state.boids.velocities),
        "accelerations": np.asarray(state.boids.accelerations),
        "headings": np.asarray(state.boids.headings),
        "actions": np.asarray(state.boids.headings),
        "step_count": np.asarray(state.step_count, dtype=np.int32),
    }


def _append_trajectory_chunk(trajectory_chunks: list[dict], trajectory, n: int):
    trajectory_np = {key: np.asarray(trajectory[key][:n]) for key in TRAJECTORY_KEYS}
    trajectory_chunks.append(trajectory_np)


def _append_multi_trajectory_chunk(trajectory_chunks: list[list[dict]], trajectory, n: int):
    trajectory_np = {key: np.asarray(trajectory[key][:n]) for key in TRAJECTORY_KEYS}
    for env_index, env_chunks in enumerate(trajectory_chunks):
        env_chunks.append({key: trajectory_np[key][:, env_index] for key in TRAJECTORY_KEYS})


def _append_multi_trajectory_frame(trajectory_chunks: list[list[dict]], trajectory):
    trajectory_np = {key: np.asarray(trajectory[key]) for key in TRAJECTORY_KEYS}
    for env_index, env_chunks in enumerate(trajectory_chunks):
        env_chunks.append({key: trajectory_np[key][env_index] for key in TRAJECTORY_KEYS})


def _trajectory_path_for_env(cfg, env_index: int, seed: int) -> str:
    trajectory = cfg.get("trajectory", {})
    if _generation_num_envs(cfg) == 1:
        return trajectory.get("path", "output/trajectory.npy")

    template = trajectory.get(
        "path_template",
        "output/env_{env:04d}_seed_{seed}_trajectory.npy",
    )
    return template.format(env=env_index, seed=seed)


def _save_trajectory(cfg, path: str, seed: int, trajectory_chunks: list[dict]):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_chunks:
        arrays = {
            key: np.concatenate(
                [_with_time_axis(key, chunk[key]) for chunk in trajectory_chunks],
                axis=0,
            )
            for key in TRAJECTORY_KEYS
        }
    else:
        arrays = {key: np.asarray([]) for key in TRAJECTORY_KEYS}

    payload = {
        **arrays,
        "seed": int(seed),
        "fps": int(cfg.video.fps),
        "warmup": _video_warmup_steps(cfg),
        "canvas_size": np.asarray([int(cfg.canvas.width), int(cfg.canvas.height)]),
        "partial_obs_size": int(cfg.canvas.partial),
    }
    np.save(out_path, payload, allow_pickle=True)


def _with_time_axis(key: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    expected_ndim = {
        "positions": 3,
        "velocities": 3,
        "accelerations": 3,
        "headings": 2,
        "actions": 2,
        "step_count": 1,
    }[key]
    if value.ndim == expected_ndim:
        return value
    if value.ndim == expected_ndim - 1:
        return value[None]
    raise ValueError(f"Unexpected trajectory shape for {key}: {value.shape}")


def main():
    cfg = load_config(sys.argv[1:])
    configure_jax_platform(cfg.get("device", "auto"))

    import jax
    import jax.numpy as jnp

    from flockworld.env.flock_env import EnvParams, env_config_from_omega, render, reset, step
    from flockworld.policies import get_policy, init_policy
    from flockworld.rendering.renderer import build_uv_grid
    from flockworld.video.recorder import VideoRecorder

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
        if num_envs == 1:
            _run_headless(
                cfg, ec, params, uv_grid, state, policy_fn, policy_state,
                step, render, VideoRecorder, jax, jnp,
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
                step, render, VideoRecorder, jax, jnp,
            )


if __name__ == "__main__":
    main()

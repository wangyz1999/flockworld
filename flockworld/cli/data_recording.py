"""FlockWorld CLI — run the boid simulation and record videos.

Usage
-----
    python -m flockworld.cli.data_recording                              # configured dataset collection
    python -m flockworld.cli.data_recording collection.enabled=false generation.num_envs=1 env.render=true
    python -m flockworld.cli.data_recording boids.num_agents=100         # override via CLI
    python -m flockworld.cli.data_recording video.duration=5 seed=123    # multiple overrides
"""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from flockworld.runtime import configure_jax_platform
from flockworld.utils import (
    load_config,
    _collection_enabled,
    _collection_output_root,
    _collection_partial_agent_count,
    _collection_save_trajectory,
    _collection_total_episodes,
    _generation_num_envs,
    _generation_seeds,
    _video_warmup_steps,
    _output_paths_for_env,
    _trajectory_path_for_env,
    _trajectory_enabled,
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
    _write_collection_metadata,
    _write_collection_settings,
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
    partial_agent_count = 1
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
                        state, params, uv_grid, n, use_straight_policy, partial_agent_count,
                    )
                    frames, partial_positions, trajectory = chunk_outputs
                    if save_trajectory:
                        _append_trajectory_chunk(trajectory_chunks, trajectory, n)

                    frames_np = np.asarray(frames)
                    positions_np = np.asarray(partial_positions)
                    for frame_u8, partial_pos in zip(frames_np[:n], positions_np[:n]):
                        recorder.record(frame_u8, partial_pos)

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

                    partial_positions = np.asarray(state.boids.positions[:partial_agent_count])
                    recorder.record(frame_u8, partial_positions)
                    if save_trajectory:
                        trajectory_chunks.append(_trajectory_snapshot_np(state))

                    recorded += 1
                    progress.advance(task)

                    if done:
                        break

    saved = [p for p in (full_path, partial_path) if p is not None]
    if save_trajectory:
        trajectory_path = _trajectory_path_for_env(cfg, env_index=0, seed=int(cfg.seed))
        saved.append(_save_trajectory(cfg, trajectory_path, int(cfg.seed), trajectory_chunks))
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
    partial_agent_count = 1

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
                        states, params, uv_grid, n, use_straight_policy, partial_agent_count,
                    )
                    frames, partial_positions, trajectory = chunk_outputs
                    if save_trajectory:
                        _append_multi_trajectory_chunk(trajectory_chunks, trajectory, n)
                    _record_multi_chunk(recorders, frames, partial_positions, n)
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
                    states, frame_outputs = step_render_fn(
                        states, action_array, params, uv_grid, partial_agent_count,
                    )
                    frames, partial_positions, trajectory = frame_outputs
                    frames_np = np.asarray(frames)
                    positions_np = np.asarray(partial_positions)
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
            saved_paths.append(
                _save_trajectory(cfg, trajectory_path, seed, trajectory_chunks[env_index])
            )

    console.print(f"[green]Done.[/green] Videos saved to: {', '.join(saved_paths)}")


def _run_collection(
    cfg, ec, params, uv_grid, policy_name, policy_fn,
    reset_fn, init_policy_fn, step_fn, render_fn, recorder_cls, jax, jnp,
):
    """Dataset collection mode with timestamped episode folders."""
    total_episodes = _collection_total_episodes(cfg)
    parallel_envs = min(_generation_num_envs(cfg), max(1, total_episodes))
    partial_agent_count = _collection_partial_agent_count(cfg, ec.num_agents)
    save_trajectory = _collection_save_trajectory(cfg)
    output_dir = _new_collection_dir(cfg)

    _prepare_collection_dirs(output_dir, partial_agent_count)
    _write_collection_settings(output_dir / "settings.yaml", cfg)

    fps = int(cfg.video.fps)
    duration = float(cfg.video.duration)
    total_frames = int(fps * duration)
    warmup_steps = min(_video_warmup_steps(cfg), ec.max_steps)
    max_frames = min(total_frames, max(0, ec.max_steps - warmup_steps))
    chunk_size = max(1, int(cfg.video.get("chunk_size", 1)))
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

    started_at = datetime.now(timezone.utc).isoformat()
    episode_records = []
    collected = 0
    stride = int(cfg.get("generation", {}).get("seed_stride", 1))
    base_seed = int(cfg.seed)

    console.print(
        f"[bold green]Collection mode[/bold green] -> {output_dir} | "
        f"episodes={total_episodes} | envs={parallel_envs} | partial_agents={partial_agent_count}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
    ) as progress:
        task = progress.add_task("Collecting episodes", total=max_frames * total_episodes)

        while collected < total_episodes:
            batch_size = min(parallel_envs, total_episodes - collected)
            episode_indices = list(range(collected, collected + batch_size))
            seeds = [base_seed + episode_index * stride for episode_index in episode_indices]
            keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
            states = jax.vmap(lambda k: reset_fn(k, ec))(keys)
            policy_states = [
                init_policy_fn(policy_name, jax.random.split(jax.random.PRNGKey(seed))[1])
                for seed in seeds
            ]

            states, policy_states = _run_headless_multi_warmup(
                cfg, states, params, policy_fn, policy_states, step_fn, jax, jnp,
                warmup_steps, chunk_size, can_chunk, show_progress=False,
            )

            first_episode = episode_indices[0]
            last_episode = episode_indices[-1]
            progress.update(
                task,
                description=f"Collecting episodes {first_episode:05d}-{last_episode:05d}",
            )

            batch_records = _record_collection_batch(
                cfg=cfg,
                ec=ec,
                params=params,
                uv_grid=uv_grid,
                states=states,
                policy_fn=policy_fn,
                policy_states=policy_states,
                seeds=seeds,
                episode_indices=episode_indices,
                output_dir=output_dir,
                partial_agent_count=partial_agent_count,
                save_trajectory=save_trajectory,
                max_frames=max_frames,
                chunk_size=chunk_size,
                chunk_fn=chunk_fn,
                step_render_fn=step_render_fn,
                step_fn=step_fn,
                recorder_cls=recorder_cls,
                progress=progress,
                progress_task=task,
                jax=jax,
                jnp=jnp,
            )
            episode_records.extend(batch_records)
            collected += batch_size

        progress.update(task, description="Collection complete")

    completed_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "started_at": started_at,
        "completed_at": completed_at,
        "output_dir": str(output_dir),
        "settings_path": "settings.yaml",
        "stats": {
            "total_count": len(episode_records),
            "total_limit": total_episodes,
            "parallel_envs": parallel_envs,
            "partial_agent_count": partial_agent_count,
            "frames_per_episode": max_frames,
            "fps": fps,
            "duration": duration,
            "warmup_steps": warmup_steps,
            "chunk_size": chunk_size,
            "trajectory_saved": save_trajectory,
        },
        "episodes": episode_records,
    }
    _write_collection_metadata(output_dir / "metadata.json", metadata)
    console.print(f"[green]Done.[/green] Collection saved to: {output_dir}")


def _record_collection_batch(
    *,
    cfg,
    ec,
    params,
    uv_grid,
    states,
    policy_fn,
    policy_states,
    seeds,
    episode_indices,
    output_dir: Path,
    partial_agent_count: int,
    save_trajectory: bool,
    max_frames: int,
    chunk_size: int,
    chunk_fn,
    step_render_fn,
    step_fn,
    recorder_cls,
    progress,
    progress_task,
    jax,
    jnp,
):
    num_envs = len(seeds)
    trajectory_chunks = [[] for _ in range(num_envs)]
    recorders = []
    episode_records = []

    try:
        for env_index, episode_index in enumerate(episode_indices):
            full_path, partial_paths, trajectory_path = _collection_paths_for_episode(
                cfg, output_dir, episode_index, partial_agent_count,
            )
            recorders.append(
                recorder_cls(
                    full_obs_path=full_path,
                    partial_obs_paths=partial_paths,
                    partial_obs_size=cfg.canvas.partial,
                    fps=cfg.video.fps,
                    canvas_w=ec.canvas_w,
                    canvas_h=ec.canvas_h,
                )
            )
            episode_records.append(
                {
                    "index": int(episode_index),
                    "seed": int(seeds[env_index]),
                    "frames": int(max_frames),
                    "video_global": _relative_path(full_path, output_dir),
                    "video_partial": [
                        _relative_path(path, output_dir) for path in partial_paths
                    ],
                    "state_action": _relative_path(trajectory_path, output_dir)
                    if save_trajectory else None,
                }
            )

        recorded = 0
        if chunk_fn is not None:
            use_straight_policy = bool(ec.controlled_agent)
            while recorded < max_frames:
                n = min(chunk_size, max_frames - recorded)
                states, chunk_outputs = chunk_fn(
                    states, params, uv_grid, n, use_straight_policy, partial_agent_count,
                )
                frames, partial_positions, trajectory = chunk_outputs
                if save_trajectory:
                    _append_multi_trajectory_chunk(trajectory_chunks, trajectory, n)
                _record_multi_chunk(recorders, frames, partial_positions, n)
                recorded += n
                progress.advance(progress_task, n * num_envs)
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
                states, frame_outputs = step_render_fn(
                    states, action_array, params, uv_grid, partial_agent_count,
                )
                frames, partial_positions, trajectory = frame_outputs
                frames_np = np.asarray(frames)
                positions_np = np.asarray(partial_positions)
                if save_trajectory:
                    _append_multi_trajectory_frame(trajectory_chunks, trajectory)
                for env_index, recorder in enumerate(recorders):
                    recorder.record(frames_np[env_index], positions_np[env_index])

                recorded += 1
                progress.advance(progress_task, num_envs)
    finally:
        for recorder in recorders:
            recorder.close()

    if save_trajectory:
        for env_index, episode_index in enumerate(episode_indices):
            _, _, trajectory_path = _collection_paths_for_episode(
                cfg, output_dir, episode_index, partial_agent_count,
            )
            _save_trajectory(cfg, str(trajectory_path), seeds[env_index], trajectory_chunks[env_index])

    return episode_records


def _new_collection_dir(cfg) -> Path:
    collection = cfg.get("collection", {})
    timestamp = collection.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = _collection_output_root(cfg) / "recording" / str(timestamp)
    if not output_dir.exists():
        return output_dir

    suffix = 1
    while True:
        candidate = output_dir.with_name(f"{output_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _prepare_collection_dirs(output_dir: Path, partial_agent_count: int):
    (output_dir / "video_global").mkdir(parents=True, exist_ok=True)
    for agent_index in range(partial_agent_count):
        (output_dir / f"video_a{agent_index + 1}").mkdir(parents=True, exist_ok=True)
    (output_dir / "state_action").mkdir(parents=True, exist_ok=True)


def _collection_paths_for_episode(cfg, output_dir: Path, episode_index: int, partial_agent_count: int):
    filename = f"{episode_index:05d}.mp4"
    full_path = None if cfg.video.partial_only else output_dir / "video_global" / filename
    partial_paths = []
    if not cfg.video.full_obs_only:
        partial_paths = [
            output_dir / f"video_a{agent_index + 1}" / filename
            for agent_index in range(partial_agent_count)
        ]
    trajectory_path = output_dir / "state_action" / f"{episode_index:05d}.parquet"
    return full_path, partial_paths, trajectory_path


def _relative_path(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return str(Path(path).relative_to(root))


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
    warmup_steps: int, chunk_size: int, can_chunk: bool, show_progress: bool = True,
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

    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.completed}/{task.total} env-steps"),
        )
        if show_progress
        else nullcontext(None)
    )

    with progress_cm as progress:
        task = (
            progress.add_task("Warming up", total=warmup_steps * num_envs)
            if progress is not None
            else None
        )
        completed = 0
        if warmup_fn is not None:
            use_straight_policy = bool(params.ec.controlled_agent)
            while completed < warmup_steps:
                n = min(chunk_size, warmup_steps - completed)
                states = warmup_fn(states, params, n, use_straight_policy)
                completed += n
                if progress is not None:
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
                if progress is not None:
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
    from flockworld.video.recorder import VideoRecorder

    ec = env_config_from_omega(cfg)
    params = EnvParams(ec)

    policy_name = cfg.env.agent_policy
    policy_fn = get_policy(policy_name)

    mode = "collection" if _collection_enabled(cfg) else ("render" if cfg.env.render else "headless")
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
    if cfg.env.render and _collection_enabled(cfg):
        raise ValueError("collection.enabled=true is only supported for headless recording.")
    if cfg.env.render and num_envs > 1:
        raise ValueError("generation.num_envs > 1 is only supported for headless video recording.")

    if cfg.env.render:
        _run_render(
            cfg, ec, params, uv_grid, state, policy_fn, policy_state,
            step, render,
        )
    else:
        recorder_cls = VideoRecorder
        if _collection_enabled(cfg):
            _run_collection(
                cfg, ec, params, uv_grid, policy_name, policy_fn,
                reset, init_policy, step, render, recorder_cls, jax, jnp,
            )
        elif num_envs == 1:
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

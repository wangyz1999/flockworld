from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

TRAJECTORY_KEYS = (
    "positions",
    "velocities",
    "accelerations",
    "headings",
    "actions",
    "step_count",
)


def load_config(cli_args: list[str] | None = None):
    """Load ``config/data_recording.yaml`` and merge CLI overrides."""
    base_path = Path(__file__).resolve().parent.parent / "config" / "data_recording.yaml"
    base_cfg = OmegaConf.load(base_path)
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg
    return cfg


# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------

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


def _trajectory_enabled(cfg) -> bool:
    trajectory = cfg.get("trajectory", {})
    return bool(trajectory.get("enabled", False))


def _collection_enabled(cfg) -> bool:
    collection = cfg.get("collection", {})
    return bool(collection.get("enabled", False))


def _collection_total_episodes(cfg) -> int:
    collection = cfg.get("collection", {})
    return max(0, int(collection.get("total_episodes", collection.get("total_limit", 1))))


def _collection_partial_agent_count(cfg, num_agents: int) -> int:
    collection = cfg.get("collection", {})
    requested = int(collection.get("partial_agents", 1))
    return max(0, min(requested, int(num_agents)))


def _collection_save_trajectory(cfg) -> bool:
    collection = cfg.get("collection", {})
    return bool(collection.get("save_trajectory", True))


def _collection_output_root(cfg) -> Path:
    collection = cfg.get("collection", {})
    return Path(collection.get("output_root", "outputs"))


def _write_collection_metadata(path: str | Path, payload: dict):
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_collection_settings(path: str | Path, cfg):
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=settings_path, resolve=True)


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


def _trajectory_path_for_env(cfg, env_index: int, seed: int) -> str:
    trajectory = cfg.get("trajectory", {})
    if _generation_num_envs(cfg) == 1:
        return trajectory.get("path", "output/trajectory.parquet")

    template = trajectory.get(
        "path_template",
        "output/env_{env:04d}_seed_{seed}_trajectory.parquet",
    )
    return template.format(env=env_index, seed=seed)


def _video_recorder_cls(cfg, opencv_recorder_cls, dlpack_recorder_cls):
    backend = str(cfg.video.get("backend", "opencv")).lower()
    if backend == "opencv":
        return opencv_recorder_cls
    if backend in {"pynv", "dlpack", "nvenc"}:
        return partial(
            dlpack_recorder_cls,
            codec=str(cfg.video.get("codec", "h264")),
            gpu_id=int(cfg.video.get("gpu_id", 0)),
            preset=str(cfg.video.get("preset", "p1")),
            bitrate=cfg.video.get("bitrate", "20M"),
        )
    raise ValueError(
        f"Unknown video.backend={backend!r}. Supported backends: opencv, pynv."
    )


# ---------------------------------------------------------------------------
# JAX tree / batching helpers
# ---------------------------------------------------------------------------

def _tree_index(jax, tree, index: int):
    return jax.tree_util.tree_map(lambda x: x[index], tree)


def _make_headless_warmup_fn(jax, jnp, step_fn):
    """Jitted no-render warmup function for one environment."""

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
    """Jitted no-render warmup function for batched environments."""

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
    """Jitted batched one-step function without rendering."""

    @partial(jax.jit, static_argnames=("params",))
    def _step(states, actions, params):
        def _one_env(state, action):
            next_state, _, _, _ = step_fn(state, action, params)
            return next_state

        return jax.vmap(_one_env)(states, actions)

    return _step


def _make_headless_chunk_fn(jax, jnp, step_fn, render_fn):
    """Jitted function that steps, renders, and uint8-converts chunks for one environment."""

    @partial(
        jax.jit,
        static_argnames=("params", "chunk_size", "use_straight_policy", "partial_agent_count"),
    )
    def _generate_chunk(
        state,
        params,
        uv_grid,
        chunk_size: int,
        use_straight_policy: bool,
        partial_agent_count: int,
    ):
        def _one_frame(carry, _):
            action = (
                carry.boids.headings[0]
                if use_straight_policy
                else jnp.float32(0.0)
            )
            next_state, _, _, _ = step_fn(carry, action, params)
            frame_f = render_fn(next_state, params, uv_grid)
            frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
            partial_positions = next_state.boids.positions[:partial_agent_count]
            trajectory = _trajectory_snapshot_jnp(next_state, jnp)
            return next_state, (frame_u8, partial_positions, trajectory)

        return jax.lax.scan(_one_frame, state, None, length=chunk_size)

    return _generate_chunk


def _make_headless_multi_chunk_fn(jax, jnp, step_fn, render_fn):
    """Jitted chunk generator with axes (time, env, ...) for batched environments."""

    @partial(
        jax.jit,
        static_argnames=("params", "chunk_size", "use_straight_policy", "partial_agent_count"),
    )
    def _generate_chunk(
        states,
        params,
        uv_grid,
        chunk_size: int,
        use_straight_policy: bool,
        partial_agent_count: int,
    ):
        def _one_env(state):
            action = (
                state.boids.headings[0]
                if use_straight_policy
                else jnp.float32(0.0)
            )
            next_state, _, _, _ = step_fn(state, action, params)
            frame_f = render_fn(next_state, params, uv_grid)
            frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
            partial_positions = next_state.boids.positions[:partial_agent_count]
            trajectory = _trajectory_snapshot_jnp(next_state, jnp)
            return next_state, (frame_u8, partial_positions, trajectory)

        def _one_frame(carry, _):
            return jax.vmap(_one_env)(carry)

        return jax.lax.scan(_one_frame, states, None, length=chunk_size)

    return _generate_chunk


def _make_batched_step_render_fn(jax, jnp, step_fn, render_fn):
    """Jitted one-frame generator for batched policy actions."""

    @partial(jax.jit, static_argnames=("params", "partial_agent_count"))
    def _step_render(states, actions, params, uv_grid, partial_agent_count: int):
        def _one_env(state, action):
            next_state, _, _, _ = step_fn(state, action, params)
            frame_f = render_fn(next_state, params, uv_grid)
            frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
            partial_positions = next_state.boids.positions[:partial_agent_count]
            trajectory = _trajectory_snapshot_jnp(next_state, jnp)
            return next_state, (frame_u8, partial_positions, trajectory)

        return jax.vmap(_one_env)(states, actions)

    return _step_render


# ---------------------------------------------------------------------------
# Trajectory utilities
# ---------------------------------------------------------------------------

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


def _record_multi_chunk(recorders, frames, positions, n: int):
    if recorders and hasattr(recorders[0], "record_jax_chunk"):
        for env_index, recorder in enumerate(recorders):
            recorder.record_jax_chunk(frames[:n, env_index], positions[:n, env_index], n)
        return

    frames_np = np.asarray(frames[:n])
    positions_np = np.asarray(positions[:n])
    for env_index, recorder in enumerate(recorders):
        for frame_u8, controlled_pos in zip(frames_np[:, env_index], positions_np[:, env_index]):
            recorder.record(frame_u8, controlled_pos)


def _save_trajectory(cfg, path: str, seed: int, trajectory_chunks: list[dict]) -> str:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Saving trajectories as Parquet requires polars. "
            "Install project dependencies with `uv sync` or install polars in the active environment."
        ) from exc

    out_path = Path(path)
    if out_path.suffix.lower() != ".parquet":
        out_path = out_path.with_suffix(".parquet")
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

    df = _trajectory_dataframe(pl, cfg, int(seed), arrays)
    df.write_parquet(out_path, compression="zstd")
    return str(out_path)


def _trajectory_dataframe(pl, cfg, seed: int, arrays: dict[str, np.ndarray]):
    positions = np.asarray(arrays["positions"])
    if positions.size == 0:
        row_count = 0
        num_agents = int(cfg.boids.num_agents)
    else:
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError(f"Expected positions with shape (T, N, 2), got {positions.shape}.")
        row_count = int(positions.shape[0] * positions.shape[1])
        num_agents = int(positions.shape[1])

    def _flat_pair(key: str, axis: int) -> np.ndarray:
        value = np.asarray(arrays[key])
        if row_count == 0:
            return np.asarray([], dtype=np.float32)
        return value[..., axis].reshape(-1).astype(np.float32, copy=False)

    def _flat_scalar(key: str, dtype) -> np.ndarray:
        value = np.asarray(arrays[key])
        if row_count == 0:
            return np.asarray([], dtype=dtype)
        return value.reshape(-1).astype(dtype, copy=False)

    if row_count == 0:
        frame = np.asarray([], dtype=np.int32)
        agent = np.asarray([], dtype=np.int32)
        step_count = np.asarray([], dtype=np.int32)
    else:
        frames = int(positions.shape[0])
        frame = np.repeat(np.arange(frames, dtype=np.int32), num_agents)
        agent = np.tile(np.arange(num_agents, dtype=np.int32), frames)
        step_count = np.repeat(
            np.asarray(arrays["step_count"], dtype=np.int32).reshape(-1),
            num_agents,
        )

    data = {
        "seed": np.full(row_count, seed, dtype=np.int64),
        "fps": np.full(row_count, int(cfg.video.fps), dtype=np.int32),
        "warmup": np.full(row_count, _video_warmup_steps(cfg), dtype=np.int32),
        "canvas_width": np.full(row_count, int(cfg.canvas.width), dtype=np.int32),
        "canvas_height": np.full(row_count, int(cfg.canvas.height), dtype=np.int32),
        "partial_obs_size": np.full(row_count, int(cfg.canvas.partial), dtype=np.int32),
        "frame": frame,
        "agent": agent,
        "step_count": step_count,
        "position_x": _flat_pair("positions", 0),
        "position_y": _flat_pair("positions", 1),
        "velocity_x": _flat_pair("velocities", 0),
        "velocity_y": _flat_pair("velocities", 1),
        "acceleration_x": _flat_pair("accelerations", 0),
        "acceleration_y": _flat_pair("accelerations", 1),
        "heading": _flat_scalar("headings", np.float32),
        "action": _flat_scalar("actions", np.float32),
    }
    return pl.DataFrame(data)


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

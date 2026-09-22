"""On-the-fly VAE training data: partial-view clips streamed from the JAX sim.

No videos are read from or written to disk. Each dataloader worker runs its
own CPU-pinned JAX boid simulation (the same ``flock_env`` + renderer used by
``flockworld/cli/data_recording.py``), renders full frames, and cuts the same zero-padded
128x128 agent-centred crops the recorder writes -- minus the H.264 round-trip.

Every episode uses a fresh seed drawn from ``(base_seed, namespace, worker_id,
episode_counter)``, so training never revisits data as long as workers persist
across epochs (``persistent_workers: true``). The val split lives in a
disjoint seed namespace and is generated once into memory at setup.

JAX is imported lazily inside the worker process. The train dataloader must
use the ``spawn`` multiprocessing context: the val set is generated in the
main process, and forking a JAX-initialised parent is unsafe.
"""

from __future__ import annotations

import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from flockworld.video.recorder import crop_partial

_TRAIN_NAMESPACE = 0
_VAL_NAMESPACE = 1


def _episode_seed(base_seed: int, namespace: int, worker_id: int, episode: int) -> int:
    ss = np.random.SeedSequence(entropy=(base_seed, namespace, worker_id, episode))
    return int(ss.generate_state(1)[0])


class SimClipGenerator:
    """Runs the boid sim and yields per-agent partial-view clips.

    Holds only plain config until :meth:`episode` is first called, so instances
    pickle cleanly into spawned dataloader workers.
    """

    def __init__(
        self,
        sim_cfg_container: dict,
        num_frames: int,
        frame_stride: int,
        num_partial_agents: int,
        windows_per_episode: int,
        warmup_steps: int,
        threads: int | None,
        return_actions: bool = False,
        return_gt_positions: bool = False,
    ):
        self.sim_cfg_container = sim_cfg_container
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.clip_span = (self.num_frames - 1) * self.frame_stride + 1
        self.num_partial_agents = int(num_partial_agents)
        self.windows_per_episode = int(windows_per_episode)
        self.warmup_steps = int(warmup_steps)
        self.threads = threads
        # World-model streaming also needs per-frame actions; the VAE path leaves this off.
        self.return_actions = bool(return_actions)
        # Eval-only: also return every simulated boid's position (not just the camera
        # agents'), so a GT-vs-detection metric can be computed without disk-recorded
        # parquet. Off by default -- training never sets this, so its output shape is
        # byte-for-byte unchanged.
        self.return_gt_positions = bool(return_gt_positions)
        assert not self.return_gt_positions or self.return_actions, (
            "return_gt_positions is only wired up alongside return_actions (eval use)"
        )
        self._runtime = None

    def _ensure_init(self):
        if self._runtime is not None:
            return
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        if self.threads is not None:
            flags = os.environ.get("XLA_FLAGS", "")
            os.environ["XLA_FLAGS"] = (
                f"{flags} --xla_cpu_multi_thread_eigen=false "
                f"intra_op_parallelism_threads={int(self.threads)}"
            ).strip()

        import jax
        import jax.numpy as jnp
        from functools import partial

        from flockworld.env.flock_env import (
            EnvParams,
            env_config_from_omega,
            render,
            reset,
            sample_boid_colors,
            step,
        )
        from flockworld.rendering.renderer import build_uv_grid

        sim_cfg = OmegaConf.create(self.sim_cfg_container)
        ec = env_config_from_omega(sim_cfg)
        if self.num_partial_agents > ec.num_agents:
            raise ValueError(
                f"num_partial_agents={self.num_partial_agents} exceeds "
                f"boids.num_agents={ec.num_agents}"
            )
        params = EnvParams(ec)
        uv_grid = build_uv_grid(ec.canvas_w, ec.canvas_h)
        num_partial = self.num_partial_agents
        return_gt = self.return_gt_positions  # plain Python bool, resolved once at trace time

        @partial(jax.jit, static_argnames=("steps",))
        def warmup_fn(state, steps: int):
            def one(carry, _):
                next_state, _, _, _ = step(carry, jnp.float32(0.0), params)
                return next_state, None

            state, _ = jax.lax.scan(one, state, None, length=steps)
            return state

        @partial(jax.jit, static_argnames=("steps",))
        def chunk_fn(state, boid_colors, steps: int):
            def one(carry, _):
                next_state, _, _, _ = step(carry, jnp.float32(0.0), params)
                frame_f = render(next_state, params, uv_grid, boid_colors=boid_colors)
                frame_u8 = jnp.clip(frame_f * 255.0, 0.0, 255.0).astype(jnp.uint8)
                out = (
                    frame_u8,
                    next_state.boids.positions[:num_partial],
                    next_state.boids.accelerations[:num_partial],  # per-agent (acc_x, acc_y)
                )
                if return_gt:
                    out = out + (next_state.boids.positions,)  # ALL simulated boids, unsliced
                return next_state, out

            return jax.lax.scan(one, state, None, length=steps)

        self._runtime = {
            "jax": jax,
            "ec": ec,
            "partial_size": int(sim_cfg.canvas.partial),
            "reset": reset,
            "warmup": warmup_fn,
            "chunk": chunk_fn,
            "params": params,
            "sample_colors": sample_boid_colors,
            # "random_hue": redraw every boid's hue each episode, so the VAE sees
            # arbitrary hues instead of one fixed palette.
            "random_hue": ec.color_mode == "random_hue",
        }

    def clips_per_episode(self) -> int:
        return self.num_partial_agents * self.windows_per_episode

    def episode(self, seed: int):
        """Simulate one fresh episode. Returns clips ``(windows*agents, T, size, size, 3)``
        uint8 RGB. If ``return_actions``, returns ``(clips, actions)`` where actions is
        ``(windows*agents, T, 2)`` -- each agent's per-frame acceleration ``(acc_x, acc_y)``,
        the signal the world model conditions on (still at pixel-frame rate; aggregate to
        latent frames at encode time). If ``return_gt_positions``, also returns
        ``gt_positions (windows, T, num_agents, 2)`` -- every simulated boid's world
        position at each of the ``T`` output frames (eval-only; unused by training)."""
        self._ensure_init()
        rt = self._runtime
        size = rt["partial_size"]
        T, stride = self.num_frames, self.frame_stride

        jax = rt["jax"]
        key = jax.random.PRNGKey(seed)
        # Colour key is folded off the episode key rather than split from it, so the
        # reset key stays byte-identical to what a given seed produced before this
        # mode existed — same seed, same trajectory.
        boid_colors = (
            rt["sample_colors"](jax.random.fold_in(key, 0xC0107), rt["ec"].num_agents)
            if rt["random_hue"] else rt["params"].boid_colors
        )

        state = rt["reset"](key, rt["ec"])
        if self.warmup_steps > 0:
            state = rt["warmup"](state, self.warmup_steps)

        clips = np.empty(
            (self.windows_per_episode, self.num_partial_agents, T, size, size, 3),
            dtype=np.uint8,
        )
        actions = (
            np.empty((self.windows_per_episode, self.num_partial_agents, T, 2), np.float32)
            if self.return_actions else None
        )
        gt_positions = None
        for w in range(self.windows_per_episode):
            chunk_out = rt["chunk"](state, boid_colors, self.clip_span)
            if self.return_gt_positions:
                state, (frames, positions, accels, full_positions) = chunk_out
                full_positions = np.asarray(full_positions)  # (span, num_agents_full, 2)
                if gt_positions is None:
                    gt_positions = np.empty(
                        (self.windows_per_episode, T, full_positions.shape[1], 2), np.float32
                    )
            else:
                state, (frames, positions, accels) = chunk_out
            frames = np.asarray(frames)          # (span, H, W, 3) uint8
            positions = np.asarray(positions)    # (span, K, 2)
            accels = np.asarray(accels)          # (span, K, 2)
            for t_out, t in enumerate(range(0, self.clip_span, stride)):
                for k in range(self.num_partial_agents):
                    clips[w, k, t_out] = crop_partial(frames[t], positions[t, k], size)
                    if actions is not None:
                        actions[w, k, t_out] = accels[t, k]
                if gt_positions is not None:
                    gt_positions[w, t_out] = full_positions[t]
        clips = clips.reshape(-1, T, size, size, 3)
        if gt_positions is not None:
            return clips, actions.reshape(-1, T, 2), gt_positions
        if actions is not None:
            return clips, actions.reshape(-1, T, 2)
        return clips


def _clip_to_item(clip_u8: np.ndarray, image_size, episode_id: str, agent_index: int) -> dict:
    """uint8 (T, H, W, 3) -> the dict format of PartialVideoVAEDataset."""
    video = torch.from_numpy(np.ascontiguousarray(clip_u8))
    video = video.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    if image_size is not None and tuple(video.shape[-2:]) != tuple(image_size):
        video = F.interpolate(video, size=tuple(image_size), mode="bilinear", align_corners=False)
    video = video * 2.0 - 1.0
    return {
        "video": video.permute(1, 0, 2, 3).contiguous(),  # (C, T, H, W)
        "episode_id": episode_id,
        "agent_index": agent_index,
    }


class StreamingPartialVAEDataset(IterableDataset):
    """Infinite stream of freshly-simulated clips, ``clips_per_epoch`` per epoch.

    Each worker owns a disjoint seed sequence; the per-worker episode counter
    survives epoch boundaries when ``persistent_workers=True``, so no episode
    is ever simulated twice.
    """

    def __init__(
        self,
        generator: SimClipGenerator,
        clips_per_epoch: int,
        image_size,
        base_seed: int,
    ):
        self.generator = generator
        self.clips_per_epoch = int(clips_per_epoch)
        self.image_size = tuple(image_size) if image_size else None
        self.base_seed = int(base_seed)
        self._episode_counter = 0

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        share = self.clips_per_epoch // num_workers + (
            1 if worker_id < self.clips_per_epoch % num_workers else 0
        )
        yielded = 0
        while yielded < share:
            seed = _episode_seed(
                self.base_seed, _TRAIN_NAMESPACE, worker_id, self._episode_counter
            )
            self._episode_counter += 1
            clips = self.generator.episode(seed)
            order = np.random.default_rng(seed).permutation(len(clips))
            num_agents = self.generator.num_partial_agents
            for i in order:
                if yielded >= share:
                    break
                yield _clip_to_item(
                    clips[i], self.image_size, f"stream_{seed:08x}", int(i) % num_agents
                )
                yielded += 1


class InMemoryClipDataset(Dataset):
    """Fixed clips generated once at setup and served from memory (val split)."""

    def __init__(self, generator: SimClipGenerator, num_clips: int, image_size, base_seed: int):
        self.image_size = tuple(image_size) if image_size else None
        self.clips: list[np.ndarray] = []
        self.episode_ids: list[str] = []
        num_clips = int(num_clips)
        episodes = ceil(num_clips / generator.clips_per_episode())
        for e in range(episodes):
            seed = _episode_seed(int(base_seed), _VAL_NAMESPACE, 0, e)
            for clip in generator.episode(seed):
                if len(self.clips) >= num_clips:
                    break
                self.clips.append(clip)
                self.episode_ids.append(f"stream_val_{seed:08x}")
        self.num_agents = generator.num_partial_agents

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        return _clip_to_item(
            self.clips[index], self.image_size, self.episode_ids[index], index % self.num_agents
        )


def load_sim_cfg_container(sim_config_path: str, overrides) -> dict:
    cfg = OmegaConf.load(sim_config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(o) for o in overrides]))
    return OmegaConf.to_container(cfg, resolve=True)

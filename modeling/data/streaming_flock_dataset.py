"""On-the-fly world-model training data: multi-agent clips streamed from the JAX sim.

The world-model analogue of ``streaming_vae_dataset.py``. Each sample is the P camera
agents' TIME-ALIGNED partial-view clips from one simulated window, plus their per-frame
actions ``(acc_x, acc_y)``. Built on the same ``SimClipGenerator`` (with
``return_actions=True``) and the same disjoint-seed / persistent-worker scheme, so no
window ever repeats.

Samples are yielded as **pixels + actions**, NOT latents: the frozen VAE encodes them on
the GPU in the main process (dataloader workers are CPU), so the encode + action
aggregation + latent normalization live in ``StreamingLatentEncoder`` (applied after the
loader), not here. Shapes per sample:
    pixels  : (P, T_pix, H, W, 3) uint8
    actions : (P, T_pix, 2) float32   (pixel-frame rate; aggregated to latent frames at encode)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from modeling.data.streaming_vae_dataset import (
    _TRAIN_NAMESPACE,
    _VAL_NAMESPACE,
    SimClipGenerator,
    _episode_seed,
)


def _windows(clips: np.ndarray, actions: np.ndarray, num_agents: int):
    """(W*P, T, ...) flat episode output -> W grouped ``(P, T, ...)`` windows.

    SimClipGenerator flattens as ``clips[w, k] -> w*P + k``, so agents of a window are
    contiguous; regroup to recover the aligned P-agent layout the multi-agent model wants.
    """
    P = num_agents
    W = clips.shape[0] // P
    T = clips.shape[1]
    clips = clips.reshape(W, P, T, *clips.shape[2:])
    actions = actions.reshape(W, P, T, actions.shape[-1])
    return W, clips, actions


def _sample(clip_wp: np.ndarray, act_wp: np.ndarray, num_context: int) -> dict:
    return {
        "pixels": torch.from_numpy(np.ascontiguousarray(clip_wp)),   # (P, T, H, W, 3) uint8
        "actions": torch.from_numpy(np.ascontiguousarray(act_wp)),   # (P, T, 2) float32
        "context_len": int(num_context),                             # LATENT frames
    }


class StreamingFlockDataset(IterableDataset):
    """Infinite stream of fresh P-agent windows, ``windows_per_epoch`` per epoch.

    Each worker owns a disjoint seed sequence; the per-worker episode counter survives
    epoch boundaries with ``persistent_workers=True``, so no window is ever simulated twice.
    """

    def __init__(
        self,
        generator: SimClipGenerator,
        windows_per_epoch: int,
        num_agents: int,
        num_context_lat: int,
        base_seed: int,
    ):
        assert generator.return_actions, "world-model streaming needs actions"
        assert generator.num_partial_agents == int(num_agents), (
            f"generator.num_partial_agents={generator.num_partial_agents} != num_agents={num_agents}"
        )
        self.generator = generator
        self.windows_per_epoch = int(windows_per_epoch)
        self.num_agents = int(num_agents)
        self.num_context = int(num_context_lat)
        self.base_seed = int(base_seed)
        self._episode_counter = 0

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        nworkers = info.num_workers if info is not None else 1
        share = self.windows_per_epoch // nworkers + (
            1 if wid < self.windows_per_epoch % nworkers else 0
        )
        yielded = 0
        while yielded < share:
            seed = _episode_seed(self.base_seed, _TRAIN_NAMESPACE, wid, self._episode_counter)
            self._episode_counter += 1
            clips, actions = self.generator.episode(seed)
            W, clips, actions = _windows(clips, actions, self.num_agents)
            for w in range(W):
                if yielded >= share:
                    break
                yield _sample(clips[w], actions[w], self.num_context)
                yielded += 1


class InMemoryFlockDataset(Dataset):
    """Fixed set of P-agent windows generated once at setup (val split, disjoint seeds)."""

    def __init__(self, generator: SimClipGenerator, num_windows: int, num_context_lat: int, base_seed: int):
        assert generator.return_actions, "world-model streaming needs actions"
        self.num_context = int(num_context_lat)
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        e = 0
        while len(self.samples) < int(num_windows):
            seed = _episode_seed(int(base_seed), _VAL_NAMESPACE, 0, e)
            e += 1
            clips, actions = generator.episode(seed)
            W, clips, actions = _windows(clips, actions, generator.num_partial_agents)
            for w in range(W):
                if len(self.samples) >= int(num_windows):
                    break
                self.samples.append((np.ascontiguousarray(clips[w]), np.ascontiguousarray(actions[w])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        clip_wp, act_wp = self.samples[index]
        return _sample(clip_wp, act_wp, self.num_context)

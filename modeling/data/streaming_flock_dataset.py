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


class StreamingLatentEncoder:
    """Bridge the pixel stream to the latent-space world model, on the GPU.

        {pixels (B,P,T,H,W,3) uint8, actions (B,P,T,A), context_len}
          -> {frames (B,P,L,z,h,w) normalized latents, actions (B,P,L,A), context_len}

    The frozen VAE runs HERE (main process, GPU), not in the CPU dataloader workers.
    Streaming has no precompute / ``stats.pt``, so per-channel latent mean/std are
    computed once via :meth:`fit_stats` from a few calibration batches and reused.
    """

    def __init__(self, vae, device, image_size=(128, 128)):
        self.vae = vae
        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        self.z_dim = int(vae.z_dim)
        self.mean = None   # (1,1,1,z,1,1) after fit_stats
        self.std = None

    @staticmethod
    def _agg_actions(actions: torch.Tensor, t_lat: int) -> torch.Tensor:
        # (T_pix, A) -> (t_lat, A), causal 1+4k groups (matches precompute.aggregate_actions)
        groups = [actions[0:1]]
        for i in range(t_lat - 1):
            groups.append(actions[1 + 4 * i: 1 + 4 * (i + 1)])
        return torch.stack([g.mean(0) for g in groups])

    @torch.no_grad()
    def _encode_raw(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B,P,T,H,W,3) uint8 -> raw (un-normalized) latents (B,P,L,z,h,w)."""
        B, P, T, H, W, _ = pixels.shape
        if (H, W) != self.image_size:
            raise ValueError(f"clip size {(H, W)} != VAE image_size {self.image_size}")
        x = pixels.to(self.device, torch.float32).div_(255.0).mul_(2.0).sub_(1.0)  # [-1,1]
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(B * P, 3, T, H, W)                 # (B*P,3,T,H,W)
        z = self.vae.encode(x)                                                     # (B*P,z,L,h,w)
        z = z.reshape(B, P, *z.shape[1:])                                          # (B,P,z,L,h,w)
        return z.permute(0, 1, 3, 2, 4, 5).contiguous()                            # (B,P,L,z,h,w)

    @torch.no_grad()
    def fit_stats(self, pixel_batches) -> None:
        """Set per-channel latent mean/std from a few calibration pixel batches."""
        csum = torch.zeros(self.z_dim, dtype=torch.float64, device=self.device)
        csq = torch.zeros(self.z_dim, dtype=torch.float64, device=self.device)
        count = 0
        for pixels in pixel_batches:
            zf = self._encode_raw(pixels).permute(3, 0, 1, 2, 4, 5).reshape(self.z_dim, -1).double()
            csum += zf.sum(1); csq += (zf * zf).sum(1); count += zf.shape[1]
        mean = (csum / count).float()
        std = (csq / count - (csum / count) ** 2).clamp_min(1e-8).sqrt().float()
        self.mean = mean.view(1, 1, 1, -1, 1, 1)
        self.std = std.view(1, 1, 1, -1, 1, 1)

    @torch.no_grad()
    def encode_batch(self, batch: dict) -> dict:
        """Streamed pixel batch -> the (frames, actions, context_len) dict the trainer expects."""
        if self.mean is None:
            raise RuntimeError("call fit_stats() before encode_batch()")
        z = self._encode_raw(batch["pixels"])                       # (B,P,L,z,h,w)
        z = (z - self.mean.to(z.device)) / self.std.to(z.device)
        L = z.shape[2]
        acts = batch["actions"].to(self.device, torch.float32)      # (B,P,T,A)
        B, P = acts.shape[:2]
        lat = torch.stack([
            torch.stack([self._agg_actions(acts[b, p], L) for p in range(P)])
            for b in range(B)
        ])                                                          # (B,P,L,A)
        ctx = batch["context_len"]
        ctx = int(ctx[0]) if torch.is_tensor(ctx) else int(ctx)
        return {"frames": z, "actions": lat, "context_len": ctx}

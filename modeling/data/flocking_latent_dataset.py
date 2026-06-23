"""Latent-space dataset for FlockDiT: reads cached VAE latents, samples a
latent-frame window, normalizes to ~unit scale.

Returns the same dict shape the FlowTrainer/world-model expect, but the
"frames" are normalized VAE latents `(L, z_dim, h, w)` rather than RGB pixels.
Train/val split mirrors FlockingVideoDataset (seeded split by episode id).
Run ``precompute_latents.py`` first to populate the cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FlockingLatentDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        val_fraction: float,
        num_context_frames: int,
        num_future_frames: int,
        random_clip: bool = True,
        split_seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        self.num_context = int(num_context_frames)
        self.window = int(num_context_frames) + int(num_future_frames)
        self.random_clip = bool(random_clip)

        files = sorted(self.cache_dir.glob("ep*_a*.pt"))
        if not files:
            raise FileNotFoundError(
                f"No cached latents in {self.cache_dir}. Run precompute_latents.py first."
            )
        stats = torch.load(self.cache_dir / "stats.pt", map_location="cpu")
        self.mean = stats["mean"].view(-1, 1, 1, 1)  # (z,1,1,1)
        self.std = stats["std"].view(-1, 1, 1, 1)

        # seeded split by episode id (matches FlockingVideoDataset semantics)
        ep_of = lambda p: p.name.split("_a")[0]  # "ep00000"
        episodes = sorted({ep_of(p) for p in files})
        rng = np.random.default_rng(int(split_seed))
        shuffled = list(rng.permutation(episodes))
        val_count = int(round(len(shuffled) * float(val_fraction)))
        val_ids = set(shuffled[:val_count])
        keep_val = split in {"val", "valid", "validation"}
        self.files = [p for p in files if (ep_of(p) in val_ids) == keep_val]
        if not self.files:
            raise ValueError(f"No latent samples for split={split} in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _start(self, t_lat: int, index: int) -> int:
        high = t_lat - self.window
        if high <= 0:
            return 0
        if self.random_clip:
            return int(np.random.randint(0, high + 1))
        return int((index * 9973) % (high + 1))

    def __getitem__(self, index: int) -> dict:
        d = torch.load(self.files[index], map_location="cpu")
        z = d["latents"].float()        # (z_dim, T_lat, h, w)
        acts = d["actions"].float()     # (T_lat, A)
        t_lat = z.shape[1]
        if t_lat < self.window:
            raise ValueError(f"{self.files[index].name} has {t_lat} latent frames < window {self.window}")
        s = self._start(t_lat, index)
        z = z[:, s:s + self.window]                 # (z_dim, L, h, w)
        acts = acts[s:s + self.window]              # (L, A)
        z = (z - self.mean) / self.std              # normalize per channel
        return {
            "frames": z.permute(1, 0, 2, 3).contiguous(),  # (L, z_dim, h, w) = (F,C,H,W)
            "actions": acts,
            "context_len": self.num_context,
            "episode_id": d["episode_id"],
            "agent_index": d["agent_index"],
        }


def build_latent_dataloader(cfg, split: str) -> DataLoader:
    data = cfg.data
    cache_dir = Path(data.root) / data.get("latent_cache_dir", "latent_cache")
    dataset = FlockingLatentDataset(
        cache_dir=cache_dir,
        split=split,
        val_fraction=data.val_fraction,
        num_context_frames=data.num_context_frames,
        num_future_frames=data.num_future_frames,
        random_clip=bool(data.get("random_clip", True)) and split == "train",
        split_seed=cfg.seed,
    )
    nw = int(cfg.dataloader.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=(split == "train"),
        num_workers=nw,
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.persistent_workers) and nw > 0,
        drop_last=bool(cfg.dataloader.drop_last) if split == "train" else False,
    )

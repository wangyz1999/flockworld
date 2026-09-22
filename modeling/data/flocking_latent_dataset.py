"""Latent-space dataset for FlockDiT: reads cached VAE latents, samples a
latent-frame window, normalizes to ~unit scale.

Returns the same dict shape the FlowTrainer/world-model expect, but the
"frames" are normalized VAE latents `(L, z_dim, h, w)` rather than RGB pixels.
Train/val split mirrors FlockingVideoDataset (seeded split by episode id).
Run ``modeling/cli/precompute_latents.py`` first to populate the cache.
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
                f"No cached latents in {self.cache_dir}. Run modeling/cli/precompute_latents.py first."
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

    def full_episode(self, index: int) -> dict:
        """Whole normalized latent trajectory + all actions (for long AR rollout).

        Unlike __getitem__ (a single window), this returns every latent frame so
        a sliding-window rollout can run far past the trained clip length.
        """
        d = torch.load(self.files[index], map_location="cpu")
        z = (d["latents"].float() - self.mean) / self.std  # (z_dim, T_lat, h, w)
        return {
            "frames": z.permute(1, 0, 2, 3).contiguous(),  # (T_lat, z_dim, h, w)
            "actions": d["actions"].float(),               # (T_lat, A)
            "context_len": self.num_context,
            "episode_id": d["episode_id"],
            "agent_index": d["agent_index"],
        }


class FlockingLatentMultiDataset(FlockingLatentDataset):
    """Multi-agent latent: returns ``frames (P, L, z_dim, h, w)`` + ``actions (P, L, A)``.

    Groups the split's cached files by episode and keeps the first ``num_agents``
    agents (sorted by agent index). All agents share one sampled latent window so
    their views are temporally aligned -- the layout the multi-agent FlockDiT path
    expects (``x: (B, P, F, C, H, W)``).
    """

    def __init__(self, *args, num_agents: int = 2, **kwargs):
        self.num_agents = int(num_agents)
        super().__init__(*args, **kwargs)
        ep_of = lambda p: p.name.split("_a")[0]          # "ep00000"
        ag_of = lambda p: int(p.stem.split("_a")[1])     # 1..10 (int, not string-sorted)
        by_episode: dict[str, list] = {}
        for p in self.files:
            by_episode.setdefault(ep_of(p), []).append(p)
        self.episodes = [
            sorted(v, key=ag_of)[: self.num_agents]
            for v in by_episode.values()
            if len(v) >= self.num_agents
        ]
        if not self.episodes:
            raise ValueError(
                f"No episodes with >= {self.num_agents} cached agents in {self.cache_dir}"
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        paths = self.episodes[index]
        ds = [torch.load(p, map_location="cpu") for p in paths]
        zs = [d["latents"].float() for d in ds]              # each (z_dim, T_lat, h, w)
        t_lat = min(z.shape[1] for z in zs)
        if t_lat < self.window:
            raise ValueError(
                f"episode {ds[0]['episode_id']} has {t_lat} latent frames < window {self.window}"
            )
        s = self._start(t_lat, index)                        # one shared window for all agents
        frames, actions = [], []
        for d, z in zip(ds, zs):
            zz = (z[:, s:s + self.window] - self.mean) / self.std      # (z_dim, L, h, w)
            frames.append(zz.permute(1, 0, 2, 3).contiguous())        # (L, z_dim, h, w)
            actions.append(d["actions"].float()[s:s + self.window])   # (L, A)
        return {
            "frames": torch.stack(frames, dim=0),     # (P, L, z_dim, h, w)
            "actions": torch.stack(actions, dim=0),   # (P, L, A)
            "context_len": self.num_context,
            "episode_id": ds[0]["episode_id"],
            "agent_indices": [d["agent_index"] for d in ds],
        }

    def full_episode(self, index: int) -> dict:
        """Whole normalized latent trajectories for all P agents of one episode.

        Multi-agent analogue of FlockingLatentDataset.full_episode: returns every
        latent frame (not a window) so a sliding-window rollout can run the full
        episode's P views. All agents share the same length (same episode).
        """
        paths = self.episodes[index]
        ds = [torch.load(p, map_location="cpu") for p in paths]
        zs, acts = [], []
        for d in ds:
            z = (d["latents"].float() - self.mean) / self.std     # (z_dim, T_lat, h, w)
            zs.append(z.permute(1, 0, 2, 3).contiguous())         # (T_lat, z_dim, h, w)
            acts.append(d["actions"].float())                     # (T_lat, A)
        t = min(z.shape[0] for z in zs)                            # robust to any length mismatch
        return {
            "frames": torch.stack([z[:t] for z in zs], dim=0),    # (P, T_lat, z_dim, h, w)
            "actions": torch.stack([a[:t] for a in acts], dim=0), # (P, T_lat, A)
            "context_len": self.num_context,
            "episode_id": ds[0]["episode_id"],
            "agent_indices": [d["agent_index"] for d in ds],
        }


def build_latent_dataloader(cfg, split: str) -> DataLoader:
    data = cfg.data
    cache_dir = Path(data.root) / data.get("latent_cache_dir", "latent_cache")
    num_agents = int(data.get("num_agents", 1))
    common = dict(
        cache_dir=cache_dir,
        split=split,
        val_fraction=data.val_fraction,
        num_context_frames=data.num_context_frames,
        num_future_frames=data.num_future_frames,
        random_clip=bool(data.get("random_clip", True)) and split == "train",
        split_seed=cfg.seed,
    )
    if num_agents > 1:
        dataset = FlockingLatentMultiDataset(num_agents=num_agents, **common)
    else:
        dataset = FlockingLatentDataset(**common)
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

"""Datasets for the FlockDiT flow-matching world model.

Both classes reuse :class:`FlockingVideoDataset`'s index building, video
decoding and action loading, but return a **single contiguous clip** (context +
future together) from each agent's own ``video_a*`` partial view, plus per-frame
actions for *all* frames -- the layout the Diffusion-Forcing trainer expects.

* :class:`FlockingDiTDataset` -- single-agent: one ``(episode, agent)`` per item.
* :class:`FlockingDiTMultiDataset` -- multi-agent: one episode per item, with
  ``num_agents`` agents stacked on a leading ``P`` axis (shared frame window so
  the views are temporally aligned).
"""

from __future__ import annotations

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader

from modeling.data.flocking_dataset import FlockingVideoDataset


class FlockingDiTDataset(FlockingVideoDataset):
    """Single-agent: returns ``frames (F,3,H,W)`` + ``actions (F,A)``."""

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        reader = VideoReader(str(sample.partial_video), ctx=cpu(0), num_threads=1)
        max_len = min(len(reader), self._state_action_len(sample.state_action))
        if max_len < self.clip_span:
            raise ValueError(
                f"Episode {sample.episode_id} has {max_len} aligned frames, "
                f"but {self.clip_span} are required."
            )
        start = self._sample_start(max_len, index)
        frame_ids = np.arange(start, start + self.clip_span, self.frame_stride, dtype=np.int64)
        return {
            "frames": self._load_video_clip(reader, frame_ids),  # (F,3,H,W) in [-1,1]
            "actions": self._load_actions(sample.state_action, sample.agent_index, frame_ids),  # (F,A)
            "context_len": self.num_context_frames,
            "episode_id": sample.episode_id,
            "agent_index": sample.agent_index,
        }


class FlockingDiTMultiDataset(FlockingVideoDataset):
    """Multi-agent: returns ``frames (P,F,3,H,W)`` + ``actions (P,F,A)``.

    Groups the base per-(episode, agent) index by episode and keeps the first
    ``num_agents`` agents (sorted by index) that exist for that episode.
    """

    def __init__(self, *args, num_agents: int = 2, **kwargs):
        self.num_agents = int(num_agents)
        super().__init__(*args, **kwargs)
        # Regroup the flat sample list into per-episode agent lists.
        by_episode: dict[str, list] = {}
        for s in self.samples:
            by_episode.setdefault(s.episode_id, []).append(s)
        self.episodes = [
            sorted(v, key=lambda s: s.agent_index)[: self.num_agents]
            for v in by_episode.values()
            if len(v) >= self.num_agents
        ]
        if not self.episodes:
            raise ValueError(
                f"No episodes with >= {self.num_agents} agents under {self.root}"
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        agents = self.episodes[index]
        ep = agents[0]
        # Align all agents on the same frame window (shared parquet length).
        readers = [VideoReader(str(a.partial_video), ctx=cpu(0), num_threads=1) for a in agents]
        max_len = min(
            min(len(r) for r in readers), self._state_action_len(ep.state_action)
        )
        if max_len < self.clip_span:
            raise ValueError(
                f"Episode {ep.episode_id} has {max_len} aligned frames, "
                f"but {self.clip_span} are required."
            )
        start = self._sample_start(max_len, index)
        frame_ids = np.arange(start, start + self.clip_span, self.frame_stride, dtype=np.int64)
        frames = torch.stack(
            [self._load_video_clip(r, frame_ids) for r in readers], dim=0
        )  # (P,F,3,H,W)
        actions = torch.stack(
            [self._load_actions(ep.state_action, a.agent_index, frame_ids) for a in agents],
            dim=0,
        )  # (P,F,A)
        return {
            "frames": frames,
            "actions": actions,
            "context_len": self.num_context_frames,
            "episode_id": ep.episode_id,
            "agent_indices": [a.agent_index for a in agents],
        }


def build_dit_dataloader(cfg, split: str) -> DataLoader:
    """Build a DiT dataloader: latent, single-, or multi-agent (per config)."""
    if bool(cfg.data.get("latent", False)):
        from modeling.data.flocking_latent_dataset import build_latent_dataloader
        return build_latent_dataloader(cfg, split)
    data = cfg.data
    num_agents = int(data.get("num_agents", 1))
    random_clip = bool(data.get("random_clip", True)) and split == "train"
    common = dict(
        root=data.root,
        split=split,
        val_fraction=data.val_fraction,
        num_context_frames=data.num_context_frames,
        num_future_frames=data.num_future_frames,
        frame_stride=data.frame_stride,
        image_size=data.image_size,
        action_features=list(data.action_features),
        partial_agent_indices=data.get("partial_agent_indices", None),
        random_clip=random_clip,
        split_seed=cfg.seed,
    )
    if num_agents > 1:
        dataset = FlockingDiTMultiDataset(num_agents=num_agents, **common)
    else:
        dataset = FlockingDiTDataset(**common)

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

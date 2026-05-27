from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PartialClipSample:
    episode_id: str
    video_path: Path
    agent_index: int


class PartialVideoVAEDataset(Dataset):
    """Loads short clips from each agent's partial observation video.

    Each item returns a tensor shaped ``(C, T, H, W)`` in ``[-1, 1]``, suitable
    for the Wan VAE encoder. No actions or global frames are used.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        val_fraction: float,
        num_frames: int,
        frame_stride: int,
        image_size: tuple[int, int] | list[int] | None,
        partial_agent_indices: list[int] | None = None,
        random_clip: bool = True,
        split_seed: int = 42,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.image_size = tuple(image_size) if image_size else None
        self.random_clip = bool(random_clip)
        self.clip_span = (self.num_frames - 1) * self.frame_stride + 1

        self.samples = self._build_index(
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            partial_agent_indices=partial_agent_indices,
        )
        if not self.samples:
            raise ValueError(f"No partial-agent videos found under {self.root}")
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        reader = VideoReader(str(sample.video_path), ctx=cpu(0))
        max_len = len(reader)
        if max_len < self.clip_span:
            raise ValueError(
                f"Video {sample.video_path} has {max_len} frames, "
                f"but {self.clip_span} are required."
            )

        start = self._sample_start(max_len, index)
        frame_ids = np.arange(start, start + self.clip_span, self.frame_stride, dtype=np.int64)
        clip = self._load_video_clip(reader, frame_ids)
        # (T, C, H, W) -> (C, T, H, W) for WanVAE
        clip = clip.permute(1, 0, 2, 3).contiguous()
        return {
            "video": clip,
            "episode_id": sample.episode_id,
            "agent_index": sample.agent_index,
        }

    def _build_index(
        self,
        val_fraction: float,
        split_seed: int,
        partial_agent_indices: list[int] | None,
    ) -> list[PartialClipSample]:
        agent_dirs = self._agent_dirs(partial_agent_indices)
        if not agent_dirs:
            raise FileNotFoundError(f"No video_a* directories found under {self.root}")

        episode_ids = sorted({p.stem for _, d in agent_dirs for p in d.glob("*.mp4")})
        rng = np.random.default_rng(split_seed)
        shuffled = list(rng.permutation(episode_ids))
        val_count = int(round(len(shuffled) * val_fraction))
        val_ids = set(shuffled[:val_count])
        keep_val = self.split in {"val", "valid", "validation"}
        selected_ids = [eid for eid in episode_ids if (eid in val_ids) == keep_val]

        samples: list[PartialClipSample] = []
        for episode_id in selected_ids:
            for agent_index, agent_dir in agent_dirs:
                video_path = agent_dir / f"{episode_id}.mp4"
                if video_path.exists():
                    samples.append(
                        PartialClipSample(
                            episode_id=episode_id,
                            video_path=video_path,
                            agent_index=agent_index,
                        )
                    )
        return samples

    def _agent_dirs(self, partial_agent_indices: list[int] | None) -> list[tuple[int, Path]]:
        if partial_agent_indices is None:
            dirs = sorted(
                (p for p in self.root.glob("video_a*") if p.is_dir()),
                key=lambda p: int(p.name.removeprefix("video_a")),
            )
            return [(int(p.name.removeprefix("video_a")), p) for p in dirs]
        return [(int(i), self.root / f"video_a{int(i)}") for i in partial_agent_indices]

    def _sample_start(self, max_len: int, index: int) -> int:
        high = max_len - self.clip_span
        if high <= 0:
            return 0
        if self.random_clip and self.split == "train":
            return int(np.random.randint(0, high + 1))
        return int((index * 9973) % (high + 1))

    def _load_video_clip(self, reader: VideoReader, frame_ids: np.ndarray) -> torch.Tensor:
        frames = reader.get_batch(frame_ids.tolist()).asnumpy()
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        if self.image_size is not None:
            video = F.interpolate(video, size=self.image_size, mode="bilinear", align_corners=False)
        return video * 2.0 - 1.0

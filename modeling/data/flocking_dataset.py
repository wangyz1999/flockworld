from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from decord import VideoReader, cpu


@dataclass(frozen=True)
class EpisodeSample:
    episode_id: str
    global_video: Path
    partial_video: Path
    state_action: Path
    agent_index: int


class FlockingVideoDataset(Dataset):
    """Dataset for multi-agent action-conditioned video generation.

    Each item returns a clip from one episode and one conditioned agent:
    ``context_video`` is the global scene history, ``partial_video`` is the
    conditioned agent's local observation history, ``actions`` are trajectory
    features for the prediction horizon, and ``target_video`` is the future
    global video to predict.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        val_fraction: float,
        num_context_frames: int,
        num_future_frames: int,
        frame_stride: int,
        image_size: tuple[int, int] | list[int] | None,
        action_features: list[str],
        partial_agent_indices: list[int] | None = None,
        random_clip: bool = True,
        split_seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.num_context_frames = int(num_context_frames)
        self.num_future_frames = int(num_future_frames)
        self.frame_stride = int(frame_stride)
        self.image_size = tuple(image_size) if image_size else None
        self.action_features = list(action_features)
        self.random_clip = bool(random_clip)
        self.clip_frames = self.num_context_frames + self.num_future_frames
        self.clip_span = (self.clip_frames - 1) * self.frame_stride + 1

        self.samples = self._build_index(
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            partial_agent_indices=partial_agent_indices,
        )
        if not self.samples:
            raise ValueError(f"No FlockWorld samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        global_reader = VideoReader(str(sample.global_video), ctx=cpu(0))
        partial_reader = VideoReader(str(sample.partial_video), ctx=cpu(0))
        max_len = min(len(global_reader), len(partial_reader), self._state_action_len(sample.state_action))
        if max_len < self.clip_span:
            raise ValueError(
                f"Episode {sample.episode_id} has {max_len} aligned frames, "
                f"but {self.clip_span} are required."
            )

        start = self._sample_start(max_len, index)
        frame_ids = np.arange(start, start + self.clip_span, self.frame_stride, dtype=np.int64)

        global_video = self._load_video_clip(global_reader, frame_ids)
        partial_video = self._load_video_clip(partial_reader, frame_ids)
        actions = self._load_actions(sample.state_action, sample.agent_index, frame_ids)

        return {
            "context_video": global_video[: self.num_context_frames],
            "partial_video": partial_video[: self.num_context_frames],
            "actions": actions[self.num_context_frames :],
            "target_video": global_video[self.num_context_frames :],
            "episode_id": sample.episode_id,
            "agent_index": sample.agent_index,
        }

    def _build_index(
        self,
        val_fraction: float,
        split_seed: int,
        partial_agent_indices: list[int] | None,
    ) -> list[EpisodeSample]:
        global_dir = self.root / "video_global"
        state_action_dir = self.root / "state_action"
        if not global_dir.is_dir() or not state_action_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {global_dir} and {state_action_dir} from data_recording.py collection output."
            )

        episode_ids = sorted(p.stem for p in global_dir.glob("*.mp4") if (state_action_dir / f"{p.stem}.parquet").exists())
        rng = np.random.default_rng(split_seed)
        shuffled = list(rng.permutation(episode_ids))
        val_count = int(round(len(shuffled) * val_fraction))
        val_ids = set(shuffled[:val_count])
        keep_val = self.split in {"val", "valid", "validation"}
        selected_ids = [episode_id for episode_id in episode_ids if (episode_id in val_ids) == keep_val]

        agent_dirs = self._agent_dirs(partial_agent_indices)
        samples: list[EpisodeSample] = []
        for episode_id in selected_ids:
            for agent_index, agent_dir in agent_dirs:
                partial_video = agent_dir / f"{episode_id}.mp4"
                if partial_video.exists():
                    samples.append(
                        EpisodeSample(
                            episode_id=episode_id,
                            global_video=global_dir / f"{episode_id}.mp4",
                            partial_video=partial_video,
                            state_action=state_action_dir / f"{episode_id}.parquet",
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

    def _load_actions(self, state_action_path: Path, agent_index: int, frame_ids: np.ndarray) -> torch.Tensor:
        prefix = f"a{agent_index}"
        required_columns = self._required_action_columns(prefix)
        df = pl.read_parquet(state_action_path, columns=required_columns)

        values = []
        for name in self.action_features:
            column = f"{prefix}_{name}"
            if column in df.columns:
                values.append(df[column].to_numpy())
            elif name == "action_x":
                values.append(np.cos(df[f"{prefix}_action"].to_numpy()))
            elif name == "action_y":
                values.append(np.sin(df[f"{prefix}_action"].to_numpy()))
            else:
                raise KeyError(
                    f"Feature {name!r} is not available in {state_action_path}. "
                    f"Expected column {column!r} or a supported derived feature."
                )

        values = np.stack(values, axis=-1)[frame_ids]
        return torch.from_numpy(values.astype(np.float32, copy=False))

    @staticmethod
    def _state_action_len(state_action_path: Path) -> int:
        return pl.scan_parquet(state_action_path).select(pl.len()).collect().item()

    def _required_action_columns(self, prefix: str) -> list[str]:
        columns = []
        for name in self.action_features:
            if name in {"action_x", "action_y"}:
                column = f"{prefix}_action"
            else:
                column = f"{prefix}_{name}"
            if column not in columns:
                columns.append(column)
        return columns

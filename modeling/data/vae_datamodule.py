from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from modeling.data.vae_dataset import PartialVideoVAEDataset


class PartialVideoVAEDataModule(L.LightningDataModule):
    """Lightning DataModule serving partial-agent observation clips for VAE training."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.streaming = bool(cfg.data.get("streaming", {}).get("enabled", False))
        self.train_set = None
        self.val_set = None

    def setup(self, stage: str | None = None) -> None:
        if self.streaming:
            self._setup_streaming(stage)
            return
        cfg = self.cfg
        # Overfit mode: cap each split to N samples and freeze the clip window.
        overfit_n = cfg.data.get("overfit_n", None)
        random_train_clip = not (overfit_n is not None and int(overfit_n) > 0)
        if stage in (None, "fit"):
            self.train_set = self._build("train", random_clip=random_train_clip, max_samples=overfit_n)
            self.val_set = self._build("val", random_clip=False, max_samples=overfit_n)
        elif stage == "validate":
            self.val_set = self._build("val", random_clip=False, max_samples=overfit_n)

    def _setup_streaming(self, stage: str | None) -> None:
        from modeling.data.streaming_vae_dataset import (
            InMemoryClipDataset,
            SimClipGenerator,
            StreamingPartialVAEDataset,
            load_sim_cfg_container,
        )

        cfg = self.cfg
        s = cfg.data.streaming
        sim_container = load_sim_cfg_container(
            str(s.sim_config), list(s.get("sim_overrides", []) or [])
        )

        def make_generator() -> SimClipGenerator:
            return SimClipGenerator(
                sim_cfg_container=sim_container,
                num_frames=cfg.data.num_frames,
                frame_stride=cfg.data.frame_stride,
                num_partial_agents=s.num_partial_agents,
                windows_per_episode=s.windows_per_episode,
                warmup_steps=s.warmup_steps,
                threads=s.get("threads_per_worker", None),
            )

        if self.val_set is None:
            # Generated once in the main process; the train loader uses spawn
            # workers, so the JAX runtime initialised here is never forked.
            self.val_set = InMemoryClipDataset(
                make_generator(),
                num_clips=int(s.val_clips),
                image_size=cfg.data.image_size,
                base_seed=int(cfg.seed),
            )
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = StreamingPartialVAEDataset(
                make_generator(),
                clips_per_epoch=int(s.clips_per_epoch),
                image_size=cfg.data.image_size,
                base_seed=int(cfg.seed),
            )

    def _build(self, split: str, random_clip: bool, max_samples: int | None) -> PartialVideoVAEDataset:
        cfg = self.cfg
        return PartialVideoVAEDataset(
            root=cfg.data.root,
            split=split,
            val_fraction=cfg.data.val_fraction,
            num_frames=cfg.data.num_frames,
            frame_stride=cfg.data.frame_stride,
            image_size=cfg.data.image_size,
            partial_agent_indices=cfg.data.partial_agent_indices,
            random_clip=random_clip,
            split_seed=cfg.seed,
            max_samples=max_samples,
        )

    def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        cfg = self.cfg
        num_workers = int(cfg.dataloader.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(cfg.dataloader.pin_memory),
            persistent_workers=bool(cfg.dataloader.persistent_workers) and num_workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        if self.streaming:
            cfg = self.cfg
            num_workers = int(cfg.dataloader.num_workers)
            return DataLoader(
                self.train_set,
                batch_size=int(cfg.dataloader.batch_size),
                shuffle=False,  # IterableDataset: workers own disjoint seed streams
                num_workers=num_workers,
                pin_memory=bool(cfg.dataloader.pin_memory),
                # spawn: workers must not inherit the main process's JAX/CUDA
                # state via fork; persistent workers keep per-worker JIT caches
                # and the never-repeat episode counters alive across epochs.
                multiprocessing_context="spawn" if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
                drop_last=bool(cfg.dataloader.drop_last),
            )
        return self._loader(self.train_set, shuffle=True, drop_last=bool(self.cfg.dataloader.drop_last))

    def val_dataloader(self) -> DataLoader:
        if self.streaming:
            # In-memory clips: plain indexing, no workers needed.
            return DataLoader(
                self.val_set,
                batch_size=int(self.cfg.dataloader.batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=bool(self.cfg.dataloader.pin_memory),
            )
        return self._loader(self.val_set, shuffle=False, drop_last=False)

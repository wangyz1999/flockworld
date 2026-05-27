from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from modeling.data.vae_dataset import PartialVideoVAEDataset


class PartialVideoVAEDataModule(L.LightningDataModule):
    """Lightning DataModule serving partial-agent observation clips for VAE training."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_set: PartialVideoVAEDataset | None = None
        self.val_set: PartialVideoVAEDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.cfg
        # Overfit mode: cap each split to N samples and freeze the clip window.
        overfit_n = cfg.data.get("overfit_n", None)
        random_train_clip = not (overfit_n is not None and int(overfit_n) > 0)
        if stage in (None, "fit"):
            self.train_set = self._build("train", random_clip=random_train_clip, max_samples=overfit_n)
            self.val_set = self._build("val", random_clip=False, max_samples=overfit_n)
        elif stage == "validate":
            self.val_set = self._build("val", random_clip=False, max_samples=overfit_n)

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
        return self._loader(self.train_set, shuffle=True, drop_last=bool(self.cfg.dataloader.drop_last))

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False, drop_last=False)

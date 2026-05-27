from __future__ import annotations

from torch.utils.data import DataLoader

from modeling.data.flocking_dataset import FlockingVideoDataset


def build_dataloader(cfg, split: str) -> DataLoader:
    data_cfg = cfg.data.copy()
    data_cfg.split = split
    if split != "train":
        data_cfg.random_clip = False

    dataset = FlockingVideoDataset(
        root=data_cfg.root,
        split=data_cfg.split,
        val_fraction=data_cfg.val_fraction,
        num_context_frames=data_cfg.num_context_frames,
        num_future_frames=data_cfg.num_future_frames,
        frame_stride=data_cfg.frame_stride,
        image_size=data_cfg.image_size,
        action_features=list(data_cfg.action_features),
        partial_agent_indices=data_cfg.partial_agent_indices,
        random_clip=data_cfg.random_clip,
        split_seed=cfg.seed,
    )

    num_workers = int(cfg.dataloader.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.persistent_workers) and num_workers > 0,
        drop_last=bool(cfg.dataloader.drop_last) if split == "train" else False,
    )


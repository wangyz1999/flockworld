"""Train the FlockDiT flow-matching world model.

Usage (always pass --config; modeling.configs defaults to a non-existent file):

    uv run python train_flock_dit.py --config config/train_flockdit.yaml \
        device=cpu dataloader.batch_size=1 train.epochs=1

Single- vs multi-agent is selected by ``data.num_agents`` in the config.
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from modeling.configs import load_cfg
from modeling.data.flocking_dit_dataset import build_dit_dataloader
from modeling.models.flock_dit import FlockDiT
from modeling.training.flow_trainer import FlowTrainer
from modeling.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train the FlockDiT flow-matching world model.")
    p.add_argument("--config", default=None, help="Path to a YAML config.")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config, args.overrides)
    seed_everything(int(cfg.seed))

    train_loader = build_dit_dataloader(cfg, split="train")
    val_loader = build_dit_dataloader(cfg, split="val")

    num_agents = int(cfg.data.get("num_agents", 1))
    action_dim = len(list(cfg.data.action_features))
    m = cfg.model
    model = FlockDiT(
        in_channels=int(m.in_channels),
        out_channels=int(m.out_channels),
        dim=int(m.dim),
        depth=int(m.depth),
        heads=int(m.heads),
        ffn_dim=int(m.ffn_dim),
        patch=int(m.patch),
        patch_t=int(m.get("patch_t", 1)),
        action_dim=action_dim,
        num_agents=num_agents,
        local_attn_size=int(m.get("local_attn_size", -1)),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(OmegaConf.to_yaml(cfg))
    print(f"FlockDiT params: {n_params/1e6:.1f}M | num_agents={num_agents}")
    FlowTrainer(cfg, model, train_loader, val_loader).fit()


if __name__ == "__main__":
    # decord's thread pool deadlocks under DataLoader's default 'fork' start
    # method (num_workers>0), so use 'spawn' -- workers start fresh, no inherited
    # locks/threads. Harmless when num_workers=0.
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

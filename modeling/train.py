from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from modeling.configs import load_cfg
from modeling.data import build_dataloader
from modeling.models import FlockWM
from modeling.training import Trainer
from modeling.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train an action-conditioned FlockWorld video model.")
    parser.add_argument("--config", default=None, help="Path to a YAML config. Defaults to modeling/config/train.yaml.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.epochs=1 data.root=...")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config, args.overrides)
    seed_everything(int(cfg.seed))

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    model = FlockWM(
        in_channels=int(cfg.model.in_channels),
        hidden_channels=int(cfg.model.hidden_channels),
        action_dim=int(cfg.model.action_dim),
        action_hidden_dim=int(cfg.model.action_hidden_dim),
        out_channels=int(cfg.model.out_channels),
    )
    print(OmegaConf.to_yaml(cfg))
    Trainer(cfg, model, train_loader, val_loader).fit()


if __name__ == "__main__":
    main()

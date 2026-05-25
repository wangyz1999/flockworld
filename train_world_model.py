"""Entry point for training the FlockWorld action-conditioned video model.

Run from the repository root:

    uv run train_world_model.py

Configuration defaults to ``config/train.yaml``. Override values with
OmegaConf dotlist syntax, for example:

    uv run train_world_model.py train.epochs=1 dataloader.batch_size=1
"""

from __future__ import annotations

from modeling.train import main


if __name__ == "__main__":
    main()

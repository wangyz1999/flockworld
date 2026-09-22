from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


def load_cfg(config_path: str | Path | None = None, overrides: Iterable[str] | None = None):
    """Load YAML config and optional dotlist overrides.

    Returned configs are OmegaConf objects, so nested keys can be accessed as
    ``cfg.data.root`` or ``cfg.train.epochs``.
    With no path, load the legacy baseline's ``config/train_wm.yaml``.
    """
    path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / "config" / "train_wm.yaml"
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg

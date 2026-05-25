"""Entry point for training the Solaris world model on FlockWorld data.

Run via ``python train_world_model.py`` from the repository root. Hydra
configuration lives under ``config/modeling/``; override with
``+key=value`` syntax as usual.

Example::

    python train_world_model.py runner.params.total_steps=10 \
        device.batch_size=1 num_frames_context=9
"""

import hydra
import wandb
from absl import logging
from omegaconf import OmegaConf

from modeling.utils.config import instantiate_from_config, resolve_device_paths
from modeling.utils.jax import init_jax_distributed


def init_wandb(cfg):
    import jax

    if not cfg.runner.params.use_wandb or jax.process_index() != 0:
        return

    wandb.login()
    wandb.init(
        project=cfg.wandb_project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.experiment_name,
        entity=cfg.wandb_entity,
        tags=cfg.wandb_tags,
        id=cfg.experiment_name,
    )


@hydra.main(
    config_path="config/modeling",
    config_name="train",
    version_base=None,
)
def main(cfg):
    resolve_device_paths(cfg)

    import jax

    from modeling.utils.jax import setup_jax_cache

    if cfg.enable_jax_cache:
        setup_jax_cache(cfg.device.jax_cache_dir)

    init_jax_distributed(cfg)

    if jax.process_index() != 0:
        logging.set_verbosity(logging.ERROR)
        cfg.runner.params.use_wandb = False

    runner = instantiate_from_config(cfg.runner)
    init_wandb(cfg)
    runner.run()


if __name__ == "__main__":
    main()

"""Train the FlockDiT flow-matching world model.

Usage (pass the experiment configuration with --config):

    uv run python -m modeling.cli.train_flock_dit --config config/train_flockdit.yaml \
        device=cpu dataloader.batch_size=1 train.epochs=1

Single- vs multi-agent is selected by ``data.num_agents`` in the config.
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from modeling.configs import load_cfg
from modeling.data.flocking_dit_dataset import build_dit_dataloader
from modeling.eval.common import build_model
from modeling.models.decoding import build_decode_fn
from modeling.training.flow_trainer import FlowTrainer
from modeling.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train the FlockDiT flow-matching world model.")
    p.add_argument("--config", required=True, help="Path to a YAML config.")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config, args.overrides)
    seed_everything(int(cfg.seed))

    train_loader = build_dit_dataloader(cfg, split="train")
    val_loader = build_dit_dataloader(cfg, split="val")

    num_agents = int(cfg.data.get("num_agents", 1))
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(OmegaConf.to_yaml(cfg))
    latent = bool(cfg.data.get("latent", False))
    st = cfg.data.get("streaming", None)
    streaming = st is not None and bool(st.get("enabled", False))
    print(f"FlockDiT params: {n_params/1e6:.1f}M | num_agents={num_agents} | latent={latent} | streaming={streaming}")

    stream_encoder = None
    if streaming:
        import torch
        from modeling.models.frozen_vae import FrozenVAE
        from modeling.data.streaming_flock_dataset import StreamingLatentEncoder
        dev = "cuda" if (torch.cuda.is_available() and str(cfg.device) != "cpu") else "cpu"
        vae = FrozenVAE(str(cfg.vae.checkpoint_path), device=dev)
        stream_encoder = StreamingLatentEncoder(
            vae, dev, image_size=tuple(cfg.vae.get("encode_image_size", [128, 128])))
        decode_fn = None  # streaming has no stats.pt; the multi run logs no video anyway
    else:
        decode_fn = build_decode_fn(cfg) if latent else None
    FlowTrainer(cfg, model, train_loader, val_loader, decode_fn=decode_fn,
                stream_encoder=stream_encoder).fit()


if __name__ == "__main__":
    # decord's thread pool deadlocks under DataLoader's default 'fork' start
    # method (num_workers>0), so use 'spawn' -- workers start fresh, no inherited
    # locks/threads. Harmless when num_workers=0.
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

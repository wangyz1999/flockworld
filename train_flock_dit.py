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
        grad_checkpointing=bool(cfg.train.get("grad_checkpointing", False)),
        agent_embed_per_layer=bool(m.get("agent_embed_per_layer", False)),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(OmegaConf.to_yaml(cfg))
    latent = bool(cfg.data.get("latent", False))
    print(f"FlockDiT params: {n_params/1e6:.1f}M | num_agents={num_agents} | latent={latent}")

    decode_fn = build_decode_fn(cfg) if latent else None
    FlowTrainer(cfg, model, train_loader, val_loader, decode_fn=decode_fn).fit()


def build_decode_fn(cfg):
    """Frozen-VAE decoder for latent-mode rollout-video logging.

    Maps a normalized latent clip (F, z, h, w) -> pixels (T_pix, 3, H, W) in [-1,1]:
    denormalize with the cached per-channel stats, then VAE-decode.
    """
    import torch
    from pathlib import Path
    from modeling.models.frozen_vae import FrozenVAE

    device = "cuda" if torch.cuda.is_available() and str(cfg.device) != "cpu" else "cpu"
    vae = FrozenVAE(str(cfg.vae.checkpoint_path), device=device)
    stats = torch.load(Path(cfg.data.root) / cfg.data.get("latent_cache_dir", "latent_cache") / "stats.pt",
                       map_location=device)
    # clip is (F, z, h, w) -> broadcast stats over the channel dim (dim 1)
    mean = stats["mean"].view(1, -1, 1, 1).to(device)
    std = stats["std"].view(1, -1, 1, 1).to(device)

    @torch.no_grad()
    def decode(latent_clip):  # (F, z, h, w) normalized -> (T_pix, 3, H, W)
        z = latent_clip.to(device) * std + mean            # denormalize
        pixels = vae.decode(z.permute(1, 0, 2, 3).unsqueeze(0))  # (1,3,T_pix,H,W)
        return pixels[0].permute(1, 0, 2, 3)               # (T_pix, 3, H, W)

    return decode


if __name__ == "__main__":
    # decord's thread pool deadlocks under DataLoader's default 'fork' start
    # method (num_workers>0), so use 'spawn' -- workers start fresh, no inherited
    # locks/threads. Harmless when num_workers=0.
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

"""Decode normalized cached latents with the frozen video autoencoder."""

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

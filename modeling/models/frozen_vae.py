"""Load the trained WanVAE checkpoint as a frozen encoder/decoder.

Wraps :class:`WanVAELightning` (which carries the proper chunked causal
`_encode`/`_decode` and the training `cfg` in its checkpoint hyper_parameters) so
the world-model pipeline can turn pixel clips into latents and back without
re-deriving any of the VAE plumbing. The VAE is frozen (eval, no grad).

Geometry (this checkpoint): RGB `(B,3,T,128,128)` in [-1,1] ->
latent `(B, 8, T_lat, 8, 8)`, where `T_lat = 1 + (T-1)//4` (4x temporal,
16x spatial). Uses the encoder mean `mu` as the latent (deterministic, cache-able).
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from modeling.models.vae_module import WanVAELightning


class FrozenVAE:
    def __init__(self, checkpoint_path: str, device: str | torch.device = "cpu"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "hyper_parameters" not in ckpt or "cfg" not in ckpt["hyper_parameters"]:
            raise ValueError(f"{checkpoint_path} lacks hyper_parameters.cfg (not a WanVAELightning ckpt?)")
        cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])
        cfg.model.pretrained_path = None  # don't let __init__ try to re-load weights
        module = WanVAELightning(cfg)
        missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[FrozenVAE] load: missing={len(missing)} unexpected={len(unexpected)}")
        module.eval().to(device)
        for p in module.parameters():
            p.requires_grad_(False)
        self.module = module
        self.device = device
        self.z_dim = module.z_dim
        self.patch_size = module.patch_size

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B,3,T,H,W) in [-1,1] -> latent mean mu (B, z_dim, T_lat, h, w)."""
        mu, _ = self.module._encode(pixels.to(self.device))
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, z_dim, T_lat, h, w) -> pixels (B,3,T,H,W) in [-1,1]."""
        return self.module._decode(z.to(self.device))

    @staticmethod
    def latent_frames(num_pixel_frames: int) -> int:
        """T_lat for a given pixel clip length (causal 1+4k -> 1+k)."""
        return 1 + (num_pixel_frames - 1) // 4

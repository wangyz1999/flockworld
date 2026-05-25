from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from modeling.models.wan_vae import WanVAE_, patchify, unpatchify


class WanVAELightning(L.LightningModule):
    """Lightning wrapper around the Wan VAE for from-scratch / fine-tune training.

    Bypasses the inference-only ``WanVAE_.encode``/``.decode`` (which apply a
    pretrained latent scale and discard ``log_var``) and instead runs the
    encoder/decoder in non-chunked mode so we can use both ``mu`` and
    ``log_var`` for the KL term.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})

        self.patch_size = int(cfg.model.patch_size)
        self.z_dim = int(cfg.model.z_dim)
        self.kl_weight = float(cfg.loss.kl_weight)
        self.recon_loss_type = str(cfg.loss.recon_loss).lower()
        self.logvar_clamp = tuple(cfg.loss.logvar_clamp)

        self.vae = WanVAE_(
            dim=int(cfg.model.dim),
            dec_dim=int(cfg.model.dec_dim),
            z_dim=self.z_dim,
            dim_mult=list(cfg.model.dim_mult),
            num_res_blocks=int(cfg.model.num_res_blocks),
            attn_scales=list(cfg.model.attn_scales),
            temperal_downsample=list(cfg.model.temperal_downsample),
            dropout=float(cfg.model.dropout),
        )

        if cfg.model.pretrained_path:
            state = torch.load(cfg.model.pretrained_path, map_location="cpu")
            missing, unexpected = self.vae.load_state_dict(state, strict=False)
            print(f"[WanVAELightning] loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors WanVAE_.encode chunking (1 + 4k frames) but returns log_var
        # too and skips the inference latent-scale shift.
        self.vae.clear_cache()
        x = patchify(x, patch_size=self.patch_size)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None
        for i in range(iter_):
            self.vae._enc_conv_idx = [0]
            if i == 0:
                chunk = x[:, :, :1, :, :]
            else:
                chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            piece = self.vae.encoder(
                chunk,
                feat_cache=self.vae._enc_feat_map,
                feat_idx=self.vae._enc_conv_idx,
            )
            out = piece if out is None else torch.cat([out, piece], dim=2)
        mu, log_var = self.vae.conv1(out).chunk(2, dim=1)
        log_var = log_var.clamp(*self.logvar_clamp)
        self.vae.clear_cache()
        return mu, log_var

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        self.vae.clear_cache()
        x = self.vae.conv2(z)
        iter_ = x.shape[2]
        out = None
        for i in range(iter_):
            self.vae._conv_idx = [0]
            piece = self.vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self.vae._feat_map,
                feat_idx=self.vae._conv_idx,
                first_chunk=(i == 0),
            )
            out = piece if out is None else torch.cat([out, piece], dim=2)
        out = unpatchify(out, patch_size=self.patch_size)
        self.vae.clear_cache()
        return out

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        recon = self._decode(z)
        return {"recon": recon, "mu": mu, "log_var": log_var, "z": z}

    def _compute_loss(self, x: torch.Tensor, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        recon, mu, log_var = out["recon"], out["mu"], out["log_var"]
        if self.recon_loss_type == "l1":
            recon_loss = F.l1_loss(recon, x)
        else:
            recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl.mean()
        loss = recon_loss + self.kl_weight * kl_loss
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def training_step(self, batch, batch_idx):
        x = batch["video"]
        out = self(x)
        losses = self._compute_loss(x, out)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        out = self(x)
        losses = self._compute_loss(x, out)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return losses["loss"]

    def configure_optimizers(self):
        cfg = self.cfg.optim
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(cfg.lr),
            betas=tuple(cfg.betas),
            weight_decay=float(cfg.weight_decay),
        )
        if cfg.get("scheduler", None) == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(cfg.t_max), eta_min=float(cfg.eta_min)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

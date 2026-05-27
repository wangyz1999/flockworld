"""Entry point for training the Wan VAE on partial-agent observation videos.

Run from the repository root::

    uv run train_vae.py
    uv run train_vae.py trainer.max_epochs=1 dataloader.batch_size=1

Defaults are loaded from ``config/train_vae.yaml``. Overrides use OmegaConf
dotlist syntax. Access nested keys with ``cfg.var1.var2``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from modeling.data.vae_datamodule import PartialVideoVAEDataModule
from modeling.models.vae_module import WanVAELightning
from modeling.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Wan VAE on partial-agent observation videos.")
    parser.add_argument("--config", default=None, help="Path to YAML config; defaults to config/train_vae.yaml.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. trainer.max_epochs=1.")
    return parser.parse_args()


def load_cfg(config_path: str | None, overrides: list[str]):
    path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config" / "train_vae.yaml"
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def build_run_dir(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{stamp}_{cfg.experiment_name}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    return run_dir


def build_loggers(cfg, run_dir: Path):
    loggers = [CSVLogger(save_dir=str(run_dir), name="csv")]
    if bool(cfg.logger.use_wandb):
        loggers.append(
            WandbLogger(
                project=str(cfg.logger.wandb_project),
                entity=cfg.logger.wandb_entity,
                name=run_dir.name,
                save_dir=str(run_dir),
            )
        )
    return loggers


class WandbVideoLoggingCallback(Callback):
    """Every ``every_n_steps`` global steps, encode one train clip and one val
    clip through the VAE, write the GT|recon side-by-side mp4 to ``save_dir``,
    and upload that local file to wandb."""

    def __init__(
        self,
        every_n_steps: int,
        fps: int,
        save_dir: Path,
        upload_to_wandb: bool = False,
        codec: str = "avc1",
    ):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.fps = int(fps)
        self.upload_to_wandb = bool(upload_to_wandb)
        self.codec = str(codec)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._val_iter = None

    def _wandb_run(self, trainer):
        for lg in (trainer.loggers or []):
            if isinstance(lg, WandbLogger):
                return lg.experiment
        return None

    def _next_val_batch(self, trainer):
        dm = trainer.datamodule
        if dm is None or dm.val_set is None:
            return None
        if self._val_iter is None:
            self._val_iter = iter(dm.val_dataloader())
        try:
            return next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(dm.val_dataloader())
            return next(self._val_iter)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0:
            return
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return

        was_training = pl_module.training
        pl_module.eval()
        train_path = self._encode_and_save(pl_module, batch["video"][:1], "train", step)
        val_batch = self._next_val_batch(trainer)
        val_path = (
            self._encode_and_save(pl_module, val_batch["video"][:1], "val", step)
            if val_batch is not None
            else None
        )
        if was_training:
            pl_module.train()

        if not self.upload_to_wandb:
            return
        run = self._wandb_run(trainer)
        if run is None:
            return
        import wandb
        log = {}
        if train_path is not None:
            log["train/recon_video"] = wandb.Video(str(train_path), format="mp4")
        if val_path is not None:
            log["val/recon_video"] = wandb.Video(str(val_path), format="mp4")
        if log:
            run.log(log, step=step)

    @torch.no_grad()
    def _encode_and_save(self, pl_module, video: torch.Tensor, tag: str, step: int) -> Path | None:
        x = video.to(pl_module.device)
        out = pl_module(x)
        recon = out["recon"].clamp(-1.0, 1.0)
        gt = ((x + 1.0) * 127.5).to(torch.uint8).cpu()
        rc = ((recon + 1.0) * 127.5).to(torch.uint8).cpu()
        pair = torch.cat([gt, rc], dim=-1)                       # (B, C, T, H, 2W)
        # B=1 -> squeeze; produce (T, H, 2W, C) for cv2.
        frames = pair[0].permute(1, 2, 3, 0).contiguous().numpy()  # (T, H, 2W, C) uint8 RGB
        path = self.save_dir / f"step_{step:08d}_{tag}.mp4"
        if not _write_mp4_cv2(frames, path, fps=self.fps, codec=self.codec):
            return None
        return path


def _write_mp4_cv2(frames_rgb, path: Path, fps: int, codec: str = "avc1") -> bool:
    """Write (T, H, W, 3) uint8 RGB array to mp4 via OpenCV. Tries the requested
    codec first (default ``avc1`` / H.264 for browser playback) and falls back
    to ``mp4v`` if the codec is unavailable in this OpenCV build."""
    try:
        import cv2
    except ImportError:
        print(f"[video] cv2 unavailable; skipping {path}")
        return False
    T, H, W, _ = frames_rgb.shape
    for code in (codec, "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
        if writer.isOpened():
            for t in range(T):
                writer.write(cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR))
            writer.release()
            return True
        print(f"[video] codec {code!r} unavailable, trying fallback")
    print(f"[video] cv2 failed to open any writer for {path}")
    return False


def build_callbacks(cfg, run_dir: Path):
    callbacks = [
        ModelSummary(max_depth=int(cfg.trainer.get("summary_depth", -1))),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            monitor=str(cfg.checkpoint.monitor),
            mode=str(cfg.checkpoint.mode),
            save_top_k=int(cfg.checkpoint.save_top_k),
            save_last=bool(cfg.checkpoint.save_last),
            every_n_epochs=int(cfg.checkpoint.every_n_epochs),
            filename="vae-{epoch:03d}-{val/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    every_n = int(cfg.logger.get("log_video_every_n_steps", 0))
    if every_n > 0:
        callbacks.append(
            WandbVideoLoggingCallback(
                every_n_steps=every_n,
                fps=int(cfg.logger.get("video_fps", 8)),
                save_dir=run_dir / "videos",
                upload_to_wandb=bool(cfg.logger.get("log_video_to_wandb", True)),
                codec=str(cfg.logger.get("video_codec", "avc1")),
            )
        )
    return callbacks


def _print_torchinfo(cfg, model) -> None:
    try:
        from torchinfo import summary
    except ImportError:
        print("[torchinfo] not installed — `uv add torchinfo` to get per-layer output shapes.")
        return
    H, W = cfg.data.image_size
    input_size = (1, 3, int(cfg.data.num_frames), int(H), int(W))
    print(summary(
        model,
        input_size=input_size,
        depth=int(cfg.trainer.get("torchinfo_depth", 4)),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        row_settings=("var_names",),
        verbose=0,
    ))


def main():
    args = parse_args()
    cfg = load_cfg(args.config, args.overrides)
    seed_everything(int(cfg.seed))
    torch.set_float32_matmul_precision("high")

    print(OmegaConf.to_yaml(cfg))

    datamodule = PartialVideoVAEDataModule(cfg)
    model = WanVAELightning(cfg)

    if bool(cfg.trainer.get("print_torchinfo", True)):
        _print_torchinfo(cfg, model)

    run_dir = build_run_dir(cfg)
    print(f"[train_vae] run dir: {run_dir}")
    loggers = build_loggers(cfg, run_dir)
    callbacks = build_callbacks(cfg, run_dir)

    trainer = L.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        deterministic=bool(cfg.trainer.deterministic),
        default_root_dir=str(run_dir),
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

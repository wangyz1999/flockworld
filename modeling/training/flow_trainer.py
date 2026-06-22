"""Flow-matching trainer for FlockDiT (Diffusion-Forcing objective).

Mirrors :class:`modeling.training.trainer.Trainer` (AdamW, AMP/GradScaler, grad
clipping, tqdm, checkpoint dict) but replaces the L1 step with the rectified-flow
step: sample per-frame timesteps (context frames clean), noise the future
frames, predict velocity, and take MSE over future frames only.

Optional Weights & Biases logging (``cfg.logger``): logs the full resolved config
(training parameters), per-step and per-epoch train/val loss, and -- for
single-agent runs -- a periodic rollout video (top = ground truth, bottom =
prediction) for visual inspection. Uses a project distinct from the VAE runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from modeling import flow_matching as fm


class FlowTrainer:
    def __init__(self, cfg, model: nn.Module, train_loader, val_loader=None):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
        )
        self.scaler = GradScaler("cuda", enabled=bool(cfg.train.amp) and self.device.type == "cuda")
        self.output_dir = Path(cfg.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.wandb = self._init_wandb()
        self.vis_sample = self._make_vis_sample()

    # -- wandb setup ------------------------------------------------------- #
    def _init_wandb(self):
        lg = self.cfg.get("logger", None)
        if lg is None or not bool(lg.get("use_wandb", False)):
            return None
        import wandb

        wandb.init(
            project=str(lg.get("wandb_project", "flock-world-model")),
            entity=lg.get("wandb_entity", None),
            name=str(self.cfg.get("experiment_name", "flock_dit")),
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        wandb.summary["num_params_M"] = sum(p.numel() for p in self.model.parameters()) / 1e6
        return wandb

    def _make_vis_sample(self):
        """A fixed single-agent val clip used for the periodic rollout video."""
        if self.wandb is None or self.val_loader is None:
            return None
        try:
            sample = self.val_loader.dataset[0]
        except Exception:
            return None
        if sample["frames"].dim() != 4:  # single-agent only: (F, C, H, W)
            return None
        return sample

    def fit(self):
        try:
            for epoch in range(1, int(self.cfg.train.epochs) + 1):
                train_loss = self._run_epoch(epoch)
                val_loss = self.evaluate() if self.val_loader is not None else None
                if epoch % int(self.cfg.train.save_every) == 0:
                    self.save_checkpoint(epoch, train_loss, val_loss)
                msg = f"epoch={epoch} train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.6f}"
                print(msg)
                self._log_epoch(epoch, train_loss, val_loss)
        finally:
            if self.wandb is not None:
                self.wandb.finish()

    # -- core flow-matching step ------------------------------------------ #
    def _loss(self, batch) -> torch.Tensor:
        frames = batch["frames"]  # (B,F,C,H,W) or (B,P,F,C,H,W)
        actions = batch["actions"]
        ctx = int(batch["context_len"][0])
        multi = frames.dim() == 6
        frame_axis = 2 if multi else 1
        leading = tuple(frames.shape[:frame_axis])  # (B,) or (B,P)
        num_frames = frames.shape[frame_axis]

        t = fm.sample_timesteps(leading, num_frames, ctx, frames.device)
        x_t, eps = fm.add_noise(frames, t)
        v_pred = self.model(x_t, t, actions)
        target = fm.velocity_target(eps, frames)

        future = slice(ctx, None)
        if multi:
            err = (v_pred[:, :, future] - target[:, :, future]) ** 2
        else:
            err = (v_pred[:, future] - target[:, future]) ** 2
        return err.mean()

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        iterator = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                loss = self._loss(batch)
            self.scaler.scale(loss).backward()
            if float(self.cfg.train.grad_clip_norm) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            losses.append(float(loss.detach().cpu()))
            if step % int(self.cfg.train.log_every) == 0:
                window = losses[-int(self.cfg.train.log_every):]
                avg = sum(window) / len(window)
                iterator.set_postfix(loss=avg)
                if self.wandb is not None:
                    self.wandb.log(
                        {"train/loss_step": losses[-1], "train/loss_avg": avg, "epoch": epoch},
                        step=self.global_step,
                    )
        return sum(losses) / max(1, len(losses))

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses: list[float] = []
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            batch = self._to_device(batch)
            losses.append(float(self._loss(batch).cpu()))
        return sum(losses) / max(1, len(losses))

    # -- wandb logging ----------------------------------------------------- #
    def _log_epoch(self, epoch, train_loss, val_loss):
        if self.wandb is None:
            return
        log = {"train/loss": train_loss, "epoch": epoch}
        if val_loss is not None:
            log["val/loss"] = val_loss
        self.wandb.log(log, step=self.global_step)
        every = int(self.cfg.logger.get("log_sample_every", 0))
        if self.vis_sample is not None and every > 0 and epoch % every == 0:
            self._log_rollout_video(epoch)

    @torch.no_grad()
    def _log_rollout_video(self, epoch):
        # Logging must never crash training, so guard the whole thing.
        try:
            s = self.vis_sample
            frames = s["frames"].unsqueeze(0).to(self.device)  # (1,F,C,H,W)
            actions = s["actions"].unsqueeze(0).to(self.device)
            ctx = int(s["context_len"])
            f = frames.shape[1]
            steps = int(self.cfg.logger.get("sample_steps", 50))
            clip = fm.euler_rollout(self.model, frames[:, :ctx], actions, f - ctx, num_steps=steps)
            # stack GT (top) over prediction (bottom) -> (T, 2H, W, C) uint8 RGB
            vid = torch.cat([frames[0], clip[0]], dim=2).clamp(-1, 1)
            vid = ((vid + 1) / 2 * 255).round().to(torch.uint8)
            vid = vid.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (T, 2H, W, C) RGB
            path = self._write_mp4(vid, epoch, fps=int(self.cfg.logger.get("video_fps", 8)))
            if path is not None:
                # Pass a file path (not a raw array) so wandb doesn't need moviepy.
                self.wandb.log({"val/rollout": self.wandb.Video(str(path), format="mp4")}, step=self.global_step)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] rollout video logging failed at epoch {epoch}: {e}")

    def _write_mp4(self, frames_thwc, epoch, fps: int):
        """Write RGB frames (T,H,W,C uint8) to a browser-playable H.264 mp4.

        Pipes raw frames to the system ffmpeg with libx264 (software encoder) --
        same approach as the VAE pipeline. The PyPI opencv wheel lacks libx264,
        and this machine's cv2 routes the 'avc1' fourcc to a missing
        h264_v4l2m2m hardware encoder, so cv2.VideoWriter is unreliable here.
        """
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("[warn] ffmpeg not on PATH; skipping rollout video")
            return None
        vdir = self.output_dir / "videos"
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / f"rollout_epoch_{epoch:04d}.mp4"
        _, h, w, _ = frames_thwc.shape
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for fr in frames_thwc:
                proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=30)
        if proc.returncode != 0:
            print(f"[warn] ffmpeg failed (code {proc.returncode}) for {path}")
            return None
        return path

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float | None):
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": OmegaConf.to_container(self.cfg.model, resolve=True),
            },
            path,
        )

    def _to_device(self, batch: dict):
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

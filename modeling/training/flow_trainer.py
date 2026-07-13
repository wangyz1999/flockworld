"""Flow-matching trainer for FlockDiT (Diffusion-Forcing objective).

Mirrors :class:`modeling.training.trainer.Trainer` (AdamW, AMP/GradScaler, grad
clipping, tqdm, checkpoint dict) but replaces the L1 step with the rectified-flow
step: sample per-frame timesteps (context frames clean), noise the future
frames, predict velocity, and take MSE over future frames only.
``train.diffusion_forcing`` switches to full diffusion forcing: every frame is
noised with its own timestep (no clean context) and the MSE covers all frames.
``train.warm_start`` initializes the model weights from another run's checkpoint
(two-stage training: single-agent pretrain -> multi-agent fine-tune).

Optional Weights & Biases logging (``cfg.logger``): logs the full resolved config
(training parameters), per-step and per-epoch train/val loss, and -- for
single-agent runs -- a periodic rollout video (top = ground truth, bottom =
prediction) for visual inspection. Uses a project distinct from the VAE runs.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from modeling import flow_matching as fm

# Two-stage training (exp 4): parameters that may be MISSING from a warm-start
# checkpoint and keep their fresh random init -- everything that only exists in the
# multi-agent model, so a single-agent (num_agents=1) pretrain can seed a multi-agent
# fine-tune. The shared backbone must match exactly; anything else missing is an error.
_WARMSTART_EXEMPT = ("agent_embed.", "agent_action_embed", "action_combine.")


class FlowTrainer:
    def __init__(self, cfg, model: nn.Module, train_loader, val_loader=None, decode_fn=None,
                 stream_encoder=None):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.model = model.to(self.device)
        warm_start = cfg.train.get("warm_start", None)
        if warm_start:
            self._load_warm_start(str(warm_start))
        self.train_loader = train_loader
        self.val_loader = val_loader
        # latent mode: maps a (F, C, h, w) latent clip -> (T_pix, 3, H, W) pixels for video logging
        self.decode_fn = decode_fn
        # streaming mode: turns streamed pixel batches into normalized latent batches on the GPU
        self.stream_encoder = stream_encoder
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

    # -- two-stage warm start (exp 4) --------------------------------------- #
    def _load_warm_start(self, path_str: str):
        """Initialize model weights from a previous run's checkpoint (``train.warm_start``).

        A warm START, not a resume: optimizer state, epoch and global_step all begin
        fresh -- only the weights carry over. The path may be a checkpoint file or a
        ``checkpoints/`` directory (resolved to the furthest-trained checkpoint), so an
        unattended two-stage pipeline needn't predict where a wall-clock kill lands.
        Keys in ``_WARMSTART_EXEMPT`` may be absent (single-agent -> multi-agent) and
        keep their random init; any other mismatch is an error, not a silent skip.
        """
        path = Path(path_str)
        if path.is_dir():
            path = self._latest_checkpoint(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = ckpt["model"]
        own = self.model.state_dict()

        unexpected = [k for k in state if k not in own]
        if unexpected:
            raise ValueError(f"warm_start {path}: checkpoint has keys the model lacks: {unexpected}")
        mismatched = [k for k in state if state[k].shape != own[k].shape]
        if mismatched:
            detail = {k: f"{tuple(state[k].shape)} vs model {tuple(own[k].shape)}" for k in mismatched}
            raise ValueError(f"warm_start {path}: shape mismatch: {detail}")
        missing = [k for k in own if k not in state]
        not_exempt = [k for k in missing if not k.startswith(_WARMSTART_EXEMPT)]
        if not_exempt:
            raise ValueError(f"warm_start {path}: checkpoint is missing non-exempt keys: {not_exempt}")

        self.model.load_state_dict(state, strict=False)
        kept = f", kept at random init: {missing}" if missing else ""
        print(f"[warm start] loaded {len(state)} tensors from {path} "
              f"(epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}){kept}")

    @staticmethod
    def _latest_checkpoint(ckpt_dir: Path) -> Path:
        """Furthest-trained checkpoint in a run's directory.

        Zero-padded names sort lexicographically == numerically, so the newest of each
        family is last; when both families exist, ``global_step`` (mmap = cheap scan)
        decides between the newest step_*.pt and the newest epoch_*.pt.
        """
        candidates = [
            files[-1]
            for files in (sorted(ckpt_dir.glob("step_*.pt")), sorted(ckpt_dir.glob("epoch_*.pt")))
            if files
        ]
        if not candidates:
            raise FileNotFoundError(f"warm_start: no step_*.pt or epoch_*.pt in {ckpt_dir}")
        return max(
            candidates,
            key=lambda p: int(torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                              .get("global_step", 0)),
        )

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

    def _fit_stream_stats(self):
        """Streaming only: measure per-channel latent mean/std once from a few fresh batches."""
        if self.stream_encoder is None or self.stream_encoder.mean is not None:
            return
        n = int(self.cfg.data.streaming.get("stats_batches", 4))
        it = iter(self.train_loader)
        pix = []
        for _ in range(n):
            try:
                pix.append(next(it)["pixels"])
            except StopIteration:
                break
        self.stream_encoder.fit_stats(pix)
        print(f"[streaming] latent stats from {len(pix)} batches: "
              f"mean~{float(self.stream_encoder.mean.mean()):.3f} std~{float(self.stream_encoder.std.mean()):.3f}")

    def fit(self):
        self._fit_stream_stats()
        n_epochs = int(self.cfg.train.epochs)
        # train.epochs=-1: no epoch limit -- run until killed. For wall-clock-budgeted
        # runs (`timeout 48h ...`) the step checkpoints make the kill lossless.
        epochs_iter = itertools.count(1) if n_epochs < 0 else range(1, n_epochs + 1)
        try:
            for epoch in epochs_iter:
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
        if self.stream_encoder is not None:                 # streaming: pixels -> normalized latents
            batch = self.stream_encoder.encode_batch(batch)
        frames = batch["frames"]  # (B,F,C,H,W) or (B,P,F,C,H,W)
        actions = batch["actions"]
        ctx = int(batch["context_len"][0])
        multi = frames.dim() == 6
        frame_axis = 2 if multi else 1
        leading = tuple(frames.shape[:frame_axis])  # (B,) or (B,P)
        num_frames = frames.shape[frame_axis]

        # Full diffusion forcing (train.diffusion_forcing): noise ALL frames with
        # independent timesteps (no clean-context pinning) and take the loss on every
        # frame, so the model trains on the imperfect context it rolls out on. The
        # default keeps the clean-context variant: context pinned to t=0, loss on
        # future frames only.
        df = bool(self.cfg.train.get("diffusion_forcing", False))
        pin = 0 if df else ctx
        if multi and bool(self.cfg.train.get("shared_timesteps", False)):
            # Tiled-view mode (MIRA): the P views of a timestep are denoised as one
            # frame, so they share a single timestep; expand keeps the (B,P,F) layout.
            t = fm.sample_timesteps(leading[:1], num_frames, pin, frames.device)
            t = t[:, None, :].expand(*leading, num_frames)
        else:
            t = fm.sample_timesteps(leading, num_frames, pin, frames.device)
        x_t, eps = fm.add_noise(frames, t)
        v_pred = self.model(x_t, t, actions)
        target = fm.velocity_target(eps, frames)

        if df:
            err = (v_pred - target) ** 2
        else:
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
            every_steps = int(self.cfg.train.get("save_every_steps", 0))
            if every_steps > 0 and self.global_step % every_steps == 0:
                self._save_step_checkpoint(epoch, losses[-every_steps:])
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
            gt, pred = frames[0], clip[0]
            if self.decode_fn is not None:  # latent mode: decode latents -> pixels first
                gt, pred = self.decode_fn(gt), self.decode_fn(pred)
            # stack GT (top) over prediction (bottom) -> (T, 2H, W, C) uint8 RGB
            vid = torch.cat([gt, pred], dim=2).clamp(-1, 1)
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
        torch.save(self._checkpoint_dict(epoch, train_loss, val_loss), path)

    def _save_step_checkpoint(self, epoch: int, recent_losses: list[float]):
        """Mid-epoch checkpoint every ``train.save_every_steps`` optimizer steps.

        Lets long streaming runs be evaluated mid-run (point an eval script at a
        ``step_*.pt``) and bound the work lost to a crash or a manual stop to one
        save interval. ``train.keep_step_checkpoints > 0`` keeps only the newest N
        step checkpoints (epoch checkpoints are never rotated); the default (0)
        keeps them all. ``val_loss`` is None -- no val pass runs mid-epoch -- so
        eval_flock_dit's best-checkpoint scan (epoch_*.pt only) is unaffected.
        """
        train_loss = sum(recent_losses) / max(1, len(recent_losses))
        path = self.checkpoint_dir / f"step_{self.global_step:08d}.pt"
        torch.save(self._checkpoint_dict(epoch, train_loss, None), path)
        keep = int(self.cfg.train.get("keep_step_checkpoints", 0))
        if keep > 0:
            # Zero-padded names sort lexicographically == numerically.
            for old in sorted(self.checkpoint_dir.glob("step_*.pt"))[:-keep]:
                old.unlink()

    def _checkpoint_dict(self, epoch: int, train_loss: float, val_loss: float | None) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": OmegaConf.to_container(self.cfg.model, resolve=True),
        }

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

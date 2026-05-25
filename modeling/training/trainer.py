from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer:
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

    def fit(self):
        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss = self._run_epoch(epoch)
            val_loss = self.evaluate() if self.val_loader is not None else None
            if epoch % int(self.cfg.train.save_every) == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
            if val_loss is None:
                print(f"epoch={epoch} train_loss={train_loss:.6f}")
            else:
                print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        iterator = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                pred = self.model(batch["context_video"], batch["partial_video"], batch["actions"])
                loss = torch.nn.functional.l1_loss(pred, batch["target_video"])

            self.scaler.scale(loss).backward()
            if float(self.cfg.train.grad_clip_norm) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.cfg.train.grad_clip_norm),
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(float(loss.detach().cpu()))
            if step % int(self.cfg.train.log_every) == 0:
                iterator.set_postfix(loss=sum(losses[-int(self.cfg.train.log_every) :]) / len(losses[-int(self.cfg.train.log_every) :]))
        return sum(losses) / max(1, len(losses))

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses: list[float] = []
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            batch = self._to_device(batch)
            pred = self.model(batch["context_video"], batch["partial_video"], batch["actions"])
            loss = torch.nn.functional.l1_loss(pred, batch["target_video"])
            losses.append(float(loss.cpu()))
        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float | None):
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            path,
        )

    def _to_device(self, batch: dict):
        return {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

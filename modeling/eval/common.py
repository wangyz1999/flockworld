"""Shared checkpoint loading, model construction, and rollout video output."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from modeling.models.flock_dit import FlockDiT


def find_best_checkpoint(ckpt_dir: Path) -> tuple[Path, float]:
    """Pick the checkpoint with the lowest stored val_loss (mmap = cheap scan).

    Falls back to the newest ``step_*.pt`` (zero-padded global_step, so sorted
    name order == numeric order) when no ``epoch_*.pt`` has a stored val_loss --
    e.g. a run packaged/transferred before its next epoch boundary. The returned
    val_loss is ``nan`` in that case (no val pass has run for a step checkpoint);
    callers that print/compare it should treat nan as "unscored", not "best".
    """
    best, best_val = None, float("inf")
    for p in sorted(ckpt_dir.glob("epoch_*.pt")):
        meta = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        v = meta.get("val_loss")
        if v is not None and float(v) < best_val:
            best, best_val = p, float(v)
    if best is not None:
        return best, best_val
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    if steps:
        return steps[-1], float("nan")
    raise FileNotFoundError(f"No checkpoints with a val_loss found in {ckpt_dir}")


def build_model(cfg) -> FlockDiT:
    m = cfg.model
    return FlockDiT(
        in_channels=int(m.in_channels),
        out_channels=int(m.out_channels),
        dim=int(m.dim),
        depth=int(m.depth),
        heads=int(m.heads),
        ffn_dim=int(m.ffn_dim),
        patch=int(m.patch),
        patch_t=int(m.get("patch_t", 1)),
        action_dim=len(list(cfg.data.action_features)),
        num_agents=int(cfg.data.get("num_agents", 1)),
        local_attn_size=int(m.get("local_attn_size", -1)),
        # Tiled-view experiment flags (MIRA Sec 4.5); defaults preserve the baseline.
        tiled_rope=bool(m.get("tiled_rope", False)),
        tile_grid=tuple(m.tile_grid) if m.get("tile_grid", None) is not None else None,
        broadcast_actions=bool(m.get("broadcast_actions", False)),
        use_agent_embed=bool(m.get("use_agent_embed", True)),
        agent_embed_per_layer=bool(m.get("agent_embed_per_layer", False)),
    )


def write_mp4(frames_thwc: np.ndarray, path: Path, fps: int) -> Path | None:
    """RGB (T,H,W,C) uint8 -> H.264 mp4 via ffmpeg/libx264 (browser-playable)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[warn] ffmpeg not on PATH; skipping", path)
        return None
    _, h, w, _ = frames_thwc.shape
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames_thwc:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    proc.wait(timeout=60)
    return path if proc.returncode == 0 else None

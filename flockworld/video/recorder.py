"""Video recording utilities for full and partial observability."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def crop_partial(frame_rgb: np.ndarray, pos: np.ndarray, size: int) -> np.ndarray:
    """Square ``size`` crop of ``frame_rgb`` centred on ``pos``, zero-padded
    where the window falls outside the frame."""
    h, w = frame_rgb.shape[:2]
    cx = int(round(float(pos[0])))
    cy = int(round(float(pos[1])))
    half = size // 2
    x0 = cx - half
    y0 = cy - half
    crop = np.zeros((size, size, frame_rgb.shape[2]), dtype=frame_rgb.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x0 + size)
    src_y1 = min(h, y0 + size)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        crop[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = \
            frame_rgb[src_y0:src_y1, src_x0:src_x1]
    return crop


class _FFmpegH264Writer:
    """Writes raw RGB frames to an H.264 mp4 via a piped ffmpeg subprocess.

    H.264 (libx264) yields ~5–10× smaller files than mp4v and plays natively
    in browsers, whereas the libopencv wheel from PyPI ships without a
    libx264 encoder."""

    def __init__(self, path: Path, fps: int, width: int, height: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; required for H.264 recording")
        self._width = width
        self._height = height
        self._path = path
        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_rgb(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape[:2] != (self._height, self._width):
            raise ValueError(
                f"frame shape {frame_rgb.shape[:2]} does not match writer "
                f"({self._height}, {self._width}) for {self._path}"
            )
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        self._proc.stdin.write(frame_rgb.tobytes())

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
        finally:
            self._proc = None


class VideoRecorder:
    """Write full-observation and/or partial-observation videos simultaneously.

    Partial observation is a square crop centred on the controlled agent
    (index 0).
    """

    def __init__(
        self,
        full_obs_path: str | Path | None = None,
        partial_obs_path: str | Path | None = None,
        partial_obs_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        partial_obs_size: int = 128,
        fps: int = 30,
        canvas_w: int = 800,
        canvas_h: int = 800,
    ):
        self.partial_obs_size = partial_obs_size
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self._full_writer: _FFmpegH264Writer | None = None
        self._partial_writers: list[_FFmpegH264Writer] = []

        if full_obs_path is not None:
            p = Path(full_obs_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._full_writer = _FFmpegH264Writer(p, fps, canvas_w, canvas_h)

        partial_paths = []
        if partial_obs_path is not None:
            partial_paths.append(partial_obs_path)
        if partial_obs_paths is not None:
            partial_paths.extend(partial_obs_paths)

        for partial_path in partial_paths:
            p = Path(partial_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._partial_writers.append(
                _FFmpegH264Writer(p, fps, partial_obs_size, partial_obs_size)
            )

        self._partial_writer = self._partial_writers[0] if self._partial_writers else None

    # ── public API ───────────────────────────────────────────────────

    def record(self, frame_rgb: np.ndarray, controlled_pos: np.ndarray):
        """Write one frame.

        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8 RGB image.
        controlled_pos : (2,) or (K, 2) pixel positions for partial crops.
        """
        if self._full_writer is not None:
            self._full_writer.write_rgb(frame_rgb)

        if self._partial_writers:
            positions = self._normalize_positions(controlled_pos)
            for writer, pos in zip(self._partial_writers, positions):
                crop = self._crop_partial(frame_rgb, pos)
                writer.write_rgb(crop)

    def close(self):
        if self._full_writer is not None:
            self._full_writer.close()
            self._full_writer = None
        for writer in self._partial_writers:
            writer.close()
        self._partial_writers = []
        self._partial_writer = None

    # ── internals ────────────────────────────────────────────────────

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions)
        if positions.shape == (2,):
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"Expected partial-observation positions with shape (K, 2), got {positions.shape}.")
        if positions.shape[0] < len(self._partial_writers):
            raise ValueError(
                f"Got {positions.shape[0]} partial-observation positions for "
                f"{len(self._partial_writers)} writers."
            )
        return positions

    def _crop_partial(self, frame_rgb: np.ndarray, pos: np.ndarray) -> np.ndarray:
        return crop_partial(frame_rgb, pos, self.partial_obs_size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Video recording utilities for full and partial observability."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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
        self._full_writer = None
        self._partial_writer = None
        self._partial_writers = []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if full_obs_path is not None:
            p = Path(full_obs_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._full_writer = cv2.VideoWriter(
                str(p), fourcc, fps, (canvas_w, canvas_h),
            )

        partial_paths = []
        if partial_obs_path is not None:
            partial_paths.append(partial_obs_path)
        if partial_obs_paths is not None:
            partial_paths.extend(partial_obs_paths)

        for partial_path in partial_paths:
            p = Path(partial_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(p), fourcc, fps, (partial_obs_size, partial_obs_size),
            )
            self._partial_writers.append(writer)

        self._partial_writer = self._partial_writers[0] if self._partial_writers else None

    # ── public API ───────────────────────────────────────────────────

    def record(self, frame_rgb: np.ndarray, controlled_pos: np.ndarray):
        """Write one frame.

        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8 RGB image.
        controlled_pos : (2,) or (K, 2) pixel positions for partial crops.
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self._full_writer is not None:
            self._full_writer.write(frame_bgr)

        if self._partial_writers:
            positions = self._normalize_positions(controlled_pos)
            for writer, pos in zip(self._partial_writers, positions):
                crop = self._crop_partial(frame_rgb, pos)
                writer.write(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._full_writer is not None:
            self._full_writer.release()
            self._full_writer = None
        for writer in self._partial_writers:
            writer.release()
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
        size = self.partial_obs_size
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

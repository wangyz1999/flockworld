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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if full_obs_path is not None:
            p = Path(full_obs_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._full_writer = cv2.VideoWriter(
                str(p), fourcc, fps, (canvas_w, canvas_h),
            )

        if partial_obs_path is not None:
            p = Path(partial_obs_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._partial_writer = cv2.VideoWriter(
                str(p), fourcc, fps, (partial_obs_size, partial_obs_size),
            )

    # ── public API ───────────────────────────────────────────────────

    def record(self, frame_rgb: np.ndarray, controlled_pos: np.ndarray):
        """Write one frame.

        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8 RGB image.
        controlled_pos : (2,) pixel position (x, y) of the controlled agent.
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self._full_writer is not None:
            self._full_writer.write(frame_bgr)

        if self._partial_writer is not None:
            crop = self._crop_partial(frame_rgb, controlled_pos)
            self._partial_writer.write(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._full_writer is not None:
            self._full_writer.release()
            self._full_writer = None
        if self._partial_writer is not None:
            self._partial_writer.release()
            self._partial_writer = None

    # ── internals ────────────────────────────────────────────────────

    def _crop_partial(self, frame_rgb: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Extract a square crop around *pos*, padding with black at edges."""
        h, w = frame_rgb.shape[:2]
        half = self.partial_obs_size // 2
        cx, cy = int(pos[0]), int(pos[1])

        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + self.partial_obs_size, y0 + self.partial_obs_size

        # Source region clamped to image bounds
        sx0 = max(x0, 0)
        sy0 = max(y0, 0)
        sx1 = min(x1, w)
        sy1 = min(y1, h)

        crop = np.zeros((self.partial_obs_size, self.partial_obs_size, 3), dtype=np.uint8)

        # Destination offsets
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)

        crop[dy0:dy1, dx0:dx1] = frame_rgb[sy0:sy1, sx0:sx1]
        return crop

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

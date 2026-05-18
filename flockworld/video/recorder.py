"""Video recording utilities for full and partial observability."""

from __future__ import annotations

import math
import subprocess
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
        size = self.partial_obs_size
        h, w = frame_rgb.shape[:2]
        cx, cy = float(pos[0]), float(pos[1])
        crop = cv2.getRectSubPix(frame_rgb, (size, size), (cx, cy))
        # getRectSubPix border-replicates pixels outside the canvas, which smears
        # edge boid colors as a streak toward the center.  Zero those regions out.
        half = (size - 1) / 2.0
        col0 = max(0, math.ceil(half - cx))
        col1 = min(size, math.floor(w - 1 - cx + half) + 1)
        row0 = max(0, math.ceil(half - cy))
        row1 = min(size, math.floor(h - 1 - cy + half) + 1)
        if col0 > 0: crop[:, :col0] = 0
        if col1 < size: crop[:, col1:] = 0
        if row0 > 0: crop[:row0] = 0
        if row1 < size: crop[row1:] = 0
        return crop

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DlpackNvencVideoRecorder:
    """Encode JAX-rendered chunks through NVENC without staging frames on CPU.

    This backend expects batched JAX frame chunks on a CUDA device. Encoded
    packets are small enough to mux through ffmpeg without defeating the main
    goal: avoiding CPU transfer of raw video frames.
    """

    def __init__(
        self,
        full_obs_path: str | Path | None = None,
        partial_obs_path: str | Path | None = None,
        partial_obs_size: int = 128,
        fps: int = 30,
        canvas_w: int = 800,
        canvas_h: int = 800,
        codec: str = "h264",
        gpu_id: int = 0,
        preset: str = "p1",
        bitrate: str | int = "20M",
    ):
        try:
            import PyNvVideoCodec as nvc
            import torch
        except ImportError as exc:
            raise ImportError(
                "video.backend=pynv requires PyNvVideoCodec and torch. "
                "Install them in the active environment and run on an NVIDIA GPU."
            ) from exc

        self.partial_obs_size = partial_obs_size
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.fps = int(fps)
        self.codec = codec
        self.gpu_id = int(gpu_id)
        self.preset = preset
        self.bitrate = bitrate
        self._nvc = nvc
        self._torch = torch
        self._full_sink = None
        self._partial_sink = None

        if full_obs_path is not None:
            self._full_sink = _NvencBitstreamSink(
                nvc=nvc,
                path=Path(full_obs_path),
                width=canvas_w,
                height=canvas_h,
                fps=self.fps,
                codec=codec,
                gpu_id=self.gpu_id,
                preset=preset,
                bitrate=bitrate,
            )

        if partial_obs_path is not None:
            self._partial_sink = _NvencBitstreamSink(
                nvc=nvc,
                path=Path(partial_obs_path),
                width=partial_obs_size,
                height=partial_obs_size,
                fps=self.fps,
                codec=codec,
                gpu_id=self.gpu_id,
                preset=preset,
                bitrate=bitrate,
            )

    def record(self, frame_rgb: np.ndarray, controlled_pos: np.ndarray):
        raise RuntimeError(
            "DlpackNvencVideoRecorder only supports batched JAX chunks. "
            "Use video.chunk_size > 1 with an uncontrolled or straight-policy run."
        )

    def record_jax_chunk(self, frames, controlled_positions, n: int | None = None):
        """Write a JAX chunk with shape ``(T, H, W, 3)`` and float RGB pixels."""
        torch = self._torch
        frames_t = torch.from_dlpack(frames)
        if n is not None:
            frames_t = frames_t[:n]

        if frames_t.device.type != "cuda":
            raise RuntimeError("video.backend=pynv requires JAX frames on a CUDA device.")

        frames_u8 = (frames_t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        if self._full_sink is not None:
            self._full_sink.write_frames(_rgb_to_abgr(frames_u8, torch))

        if self._partial_sink is not None:
            positions_t = torch.from_dlpack(controlled_positions)
            if n is not None:
                positions_t = positions_t[:n]
            crops = _crop_partial_torch(frames_u8, positions_t, self.partial_obs_size, torch)
            self._partial_sink.write_frames(_rgb_to_abgr(crops, torch))

    def close(self):
        for sink in (self._full_sink, self._partial_sink):
            if sink is not None:
                sink.close()
        self._full_sink = None
        self._partial_sink = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _NvencBitstreamSink:
    def __init__(
        self,
        nvc,
        path: Path,
        width: int,
        height: int,
        fps: int,
        codec: str,
        gpu_id: int,
        preset: str,
        bitrate: str | int,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.codec = codec.lower()
        self._file = None
        self._ffmpeg = None
        self._closed = False
        self._encoder = self._create_encoder(
            nvc, width, height, fps, self.codec, gpu_id, preset, bitrate,
        )

        if path.suffix.lower() == ".mp4":
            try:
                self._ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        self.codec,
                        "-r",
                        str(fps),
                        "-i",
                        "pipe:0",
                        "-c",
                        "copy",
                        str(path),
                    ],
                    stdin=subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "video.backend=pynv needs ffmpeg on PATH when writing .mp4 outputs. "
                    "Install ffmpeg or use a raw bitstream extension such as .h264."
                ) from exc
        else:
            self._file = path.open("wb")

    @staticmethod
    def _create_encoder(nvc, width, height, fps, codec, gpu_id, preset, bitrate):
        kwargs = {
            "width": width,
            "height": height,
            "format": "ABGR",
            "usecpuinputbuffer": False,
            "gpu_id": gpu_id,
            "codec": codec,
            "preset": preset,
            "fps": str(fps),
            "bitrate": str(bitrate),
        }
        try:
            return nvc.CreateEncoder(**kwargs)
        except TypeError:
            # Newer PyNvVideoCodec uses "fmt" instead of "format"
            kwargs["fmt"] = kwargs.pop("format")
            try:
                return nvc.CreateEncoder(**kwargs)
            except TypeError:
                kwargs["gpuid"] = kwargs.pop("gpu_id")
                kwargs["framerate"] = fps
                kwargs.pop("fps")
                return nvc.CreateEncoder(**kwargs)

    def write_frames(self, frames_abgr):
        for frame in frames_abgr:
            self._write_packet(self._encoder.Encode(frame.contiguous()))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._write_packet(self._encoder.EndEncode())
        if self._ffmpeg is not None:
            assert self._ffmpeg.stdin is not None
            self._ffmpeg.stdin.close()
            returncode = self._ffmpeg.wait()
            if returncode != 0:
                raise RuntimeError(f"ffmpeg failed to mux {self.path} (exit {returncode}).")
        if self._file is not None:
            self._file.close()

    def _write_packet(self, packet):
        if not packet:
            return
        data = bytearray(packet)
        if self._ffmpeg is not None:
            assert self._ffmpeg.stdin is not None
            self._ffmpeg.stdin.write(data)
        elif self._file is not None:
            self._file.write(data)


def _rgb_to_abgr(frames_rgb, torch):
    alpha = torch.full_like(frames_rgb[..., :1], 255)
    return torch.cat(
        [alpha, frames_rgb[..., 2:3], frames_rgb[..., 1:2], frames_rgb[..., 0:1]],
        dim=-1,
    )


def _crop_partial_torch(frames_rgb, positions, size: int, torch):
    import torch.nn.functional as F
    t, h, w, _ = frames_rgb.shape
    # (T, 3, H, W) float for grid_sample
    frames_f = frames_rgb.permute(0, 3, 1, 2).float()
    cx = positions[:, 0].float()
    cy = positions[:, 1].float()
    # pixel offsets for the output patch, centred at 0
    offsets = torch.arange(size, device=frames_rgb.device, dtype=torch.float32) - (size - 1) / 2.0
    # normalised grid coords in [-1, 1] (align_corners=True: -1 = pixel 0, 1 = pixel W-1)
    gx = (cx[:, None] + offsets[None, :]) * 2.0 / (w - 1) - 1.0   # (T, size)
    gy = (cy[:, None] + offsets[None, :]) * 2.0 / (h - 1) - 1.0   # (T, size)
    grid = torch.stack(
        [gx[:, None, :].expand(t, size, size),
         gy[:, :, None].expand(t, size, size)],
        dim=-1,
    )  # (T, size, size, 2)
    crops = F.grid_sample(frames_f, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crops.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)

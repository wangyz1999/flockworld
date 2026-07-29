"""Streaming-mode analogue of ``FlockingLatentMultiDataset`` for eval scripts.

The disk-based multi-agent eval path (``FlockingLatentMultiDataset`` + parquet
``state_action`` files) doesn't exist for streaming-trained checkpoints -- those
configs (``data.streaming.enabled: true``) have no ``data.root``, no precomputed
latent cache, and no recorded GT trajectories on disk. This module generates a
small, fixed set of long held-out episodes directly from the same JAX sim used at
train time (``SimClipGenerator``, with ``return_gt_positions=True``) and VAE-encodes
them, exposing the ``.episodes`` / ``.full_episode(e)`` contract ``eval_flock_multi.py``
already expects.

Normalization: streaming checkpoints save no ``stats.pt`` (see
``modeling/training/flow_trainer.py:_fit_stream_stats`` -- mean/std are fit fresh
from a few TRAIN batches at the start of each run and never persisted). An earlier
version of this module fit its OWN mean/std from the held-out eval episodes and
reused it for every model; measured against the actual value each run logged
(``[streaming] latent stats from 8 batches: mean~... std~...`` in its stdout.log),
that eval-time refit came out ~25% off on std -- large enough to put every rollout
mildly out-of-distribution, not a rounding-error-sized gap. That IS avoidable: the
calibration is a deterministic function of ``(base_seed, dataloader construction)``,
so replaying the exact same first-N-batches draw the trainer did (``fit_train_stats``
below) reproduces a run's true stats exactly (verified bit-for-bit against a logged
run). So: this dataset now stores UN-normalized latents (``full_episode``'s "frames"
key), and callers normalize per-model via ``fit_train_stats(that model's cfg, ...)``
before feeding them in, and decode with that same mean/std. The "ceiling" column
doesn't touch a model at all, so it can be decoded straight from the raw latents
via ``raw_decode_fn`` -- no normalization needed.

Note different training configs can give genuinely different stats, not just
sampling noise: the single-agent config uses ``num_workers=16`` vs. the multi
configs' ``8``, which changes the calibration batches actually drawn even with the
same ``base_seed`` -- so the single-agent "floor" model needs its own replay, not
the multi models' stats.
"""

from __future__ import annotations

import torch

from modeling.data.streaming_flock_dataset import StreamingLatentEncoder, build_streaming_flock_dataloader
from modeling.data.streaming_vae_dataset import (
    SimClipGenerator,
    _episode_seed,
    load_sim_cfg_container,
)

_EVAL_NAMESPACE = 2  # disjoint from _TRAIN_NAMESPACE=0 / _VAL_NAMESPACE=1 in streaming_vae_dataset.py


def fit_train_stats(cfg, vae, device):
    """Recover a streaming checkpoint's true training-time latent (mean, std).

    Replays the exact calibration ``flow_trainer.py:_fit_stream_stats`` runs at
    the start of training (same ``cfg.seed``, same train dataloader construction,
    first ``stats_batches`` batches) -- deterministic, so this reproduces the
    checkpoint's real normalization instead of approximating it. Spins up (and
    tears down) the real train dataloader's worker processes; only meant to be
    called a handful of times (once per distinct config in a comparison run).
    """
    encoder = StreamingLatentEncoder(
        vae, device, image_size=tuple(cfg.vae.get("encode_image_size", [128, 128])))
    n = int(cfg.data.streaming.get("stats_batches", 4))
    loader = build_streaming_flock_dataloader(cfg, split="train")
    it = iter(loader)
    pix = [next(it)["pixels"] for _ in range(n)]
    encoder.fit_stats(pix)
    del loader, it
    return encoder.mean.view(-1).cpu(), encoder.std.view(-1).cpu()  # (z_dim,)


def make_decode_fn(vae, device, mean, std):
    """(F, z, h, w) normalized -> (T_pix, 3, H, W), matching train_flock_dit.build_decode_fn."""
    mean = mean.view(1, -1, 1, 1).to(device)
    std = std.view(1, -1, 1, 1).to(device)

    def decode(latent_clip: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = latent_clip.to(device) * std + mean
            pixels = vae.decode(z.permute(1, 0, 2, 3).unsqueeze(0))  # (1,3,T_pix,H,W)
        return pixels[0].permute(1, 0, 2, 3)

    return decode


def raw_decode_fn(vae, device):
    """(F, z, h, w) UN-normalized -> (T_pix, 3, H, W). For ceiling only (no model involved)."""

    def decode(latent_clip: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pixels = vae.decode(latent_clip.to(device).permute(1, 0, 2, 3).unsqueeze(0))
        return pixels[0].permute(1, 0, 2, 3)

    return decode


def normalize_frames(frames: torch.Tensor, mean, std) -> torch.Tensor:
    """(P, T, z, h, w) or (T, z, h, w) raw latents -> normalized, given (z_dim,) mean/std."""
    z = mean.numel()
    shape = (1,) * (frames.dim() - 3) + (z, 1, 1)
    return (frames - mean.view(shape)) / std.view(shape)


def normalize_episode(ep: dict, mean, std) -> dict:
    """Un-normalized full_episode() dict -> a copy with "frames" normalized for one model."""
    out = dict(ep)
    out["frames"] = normalize_frames(ep["frames"], mean, std)
    return out


class StreamingMultiEvalDataset:
    """Fixed set of held-out long multi-agent episodes, generated once from the sim.

    ``full_episode(index)`` mirrors ``FlockingLatentMultiDataset.full_episode``:
    ``frames (P, T, z, h, w)`` -- UN-normalized VAE latents (mu) -- ``actions
    (P, T, A)``, ``context_len``, ``episode_id``, ``agent_indices`` (1-based).
    Normalize with ``normalize_episode(ep, *fit_train_stats(model_cfg, ...))``
    before feeding into a specific model. ``gt_positions(episode_id)`` replaces
    ``gt_project.load_gt_positions`` -- positions come straight from the sim
    instead of a parquet file, shape ``(T_pix, num_sim_boids, 2)``.
    """

    def __init__(self, cfg, vae, device, num_episodes: int, seconds: float, sim_fps: int,
                 eval_seed: int = 0):
        s = cfg.data.streaming
        P = int(cfg.data.num_agents)
        num_context = int(cfg.data.num_context_frames)
        pix_frames = round(float(seconds) * int(sim_fps))

        gen = SimClipGenerator(
            load_sim_cfg_container(str(s.sim_config), list(s.get("sim_overrides", []))),
            num_frames=pix_frames,
            frame_stride=1,
            num_partial_agents=P,
            windows_per_episode=1,
            warmup_steps=int(s.get("warmup_steps", 0)),
            threads=s.get("threads", None),
            return_actions=True,
            return_gt_positions=True,
        )

        # Only need VAE-encode + causal action aggregation here, no normalization
        # (StreamingLatentEncoder's mean/std-dependent methods are never called).
        encoder = StreamingLatentEncoder(
            vae, device, image_size=tuple(cfg.vae.get("encode_image_size", [128, 128])))

        self._items = []
        self._gt = {}
        for e in range(int(num_episodes)):
            seed = _episode_seed(int(eval_seed), _EVAL_NAMESPACE, 0, e)
            clips, actions, gt_positions = gen.episode(seed)   # clips/actions: (P,T,...); gt: (1,T,N,2)
            pixels = torch.from_numpy(clips)[None]             # (1,P,T,H,W,3)
            z = encoder._encode_raw(pixels)[0]                 # (P, L, z, h, w) UN-normalized mu
            acts = torch.from_numpy(actions).to(device, torch.float32)  # (P, T_pix, 2)
            L = z.shape[1]
            agg = torch.stack([encoder._agg_actions(acts[p], L) for p in range(P)])  # (P, L, 2)
            episode_id = f"stream_eval_{seed:08x}"
            self._items.append({
                "frames": z.cpu(),
                "actions": agg.cpu(),
                "context_len": num_context,
                "episode_id": episode_id,
                "agent_indices": list(range(1, P + 1)),
            })
            self._gt[episode_id] = gt_positions[0]  # (T_pix, num_sim_boids, 2) float32

        self.episodes = self._items  # only .__len__ is used by callers

    def full_episode(self, index: int) -> dict:
        return self._items[index]

    def gt_positions(self, episode_id: str):
        return self._gt[episode_id]

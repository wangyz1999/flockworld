"""Pre-encode FlockWorld partial videos into VAE latents and cache to disk.

For each (episode, agent), encodes the full partial video (at the VAE's native
128px) through the frozen VAE into a latent trajectory ``(z_dim, T_lat, h, w)``,
aggregates per-pixel-frame actions to per-latent-frame actions (causal 1+4k
grouping), and caches both as one file. Also computes per-channel latent
mean/std for normalization. Run once; FlockingLatentDataset reads the cache.

Usage:
    uv run python precompute_latents.py --config config/train_flockdit_latent.yaml [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from modeling.configs import load_cfg
from modeling.data.flocking_dit_dataset import FlockingDiTDataset
from modeling.models.frozen_vae import FrozenVAE


def aggregate_actions(actions: torch.Tensor, t_lat: int) -> torch.Tensor:
    """(P, A) per-pixel-frame -> (t_lat, A) per-latent-frame (causal 1+4k groups)."""
    groups = [actions[0:1]]
    for i in range(t_lat - 1):
        groups.append(actions[1 + 4 * i: 1 + 4 * (i + 1)])
    return torch.stack([g.mean(0) for g in groups])


def parse_shard(spec: str, n_total: int) -> tuple[int, int]:
    """'i/N' (0-based) -> (i, N). Shard i processes samples[i::N] (strided, balanced)."""
    i_str, n_str = spec.split("/")
    i, n = int(i_str), int(n_str)
    if not (0 <= i < n):
        raise ValueError(f"--shard {spec}: need 0 <= i < N")
    return i, n


def compute_stats(cache_dir: Path, z_dim: int) -> None:
    """Scan every cached latent file and (re)write stats.pt. Used after sharded runs."""
    csum = torch.zeros(z_dim, dtype=torch.float64)
    csq = torch.zeros(z_dim, dtype=torch.float64)
    count = 0
    files = sorted(cache_dir.glob("ep*_a*.pt"))
    print(f"computing stats over {len(files)} cached files...")
    for j, p in enumerate(files):
        z = torch.load(p, map_location="cpu")["latents"].float().reshape(z_dim, -1)
        csum += z.sum(1).double()
        csq += (z * z).sum(1).double()
        count += z.shape[1]
        if (j + 1) % 5000 == 0:
            print(f"  stats {j + 1}/{len(files)}")
    mean = (csum / count).float()
    std = (csq / count - (csum / count) ** 2).clamp_min(1e-8).sqrt().float()
    torch.save({"mean": mean, "std": std}, cache_dir / "stats.pt")
    print(f"per-channel latent mean={mean.tolist()}")
    print(f"per-channel latent std ={std.tolist()}")
    print(f"wrote {cache_dir / 'stats.pt'}")


def main():
    ap = argparse.ArgumentParser(description="Cache VAE latents for FlockDiT latent-space training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap (episode,agent) count (debug)")
    ap.add_argument("--shard", default=None, help="'i/N' (0-based): this process encodes samples[i::N]. "
                    "Run N processes in parallel, then '--stats-only' once to write stats.pt.")
    ap.add_argument("--stats-only", action="store_true",
                    help="skip encoding; (re)compute stats.pt from all cached files (run after shards).")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    device = "cuda" if (torch.cuda.is_available() and str(cfg.device) != "cpu") else "cpu"
    vae = FrozenVAE(str(cfg.vae.checkpoint_path), device=device)
    img = tuple(int(v) for v in cfg.vae.get("encode_image_size", [128, 128]))
    cache_dir = Path(cfg.data.root) / cfg.data.get("latent_cache_dir", "latent_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    z_dim = vae.z_dim

    if args.stats_only:
        compute_stats(cache_dir, z_dim)
        return

    # enumerate all (episode, agent) over both splits; reuse one ds for action loading
    ds_ref = None
    samples = []
    for split in ("train", "val"):
        ds = FlockingDiTDataset(
            root=cfg.data.root, split=split, val_fraction=cfg.data.val_fraction,
            num_context_frames=1, num_future_frames=1, frame_stride=1,
            image_size=list(img), action_features=list(cfg.data.action_features),
            partial_agent_indices=cfg.data.get("partial_agent_indices", None),
            random_clip=False, split_seed=cfg.seed,
        )
        ds_ref = ds
        samples.extend(ds.samples)
    if args.limit:
        samples = samples[: args.limit]
    n_all = len(samples)
    shard = parse_shard(args.shard, n_all) if args.shard else None
    if shard is not None:
        i, n = shard
        samples = samples[i::n]
        print(f"[shard {i}/{n}] caching {len(samples)} of {n_all} samples -> {cache_dir}")
    else:
        print(f"caching latents for {len(samples)} (episode,agent) samples -> {cache_dir}")

    csum = torch.zeros(z_dim, dtype=torch.float64)
    csq = torch.zeros(z_dim, dtype=torch.float64)
    count = 0
    skipped = 0
    for i, s in enumerate(samples):
        out_path = cache_dir / f"ep{s.episode_id}_a{s.agent_index}.pt"
        if out_path.exists():  # resumable: don't re-encode existing files
            skipped += 1
            continue
        reader = VideoReader(str(s.partial_video), ctx=cpu(0), num_threads=1)
        n = len(reader)
        frames = torch.from_numpy(reader.get_batch(list(range(n))).asnumpy())  # (P,H,W,3)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        if (frames.shape[-2], frames.shape[-1]) != img:
            frames = F.interpolate(frames, size=img, mode="bilinear", align_corners=False)
        pixels = (frames * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,P,H,W) in [-1,1]
        z = vae.encode(pixels)[0].cpu()  # (z_dim, T_lat, h, w)
        t_lat = z.shape[1]
        acts = ds_ref._load_actions(s.state_action, s.agent_index, np.arange(n, dtype=np.int64))
        lat_acts = aggregate_actions(acts, t_lat)  # (T_lat, A)
        torch.save(
            {"latents": z.half(), "actions": lat_acts.float(),
             "episode_id": s.episode_id, "agent_index": s.agent_index},
            out_path,
        )
        zf = z.float().reshape(z_dim, -1)
        csum += zf.sum(1).double()
        csq += (zf * zf).sum(1).double()
        count += zf.shape[1]
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"  {i + 1}/{len(samples)}  latest T_lat={t_lat}  (skipped {skipped})")

    # Inline stats are correct only for a single full pass with nothing skipped.
    # Sharded or resumed runs must recompute globally with `--stats-only`.
    if shard is None and skipped == 0:
        mean = (csum / count).float()
        std = (csq / count - (csum / count) ** 2).clamp_min(1e-8).sqrt().float()
        torch.save({"mean": mean, "std": std}, cache_dir / "stats.pt")
        print(f"per-channel latent mean={mean.tolist()}")
        print(f"per-channel latent std ={std.tolist()}")
        print(f"done -> {cache_dir}  ({len(samples)} files + stats.pt)")
    else:
        why = "sharded" if shard is not None else f"{skipped} skipped"
        print(f"done -> {cache_dir} ({why}). Now run once to write stats.pt:\n"
              f"  uv run python precompute_latents.py --config {args.config} --stats-only "
              + (" ".join(args.overrides) if args.overrides else ""))


if __name__ == "__main__":
    main()

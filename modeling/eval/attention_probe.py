"""Shared plumbing for cross-view attention interpretability (see scripts/figures/gen_attention_heatmap.py).

Loads a multi-agent FlockDiT checkpoint + its non-streaming val dataset, runs one
forward pass with ``return_attn=True`` at a chosen noise level, and returns the
query agent's attention distribution over (key_frame, key_agent, key_spatial) --
averaged over heads and layers, before any aggregation/visualization choices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from modeling.eval.common import build_model, find_best_checkpoint
from modeling.configs import load_cfg
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.models.frozen_vae import FrozenVAE
from modeling.models.decoding import build_decode_fn

ROOT = "data/recording/20260702160453"
T_MAX = 1000.0


def load_model_and_data(config_path: str, output_dir: str, device: str):
    cfg = load_cfg(config_path, [
        "data.streaming.enabled=false", f"data.root={ROOT}", "data.val_fraction=0.1", f"output_dir={output_dir}",
    ])
    decode_fn = build_decode_fn(cfg)
    ck, val = find_best_checkpoint(Path(output_dir) / "checkpoints")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
    model.eval()
    ds = FlockingLatentMultiDataset(
        cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
        num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
        random_clip=False, num_agents=int(cfg.data.num_agents))
    return cfg, model, ds, decode_fn, ck, val


@torch.no_grad()
def extract_dist_from_ep(model, cfg, ep, query_agent: int, sigma: float, device: str,
                          ctx_override: torch.Tensor | None = None, seed: int = 0):
    """Core forward pass + aggregation, given an already-built ``ep`` dict.

    Returns (dist, ctx, f, p, s, h, w); see ``extract_dist`` for what ``dist`` is.
    """
    ctx = int(ep["context_len"])
    wf = int(cfg.data.num_future_frames)
    T = ctx + wf
    frames = ep["frames"][:, :T].clone()
    if ctx_override is not None:
        frames[:, :ctx] = ctx_override.to(frames.device, frames.dtype)
    frames = frames.unsqueeze(0).to(device)   # (1, P, T, C, H, W)
    actions = ep["actions"][:, :T].unsqueeze(0).to(device)
    b, p, f, c, h, w = frames.shape

    torch.manual_seed(seed)
    x = frames.clone()
    x[:, :, ctx:] = torch.randn(b, p, wf, c, h, w, device=device)
    t = torch.zeros(b, p, f, device=device)
    t[:, :, ctx:] = sigma * T_MAX

    _, attn_maps = model(x, t, actions, return_attn=True)
    attn = torch.stack(attn_maps, dim=0).mean(dim=0).mean(dim=1)[0]  # (L, L)
    s = h * w
    attn6 = rearrange(attn, "(fq pq sq) (fk pk sk) -> fq pq sq fk pk sk", fq=f, pq=p, sq=s, fk=f, pk=p, sk=s)
    qf = f - 1
    dist = attn6[qf, query_agent].mean(dim=0)  # (fk, pk, sk)
    return dist, ctx, f, p, s, h, w


@torch.no_grad()
def extract_dist(model, cfg, ds, episode: int, query_agent: int, sigma: float, device: str,
                  ctx_override: torch.Tensor | None = None, seed: int = 0):
    """Returns (dist, ep, ctx, f, p, s, h, w).

    dist: (fk, pk, sk) attention distribution of the query (last frame, query_agent),
    averaged over the query's own spatial tokens, heads, and layers. Rows still sum
    to ~1.0 over the full (fk, pk, sk) key set.

    ``ctx_override``: optional (P, ctx, C, H, W) tensor to substitute for the real
    context latents (e.g. border-masked-then-re-encoded), for ablation tests --
    everything else (actions, future noise, mask) stays identical.
    """
    ep = ds.full_episode(episode)
    dist, ctx, f, p, s, h, w = extract_dist_from_ep(
        model, cfg, ep, query_agent, sigma, device, ctx_override=ctx_override, seed=seed)
    return dist, ep, ctx, f, p, s, h, w


def mask_border_and_reencode(ep, ctx: int, cfg, device: str, border_px: int = 4):
    """Re-render each context frame's border as background color, then VAE re-encode.

    Ablation for the "is the corner attention hotspot actually the rendered white
    border?" question: decode the real context latents -> pixels, zero out the
    outer ``border_px`` pixels (rendered border_width=2 + a little anti-aliasing
    margin) to the background color, re-encode with the SAME frozen VAE, and
    renormalize with the SAME cached stats -- so the only thing that changes is
    the border pixels, nothing else about the pipeline.

    Returns (P, ctx, C, h, w) normalized latents, same convention as ep["frames"].
    """
    vae = FrozenVAE(str(cfg.vae.checkpoint_path), device=device)
    stats_path = Path(ROOT) / "latent_cache" / "stats.pt"
    stats = torch.load(stats_path, map_location=device)
    mean = stats["mean"].view(1, -1, 1, 1).to(device)
    std = stats["std"].view(1, -1, 1, 1).to(device)

    ctx_latents = ep["frames"][:, :ctx].to(device)  # (P, ctx, z, h, w) normalized
    p = ctx_latents.shape[0]
    out = torch.empty_like(ctx_latents)
    bg = torch.zeros(3, device=device)  # rendering.background_color = [0,0,0]
    for j in range(p):
        z = ctx_latents[j] * std[0] + mean[0]              # denormalize (ctx, z, h, w)
        pixels = vae.decode(z.permute(1, 0, 2, 3).unsqueeze(0))  # (1, 3, ctx, H, W)
        pixels = pixels[0].permute(1, 0, 2, 3).clone()      # (ctx, 3, H, W) in [-1, 1]
        bp = border_px
        pixels[:, :, :bp, :] = bg.view(1, 3, 1, 1)
        pixels[:, :, -bp:, :] = bg.view(1, 3, 1, 1)
        pixels[:, :, :, :bp] = bg.view(1, 3, 1, 1)
        pixels[:, :, :, -bp:] = bg.view(1, 3, 1, 1)
        mu = vae.encode(pixels.permute(1, 0, 2, 3).unsqueeze(0))  # (1, z, ctx, h, w)
        z_norm = (mu[0].permute(1, 0, 2, 3) - mean[0]) / std[0]   # (ctx, z, h, w)
        out[j] = z_norm
    return out.cpu()


CORNER_IDX = (0, 7, 56, 63)   # flat (row*8+col) indices of an 8x8 grid's 4 corners
CENTER_IDX = (27, 28, 35, 36)  # the 4 innermost cells, as a reference "non-edge" region
RING_IDX = tuple(r * 8 + c for r in range(8) for c in range(8) if r in (0, 7) or c in (0, 7))  # 28 outer cells
INTERIOR_IDX = tuple(i for i in range(64) if i not in RING_IDX)                                # 36 inner cells


def corner_heaviness(per_agent_ctx: torch.Tensor, query_agent: int) -> float:
    """per_agent_ctx: (p, sk=64) summed context-frame attention per key agent.

    Returns mean(corner cells) / mean(center cells), averaged over all OTHER
    agents (excludes the query's own tile). >1 means corners draw disproportionate
    attention relative to the view's own center.
    """
    other = [j for j in range(per_agent_ctx.shape[0]) if j != query_agent]
    sub = per_agent_ctx[other]  # (P-1, 64)
    corner = sub[:, list(CORNER_IDX)].mean().item()
    center = sub[:, list(CENTER_IDX)].mean().item()
    return corner / (center + 1e-12)


EDGE_ONLY_IDX = tuple(i for i in RING_IDX if i not in CORNER_IDX)  # 24 border-but-not-corner cells


def corner_edge_interior(per_agent_ctx: torch.Tensor, query_agent: int) -> tuple[float, float, float]:
    """Three-way split: mean attention density (per cell) in corner (4 cells),
    edge-only (24 cells, border minus corners), and interior (36 cells) -- all
    on the SAME scale so their magnitudes are directly comparable, not just
    each one's ratio to interior.
    """
    other = [j for j in range(per_agent_ctx.shape[0]) if j != query_agent]
    sub = per_agent_ctx[other]  # (P-1, 64)
    corner = sub[:, list(CORNER_IDX)].mean().item()
    edge = sub[:, list(EDGE_ONLY_IDX)].mean().item()
    interior = sub[:, list(INTERIOR_IDX)].mean().item()
    return corner, edge, interior


def ring_heaviness(per_agent_ctx: torch.Tensor, query_agent: int) -> float:
    """Same idea as ``corner_heaviness`` but over the full 28-cell outer ring vs.
    the 36-cell interior, not just the 4 literal corners -- catches "hot edges"
    that aren't corner-shaped (e.g. a boid mid-exit along one side).
    """
    other = [j for j in range(per_agent_ctx.shape[0]) if j != query_agent]
    sub = per_agent_ctx[other]  # (P-1, 64)
    ring = sub[:, list(RING_IDX)].mean().item()
    interior = sub[:, list(INTERIOR_IDX)].mean().item()
    return ring / (interior + 1e-12)


def boid_density_ring_ratio(ep, ctx: int, agent_indices: list[int], query_agent: int,
                             parquet_path, num_sim_boids: int = 100) -> float:
    """GT content control for ``ring_heaviness``: same ring-vs-interior ratio, but
    for REAL boid density (from ground-truth positions), not attention.

    If attention's ring/interior ratio exceeds this content baseline, the model is
    looking at edges MORE than the actual boid density there would justify -- i.e.
    a genuine edge/boundary effect, not just "more boids happen to be at edges."

    Uses the pixel frame corresponding to the context's last latent frame
    (causal grouping: latent frame ``ctx-1`` covers pixel frames up to
    ``4*(ctx-1)``, so we use that pixel frame as the representative GT snapshot).
    """
    from modeling.eval import gt_project as gp

    positions = gp.load_gt_positions(parquet_path, num_sim_boids)  # (T_pix, num_boids, 2)
    pixel_frame = 4 * (ctx - 1)
    counts = np.zeros((len(agent_indices), 64), dtype=np.float64)
    for j, world_idx_1based in enumerate(agent_indices):
        idx, px = gp.visible_boids(positions[pixel_frame], world_idx_1based - 1, size=128)
        cells = (px[:, 1] // 16).astype(int) * 8 + (px[:, 0] // 16).astype(int)  # (row*8+col)
        cells = cells[(cells >= 0) & (cells < 64)]
        for c in cells:
            counts[j, c] += 1
    other = [j for j in range(len(agent_indices)) if j != query_agent]
    sub = counts[other]
    ring = sub[:, list(RING_IDX)].mean()
    interior = sub[:, list(INTERIOR_IDX)].mean()
    return float(ring / (interior + 1e-12))


def synth_border0_episode(cfg, seed: int, ctx: int, wf: int, device: str):
    """Render a FRESH, self-consistent episode straight from the sim with
    ``rendering.border_width=0`` (no post-hoc masking, no artificial edge --
    real in-distribution content the VAE has genuinely never seen a border on),
    encode through the same frozen VAE, and return an ``ep``-shaped dict usable
    with ``extract_dist_from_ep``.

    Unlike ``mask_border_and_reencode`` (which only swaps the context of a REAL
    recorded episode, keeping its original future/actions), this generates
    context AND future together from one sim rollout, so nothing is mismatched.
    """
    from modeling.data.streaming_vae_dataset import SimClipGenerator, load_sim_cfg_container

    s = cfg.data.streaming
    sim_cfg_container = load_sim_cfg_container(str(s.sim_config), list(s.sim_overrides) + ["rendering.border_width=0"])
    num_frames = 1 + 4 * (ctx + wf - 1)  # T_pix such that T_lat == ctx+wf
    p = int(cfg.data.num_agents)
    gen = SimClipGenerator(
        sim_cfg_container, num_frames=num_frames, frame_stride=1, num_partial_agents=p,
        windows_per_episode=1, warmup_steps=int(s.warmup_steps), threads=int(s.get("threads", 2)),
        return_actions=True,
    )
    # windows_per_episode=1 -> episode() already returns the windows*agents axis flattened
    # to just (P, T_pix, 128, 128, 3) / (P, T_pix, 2), agent-major order.
    clips, actions = gen.episode(seed)

    vae = FrozenVAE(str(cfg.vae.checkpoint_path), device=device)
    stats = torch.load(Path(ROOT) / "latent_cache" / "stats.pt", map_location=device)
    mean = stats["mean"].view(1, -1, 1, 1).to(device)
    std = stats["std"].view(1, -1, 1, 1).to(device)

    def agg_actions(a, t_lat):  # (T_pix, 2) -> (t_lat, 2), causal 1+4k groups (matches modeling/cli/precompute_latents.py)
        groups = [a[0:1]]
        for i in range(t_lat - 1):
            groups.append(a[1 + 4 * i: 1 + 4 * (i + 1)])
        return torch.stack([g.mean(0) for g in groups])

    frames_out, actions_out = [], None
    for j in range(p):
        pixels = torch.from_numpy(clips[j]).permute(0, 3, 1, 2).float() / 255.0  # (T_pix, 3, H, W)
        pixels = (pixels * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(device)     # (1, 3, T_pix, H, W)
        mu = vae.encode(pixels)  # (1, z, t_lat, 8, 8)
        z_norm = (mu[0].permute(1, 0, 2, 3) - mean[0]) / std[0]  # (t_lat, z, 8, 8)
        frames_out.append(z_norm.cpu())
        a = agg_actions(torch.from_numpy(actions[j]).float(), z_norm.shape[0])
        actions_out = a.unsqueeze(0) if actions_out is None else torch.cat([actions_out, a.unsqueeze(0)], dim=0)

    return {
        "frames": torch.stack(frames_out, dim=0),      # (P, t_lat, z, 8, 8)
        "actions": actions_out,                        # (P, t_lat, 2)
        "context_len": ctx,
        "episode_id": f"synth_seed{seed}_border0",
        "agent_indices": list(range(1, p + 1)),
    }

"""10s rollouts of the tiled multi model, all 10 POVs tiled into one grid per frame.

Arranges views in the model's own tile layout (2 rows x 5 cols, row-major by
agent index) so the grid mirrors what the tiled RoPE actually sees -- lets you
eyeball cross-view consistency across all agents at once, one video per episode.

    uv run python -m scripts.figures.gen_multi_tiled_pov_grid
"""
import os, torch
import numpy as np
from pathlib import Path
from modeling.configs import load_cfg
from modeling.models.decoding import build_decode_fn
from modeling.eval.common import find_best_checkpoint, build_model, write_mp4
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import overlay as ov

ROOT = "data/recording/20260702160453"
OUTD = "output/flock_dit_multi_tiled"
NLAT, NEP, STEPS = 75, 3, 50          # 75 latent ~= 10s; 3 episodes

cfg = load_cfg("config/train_flockdit_latent_multi_stream_tiled.yaml", [
    "data.streaming.enabled=false", f"data.root={ROOT}", "data.val_fraction=0.1", f"output_dir={OUTD}",
])
device = "cuda" if torch.cuda.is_available() else "cpu"
decode_fn = build_decode_fn(cfg)
ck, val = find_best_checkpoint(Path(OUTD) / "checkpoints"); print(f"ckpt {ck} val {val:.4f}", flush=True)
model = build_model(cfg).to(device)
model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); model.eval()

ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))

out = f"{OUTD}/eval/pov_grid"; os.makedirs(out, exist_ok=True)
wf = int(cfg.data.num_future_frames)
rows, cols = tuple(cfg.model.tile_grid)   # [2, 5] -- matches the model's own tiling layout


def tile_fixed(views, rows, cols, pad=2, bg=40) -> np.ndarray:
    T, H, W, _ = views[0].shape
    gh, gw = rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad
    grid = np.full((T, gh, gw, 3), bg, np.uint8)
    for i, v in enumerate(views):
        r, c = divmod(i, cols)
        y0, x0 = pad + r * (H + pad), pad + c * (W + pad)
        grid[:, y0:y0 + H, x0:x0 + W] = v[:T]
    return grid


with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e); ctx = int(ep["context_len"])
        frames = ep["frames"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        ntot = min(NLAT, frames.shape[2])
        clip = fm.multi_autoregressive_rollout(model, frames[:, :, :ctx], actions, ntot,
                                               window_future=wf, num_steps=STEPS)
        views = [ov.to_uint8(decode_fn(clip[0, j])) for j in range(clip.shape[1])]
        grid = tile_fixed(views, rows, cols)
        write_mp4(grid, f"{out}/ep{ep['episode_id']}_grid.mp4", 30)
        print(f"wrote ep{ep['episode_id']} grid ({len(views)} views, {rows}x{cols})", flush=True)
print("DONE ->", out, flush=True)

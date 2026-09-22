"""Diagnose boid vs wall-border detection shapes on GT-decoded frames (repo root).

The detector currently grabs the white world border as a "boid". Boids are small
compact dots; the border is a thin straight LINE (elongated, large bbox extent).
This measures, for every detection, its size / bbox-extent / elongation, and
labels it:
  matched   = within 6px of a GT-visible boid  -> a real boid,
  unmatched = not near any GT boid             -> likely the border (or spurious).
Prints both distributions + an extent-threshold sweep so we set the wall cutoff
from data (keep ~all boids, drop ~all walls). No model needed -- GT frames only.

    uv run python -m scripts.diagnostics.diag_boid_shapes
"""
import numpy as np, torch
from pathlib import Path
from scipy import ndimage
from modeling.configs import load_cfg
from modeling.models.decoding import build_decode_fn
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling.eval import gt_project as gp, overlay as ov

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
NEP, NLAT, MATCH = 3, 12, 6.0            # episodes, latent frames to decode/agent, GT match radius (px)
V_THRESH, MIN_SIZE, NUM_BOIDS = 0.25, 2, 100
MAX_ELONG, MAX_EXTENT = 5.0, 40          # mirror detect_boids: reject border lines/corners
BORDER_PX, BORDER_MIN_SIZE = 5, 25       # mirror detect_boids: reject small detections hugging the frame edge
EDGE = 6                                 # a residual detection within EDGE px of the frame border = wall-like

cfg = load_cfg("config/train_flockdit_latent_multi.yaml", [f"data.root={ROOT}", f"output_dir={OUTD}"])
device = "cuda" if torch.cuda.is_available() else "cpu"
decode_fn = build_decode_fn(cfg)
ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))


def detections(frame_rgb):
    """-> list of (size, extent, elong, u, v) for each detection passing the min-size filter."""
    f = np.asarray(frame_rgb, np.float32); f = f / 255.0 if f.max() > 1.5 else f
    V = f.max(-1)
    lbl, n = ndimage.label(V > V_THRESH)
    if n == 0:
        return []
    objs = ndimage.find_objects(lbl)
    coms = ndimage.center_of_mass(V, lbl, np.arange(1, n + 1))   # list of (row, col)
    H, W = V.shape
    out = []
    for i in range(1, n + 1):
        sl = objs[i - 1]
        size = int((lbl[sl] == i).sum())
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        extent, elong = max(h, w), max(h, w) / max(min(h, w), 1)
        row, col = coms[i - 1]
        on_border = col < BORDER_PX or col > W - BORDER_PX or row < BORDER_PX or row > H - BORDER_PX
        if (size < MIN_SIZE or elong > MAX_ELONG or extent > MAX_EXTENT     # mirror detect_boids' wall rejection
                or (on_border and size < BORDER_MIN_SIZE)):
            continue
        out.append((size, extent, elong, col, row))   # u=col, v=row
    return out


matched, unmatched = [], []      # each: (size, extent, elong, u, v)
n_frames = 0
with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e)
        pos = gp.load_gt_positions(Path(cfg.data.root) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
        for j in range(ep["frames"].shape[0]):
            frames = ov.to_uint8(decode_fn(ep["frames"][j, :NLAT]).detach().cpu())   # pixel frames, frame t <-> pos[t]
            cam = ep["agent_indices"][j] - 1
            for t in range(min(len(frames), pos.shape[0])):
                n_frames += 1
                _, gpx = gp.visible_boids(pos[t], cam)
                gpx = np.asarray(gpx, np.float32).reshape(-1, 2)
                for size, extent, elong, u, v in detections(frames[t]):
                    near = len(gpx) and np.min(np.linalg.norm(gpx - np.array([u, v]), axis=1)) <= MATCH
                    (matched if near else unmatched).append((size, extent, elong, u, v))
        print(f"ep{ep['episode_id']}: matched {len(matched)} unmatched {len(unmatched)} (cumulative)", flush=True)


def stats(name, arr):
    a = np.asarray(arr, np.float32)
    if not len(a):
        print(f"  {name}: none"); return
    for k, col in zip(["size", "extent", "elong"], a.T):
        q = np.percentile(col, [50, 90, 99, 100])
        print(f"  {name:>10} {k:>7}: median {q[0]:6.1f}  p90 {q[1]:6.1f}  p99 {q[2]:6.1f}  max {q[3]:6.1f}")


M, U = np.asarray(matched, np.float32), np.asarray(unmatched, np.float32)
print(f"\n=== POST-FILTER detection shapes (matched={len(M)} real boids | unmatched={len(U)} residual) ===")
stats("matched", M); stats("unmatched", U)

# Chase the residual over-detection: what ARE the surviving unmatched detections?
if len(U):
    u, v = U[:, 3], U[:, 4]
    edge = (u < EDGE) | (u > 128 - EDGE) | (v < EDGE) | (v > 128 - EDGE)   # border-hugging => wall-like
    print(f"\n=== residual unmatched analysis ({len(U)} detections) ===")
    print(f"  near frame border (<{EDGE}px, wall-like): {edge.mean():.2f}")
    print(f"  interior (split/genuine boid):            {1 - edge.mean():.2f}")
    print(f"  of border-hugging: median elong {np.median(U[edge, 2]) if edge.any() else float('nan'):.1f}, "
          f"extent {np.median(U[edge, 1]) if edge.any() else float('nan'):.1f}")
    print(f"  residual unmatched per frame = {len(U) / max(n_frames, 1):.2f}  (this is the +over-detection)")
print("\nborder-hugging + elongated => tighten max_elong/max_extent; interior + compact => split boids (not walls)")

"""Feasibility check for the wall-anchored consistency probe (repo root, GT-only).

Before building a wall/corner-anchored metric, measure how often two of the 10
camera agents are near the SAME wall / SAME corner at the SAME frame -- and how
often that ALSO coincides with their crops overlapping (the condition the probe
needs: both anchored by a wall AND sharing a view). Pure GT geometry from the
parquet; no model, no decode.

    uv run python diag_wall_cooccurrence.py
"""
import numpy as np
from pathlib import Path
from modeling.eval import gt_project as gp

ROOT = "data/recording/20260623_135735"
WORLD = 720          # canvas is 720x720
HALF = 64            # crop half-size: a wall is in view when the agent is within HALF of that boundary
CROP = 128           # two crops overlap when agents are within CROP in both axes
NUM_CAM = 10         # first 10 boids are the camera agents
NUM_BOIDS = 100
N_EP = 20            # episodes to scan


def wall_flags(xy):
    """(...,2) -> (...,4) booleans [near_left, near_right, near_top, near_bottom]."""
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([x < HALF, x > WORLD - HALF, y < HALF, y > WORLD - HALF], axis=-1)


def corner_flags(w):
    """wall flags (...,4)[L,R,T,B] -> (...,4) corner booleans [TL, TR, BL, BR]."""
    L, R, T, B = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    return np.stack([L & T, R & T, L & B, R & B], axis=-1)


eps = sorted(Path(ROOT, "state_action").glob("*.parquet"))[:N_EP]
print(f"scanning {len(eps)} episodes, {NUM_CAM} camera agents, world {WORLD}px, wall-in-view within {HALF}px\n")

tot_pf = 0                      # total pair-frames
c_overlap = 0                   # crops overlap (within CROP in both axes)
c_same_edge = 0                 # both near the same wall
c_same_corner = 0               # both at the same corner
c_edge_and_overlap = 0          # same wall AND crops overlap  (the probe's condition)
c_corner_and_overlap = 0        # same corner AND crops overlap
agent_frames = 0                # single-agent frames
agent_near_wall = 0             # ... near any wall

for ep in eps:
    pos = gp.load_gt_positions(ep, NUM_BOIDS)          # (T, 100, 2)
    cam = pos[:, :NUM_CAM, :]                          # (T, 10, 2)
    T = cam.shape[0]
    wf = wall_flags(cam)                               # (T, 10, 4)
    cf = corner_flags(wf)                              # (T, 10, 4)
    agent_frames += T * NUM_CAM
    agent_near_wall += int(wf.any(-1).sum())
    for t in range(T):
        for a in range(NUM_CAM):
            for b in range(a + 1, NUM_CAM):
                tot_pf += 1
                d = np.abs(cam[t, a] - cam[t, b])
                overlap = bool(d[0] < CROP and d[1] < CROP)
                same_edge = bool((wf[t, a] & wf[t, b]).any())
                same_corner = bool((cf[t, a] & cf[t, b]).any())
                c_overlap += overlap
                c_same_edge += same_edge
                c_same_corner += same_corner
                c_edge_and_overlap += (same_edge and overlap)
                c_corner_and_overlap += (same_corner and overlap)

pf = max(tot_pf, 1)
print(f"single camera agent near ANY wall: {agent_near_wall / max(agent_frames,1):.3f} of agent-frames\n")
print(f"{'condition (over pair-frames)':<40}{'frac':>10}{'count':>12}")
print(f"{'crops overlap':<40}{c_overlap/pf:>10.4f}{c_overlap:>12}")
print(f"{'both near SAME wall (edge)':<40}{c_same_edge/pf:>10.4f}{c_same_edge:>12}")
print(f"{'both at SAME corner':<40}{c_same_corner/pf:>10.4f}{c_same_corner:>12}")
print(f"{'same wall AND overlap  <- probe (edge)':<40}{c_edge_and_overlap/pf:>10.4f}{c_edge_and_overlap:>12}")
print(f"{'same corner AND overlap <- probe (2D)':<40}{c_corner_and_overlap/pf:>10.4f}{c_corner_and_overlap:>12}")
print(f"\ntotal pair-frames scanned: {tot_pf}")
print("decision: 'same wall AND overlap' count is the usable-sample size for the wall-anchored probe.")

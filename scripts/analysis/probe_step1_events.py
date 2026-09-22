"""Wall-probe step 1 (repo root, GT-only): list corner-share events.

Scans episodes and finds every event where two of the 10 camera agents share the
SAME corner with overlapping crops (maximal runs of consecutive frames). Prints
how many INDEPENDENT events there are (not just correlated frames), their
durations, per-corner / per-episode spread, and a sample -- so we know the real
sample size before spending any rollout compute.

    uv run python -m scripts.analysis.probe_step1_events
"""
import numpy as np
from pathlib import Path
from modeling.eval import gt_project as gp
from modeling.eval import wall_probe as wp

ROOT = "data/recording/20260623_135735"
NUM_CAM, NUM_BOIDS, N_EP, MIN_LEN = 10, 100, 20, 1

eps = sorted(Path(ROOT, "state_action").glob("*.parquet"))[:N_EP]
cam_idx = list(range(NUM_CAM))            # first 10 boids are the camera agents
print(f"scanning {len(eps)} episodes for corner-share events (min_len={MIN_LEN})\n")

all_events = []      # (episode_id, event dict)
for ep in eps:
    pos = gp.load_gt_positions(ep, NUM_BOIDS)
    for ev in wp.find_corner_events(pos, cam_idx, min_len=MIN_LEN):
        all_events.append((ep.stem, ev))

if not all_events:
    print("no corner-share events found."); raise SystemExit

lens = np.array([ev["len"] for _, ev in all_events])
eids = sorted({eid for eid, _ in all_events})
print(f"independent events: {len(all_events)}   total frames: {int(lens.sum())}")
print(f"event length (frames): median {np.median(lens):.0f}  mean {lens.mean():.1f}  "
      f"min {lens.min()}  max {lens.max()}")
print(f"episodes with >=1 event: {len(eids)}/{len(eps)}\n")

print("per corner:")
for c in wp.CORNERS:
    ce = [ev for _, ev in all_events if ev["corner"] == c]
    print(f"  {c}: {len(ce):3d} events, {sum(ev['len'] for ev in ce):5d} frames")

print("\nsample events (episode, agents a-b (cam slot), corner, frames, len):")
for eid, ev in all_events[:20]:
    print(f"  {eid}  a{ev['a']+1}-a{ev['b']+1:<2}  {ev['corner']}  "
          f"[{ev['t0']:4d}..{ev['t1']:4d}]  len {ev['len']}")
if len(all_events) > 20:
    print(f"  ... (+{len(all_events)-20} more)")
print("\nusable events for the corner probe = independent-events count above.")

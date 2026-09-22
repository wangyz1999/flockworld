"""Wall-probe step 3b (repo root): quantify corner convergence + consistency.

Over ALL corner events with an independent seed, roll the model out from the
apart seed through the event, and also decode the GT latents over the same window
(ceiling). In both, detect the corner in each bird's frames and measure:

  * convergence     - do BOTH birds render the SAME corner (per-event 'ever', per-frame),
  * correspondence  - shared boids matched across the two views, anchored on the
                      CORNER-recovered positions (GT-free),
  * mutual_render   - does each bird render the OTHER at its expected spot.

Reports model vs ceiling. This is the Setup-1 emergent-consistency result.

    uv run python -m scripts.analysis.probe_step3_quantify
"""
import numpy as np, torch
from pathlib import Path
from modeling.configs import load_cfg
from modeling.models.decoding import build_decode_fn
from modeling.eval.common import find_best_checkpoint, build_model
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import gt_project as gp, overlay as ov, wall_probe as wp, consistency as cs
from modeling.eval.boid_detect import detect_boids

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
STEPS, NUM_BOIDS, MATCH, MUTUAL_TOL = 50, 100, 8.0, 8.0
MAX_EVENTS = 150         # cap for a first run (~1000 val episodes hold ~1000+ events); raise for tighter CIs

cfg = load_cfg("config/train_flockdit_latent_multi.yaml", [f"data.root={ROOT}", f"output_dir={OUTD}"])
device = "cuda" if torch.cuda.is_available() else "cpu"
decode_fn = build_decode_fn(cfg)
ck, val = find_best_checkpoint(Path(OUTD) / "checkpoints"); print(f"ckpt {ck} val {val:.4f}", flush=True)
model = build_model(cfg).to(device)
model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); model.eval()

ds = FlockingLatentMultiDataset(
    cache_dir=f"{ROOT}/latent_cache", split="val", val_fraction=cfg.data.val_fraction,
    num_context_frames=cfg.data.num_context_frames, num_future_frames=cfg.data.num_future_frames,
    random_clip=False, num_agents=int(cfg.data.num_agents))
CTX, WF = int(cfg.data.num_context_frames), int(cfg.data.num_future_frames)

# collect events with an independent seed (stop once we have MAX_EVENTS)
events = []
scanned = 0
for e in range(len(ds.episodes)):
    if len(events) >= MAX_EVENTS:
        break
    scanned += 1
    ep = ds.full_episode(e)
    pos = gp.load_gt_positions(Path(ROOT) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
    cam = pos[:, [ai - 1 for ai in ep["agent_indices"]], :]
    for ev in wp.find_corner_events(pos, [ai - 1 for ai in ep["agent_indices"]]):
        ts = wp.find_seed_frame(cam[:, ev["a"]], cam[:, ev["b"]], ev["t0"])
        if ts is not None:
            events.append((e, ev, ts))
print(f"events: {len(events)} (from {scanned} val episodes; {len(ds.episodes)} available, cap {MAX_EVENTS})\n", flush=True)

rng = np.random.default_rng(0)   # for the shuffle/null (chance) control

# per-source accumulators (real + _null chance floor)
S = {s: {"ev": 0, "ev_conv": 0, "fr": 0, "fr_conv": 0, "nm": 0, "no": 0, "mut": 0, "mut_hit": 0,
         "nm_null": 0, "no_null": 0, "mut_null_hit": 0}
     for s in ("ceiling", "model")}


def score_pair(fa, fb, acc):
    """fa, fb: (T,H,W,3) uint8 frames of the two agents over the window. Update acc in place;
    return True if the pair ever converged (both same corner)."""
    ever = False
    for t in range(min(len(fa), len(fb))):
        acc["fr"] += 1
        ca, cb = wp.detect_corner(fa[t]), wp.detect_corner(fb[t])
        if not (ca and cb and ca["corner"] == cb["corner"]):
            continue
        acc["fr_conv"] += 1; ever = True
        pa, pb = np.array(ca["agent_xy"], np.float32), np.array(cb["agent_xy"], np.float32)
        da, db = detect_boids(fa[t])["centroids"], detect_boids(fb[t])["centroids"]
        wa, wb = cs.implied_world(da, pa), cs.implied_world(db, pb)              # corner-anchored world pts
        wa_o, wb_o = wa[cs._in_crop(wa, pb)], wb[cs._in_crop(wb, pa)]            # overlap region
        pairs, _ = cs.match_cross_view(wa_o, wb_o, MATCH)
        acc["nm"] += len(pairs); acc["no"] += max(len(wa_o), len(wb_o))
        # CHANCE floor: A's real boids vs RANDOM B content (same count, same geometry pb)
        db_rand = rng.uniform(0, 128, size=db.shape).astype(np.float32) if len(db) else db
        wb_ro = cs.implied_world(db_rand, pb)[cs._in_crop(cs.implied_world(db_rand, pb), pa)]
        pairs_n, _ = cs.match_cross_view(wa_o, wb_ro, MATCH)
        acc["nm_null"] += len(pairs_n); acc["no_null"] += max(len(wa_o), len(wb_ro))
        # mutual rendering: is the OTHER bird drawn at its expected pixel? (+ chance at a random pixel)
        for src_d, exp in ((da, pb - pa + cs.HALF), (db, pa - pb + cs.HALF)):
            if 0 <= exp[0] < 128 and 0 <= exp[1] < 128:                          # expected in view
                acc["mut"] += 1
                if len(src_d) and np.min(np.linalg.norm(src_d - exp, axis=1)) <= MUTUAL_TOL:
                    acc["mut_hit"] += 1
                r = rng.uniform(0, 128, size=2).astype(np.float32)               # random in-view target = chance
                if len(src_d) and np.min(np.linalg.norm(src_d - r, axis=1)) <= MUTUAL_TOL:
                    acc["mut_null_hit"] += 1
    return ever


with torch.no_grad():
    for k, (e, ev, ts) in enumerate(events):
        if k % 10 == 0:
            print(f"  [{k}/{len(events)}] rolling out events...", flush=True)
        ep = ds.full_episode(e)
        frames, acts = ep["frames"].to(device), ep["actions"].to(device)
        Lb, Lend = wp.latent_index(ts), wp.latent_index(ev["t1"])
        start = Lb - CTX
        if start < 0 or Lend + 1 > frames.shape[1]:
            continue
        num_total = (Lend + 1) - start
        a, b = ev["a"], ev["b"]
        gen = fm.multi_autoregressive_rollout(
            model, frames[:, start:Lb].unsqueeze(0), acts[:, start:start + num_total].unsqueeze(0),
            num_total, window_future=WF, num_steps=STEPS)[0][:, CTX:]            # (P, gen_len, z,h,w)
        gt = frames[:, start:start + num_total][:, CTX:]                          # same window, real latents
        for src, lat in (("ceiling", gt), ("model", gen)):
            S[src]["ev"] += 1
            fa = ov.to_uint8(decode_fn(lat[a]).detach().cpu())
            fb = ov.to_uint8(decode_fn(lat[b]).detach().cpu())
            if score_pair(fa, fb, S[src]):
                S[src]["ev_conv"] += 1

print(f"{'metric':>26}{'ceiling':>10}{'model':>10}")
def row(name, num, den):
    print(f"{name:>26}" + "".join(f"{(S[s][num]/max(S[s][den],1)):>10.3f}" for s in ('ceiling', 'model')))
row("convergence (per-event)", "ev_conv", "ev")
row("convergence (per-frame)", "fr_conv", "fr")
row("correspondence", "nm", "no")
row("  correspondence CHANCE", "nm_null", "no_null")
row("mutual_render", "mut_hit", "mut")
row("  mutual_render CHANCE", "mut_null_hit", "mut")
print("\nraw:", {s: S[s] for s in S})
print("real vs CHANCE is the test: if real ~= chance, the agreement is luck (no consistency).")
print("ceiling real should sit FAR above its chance floor (validates the metric).")

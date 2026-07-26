"""Wall-probe step 2 (repo root): roll out corner events from an INDEPENDENT seed.

For each corner event, seed all 10 agents from a moment when the two event birds
are still APART (crops don't overlap -> independent contexts), then roll out
through the corner co-location. The convergence happens in the GENERATED part, so
any consistency is constructed by the model, not handed to it. Write a
side-by-side [agent a | agent b] video: they should start on different scenes and
converge onto the same corner.

    uv run python probe_step2_rollout.py
"""
import os, numpy as np, torch
from pathlib import Path
from modeling.configs import load_cfg
from train_flock_dit import build_decode_fn
from eval_flock_dit import find_best_checkpoint, build_model, write_mp4
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import gt_project as gp, overlay as ov, wall_probe as wp

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
N_VIZ, STEPS, UP, NUM_BOIDS = 6, 50, 4, 100

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
out = f"{OUTD}/eval/wall_probe_step2"; os.makedirs(out, exist_ok=True)

# collect events that have a valid independent seed (both birds apart before converging)
events = []
for e in range(len(ds.episodes)):
    ep = ds.full_episode(e)
    pos = gp.load_gt_positions(Path(ROOT) / "state_action" / f"{ep['episode_id']}.parquet", NUM_BOIDS)
    cam_idx = [ai - 1 for ai in ep["agent_indices"]]
    cam = pos[:, cam_idx, :]
    for ev in wp.find_corner_events(pos, cam_idx):
        t_seed = wp.find_seed_frame(cam[:, ev["a"]], cam[:, ev["b"]], ev["t0"])
        if t_seed is not None:
            events.append((e, ep["episode_id"], ev, t_seed))
print(f"val episodes: {len(ds.episodes)}   events with independent seed: {len(events)}\n", flush=True)

with torch.no_grad():
    shown = 0
    for e, eid, ev, t_seed in events:
        if shown >= N_VIZ:
            break
        ep = ds.full_episode(e)
        frames, acts = ep["frames"].to(device), ep["actions"].to(device)      # (P,T_lat,..), (P,T_lat,A)
        Lb, Lend = wp.latent_index(t_seed), wp.latent_index(ev["t1"])          # generate Lb..Lend, seed apart at Lb
        start = Lb - CTX
        if start < 0 or Lend + 1 > frames.shape[1]:
            print(f"  skip {eid} a{ev['a']+1}-a{ev['b']+1} {ev['corner']} (context/length)"); continue
        num_total = (Lend + 1) - start
        clip = fm.multi_autoregressive_rollout(
            model, frames[:, start:Lb].unsqueeze(0), acts[:, start:start + num_total].unsqueeze(0),
            num_total, window_future=WF, num_steps=STEPS)[0]                   # (P, num_total, z,h,w)
        a, b = ev["a"], ev["b"]
        va = ov.upscale(ov.to_uint8(decode_fn(clip[a]).detach().cpu()), UP)
        vb = ov.upscale(ov.to_uint8(decode_fn(clip[b]).detach().cpu()), UP)
        write_mp4(np.concatenate([va, vb], axis=2),
                  f"{out}/ev{shown:02d}_{eid}_a{a+1}-a{b+1}_{ev['corner']}.mp4", 30)
        print(f"  ev{shown} {eid} a{a+1}-a{b+1} {ev['corner']} seed_pix {t_seed} -> event [{ev['t0']}..{ev['t1']}] "
              f"lat[{Lb}..{Lend}] ({num_total} frames) -> video", flush=True)
        shown += 1
print("\nDONE ->", out, "  (start APART on different scenes, should converge onto the same corner)", flush=True)

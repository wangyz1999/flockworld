"""10s rollouts of the tiled multi model, single-agent style (run from repo root).

For each agent: GT reconstruction (top) over the model's prediction (bottom),
one video per agent. No grid, no overlay -- just the prediction behavior.

    uv run python gen_multi_tiled_10s_rollout.py
"""
import os, torch
from pathlib import Path
from modeling.configs import load_cfg
from train_flock_dit import build_decode_fn
from eval_flock_dit import find_best_checkpoint, build_model, write_mp4
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm

ROOT = "data/recording/20260702160453"
OUTD = "output/flock_dit_multi_tiled"
NLAT, NEP, STEPS = 75, 2, 50          # 75 latent ~= 10s; 2 episodes x all 10 agents

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

out = f"{OUTD}/eval/10s_rollout"; os.makedirs(out, exist_ok=True)
wf = int(cfg.data.num_future_frames)

def gt_over_pred(gt, pred):           # each (T,3,H,W) in [-1,1] -> (T, 2H, W, 3) uint8 RGB
    vid = torch.cat([gt, pred], dim=2).clamp(-1, 1)
    return ((vid + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e); ctx = int(ep["context_len"])
        frames = ep["frames"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        ntot = min(NLAT, frames.shape[2])
        clip = fm.multi_autoregressive_rollout(model, frames[:, :, :ctx], actions, ntot,
                                               window_future=wf, num_steps=STEPS)
        for j in range(clip.shape[1]):
            gt = decode_fn(ep["frames"][j, :ntot])
            pred = decode_fn(clip[0, j])
            write_mp4(gt_over_pred(gt, pred),
                      f"{out}/ep{ep['episode_id']}_a{ep['agent_indices'][j]}_10s.mp4", 30)
        print(f"wrote ep{ep['episode_id']} ({clip.shape[1]} agents)", flush=True)
print("DONE ->", out, flush=True)

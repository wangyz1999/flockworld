"""Visualize the boid detector on GENERATED frames (run from repo root).

Per agent: the model's GENERATED view with a green ring on every detection
(exactly what ``detect_boids`` found). One mp4 per agent, no GT half.

    uv run python -m scripts.figures.gen_detect_viz
"""
import os, torch, numpy as np
from pathlib import Path
from modeling.configs import load_cfg
from modeling.models.decoding import build_decode_fn
from modeling.eval.common import find_best_checkpoint, build_model, write_mp4
from modeling.data.flocking_latent_dataset import FlockingLatentMultiDataset
from modeling import flow_matching as fm
from modeling.eval import overlay as ov
from modeling.eval.boid_detect import detect_boids

ROOT = "data/recording/20260623_135735"
OUTD = "output/flock_dit_multi_10kep_nocolor_nobg"
NLAT, NEP, STEPS, UP = 38, 1, 50, 4     # ~5s, 1 episode x all 10 agents, 4x upscale

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

out = f"{OUTD}/eval/detect_viz"; os.makedirs(out, exist_ok=True)
wf = int(cfg.data.num_future_frames)


def viz(lat):                    # (T,z,h,w) latents -> (T,H,W,3) uint8 generated frames + green detection rings
    frames = ov.to_uint8(decode_fn(lat).detach().cpu())          # (T, 128, 128, 3) RGB
    dets = [detect_boids(frames[t])["centroids"] for t in range(len(frames))]
    return ov.draw_detections(ov.upscale(frames, UP), dets, radius=6, color=(0, 255, 0)), dets


with torch.no_grad():
    for e in range(min(NEP, len(ds.episodes))):
        ep = ds.full_episode(e); ctx = int(ep["context_len"])
        frames = ep["frames"].unsqueeze(0).to(device)
        actions = ep["actions"].unsqueeze(0).to(device)
        ntot = min(NLAT, frames.shape[2])
        clip = fm.multi_autoregressive_rollout(model, frames[:, :, :ctx], actions, ntot,
                                               window_future=wf, num_steps=STEPS)
        for j in range(clip.shape[1]):
            v, d = viz(clip[0, j])
            write_mp4(v, f"{out}/ep{ep['episode_id']}_a{ep['agent_indices'][j]}_detect.mp4", 30)
            print(f"  a{ep['agent_indices'][j]}: generated {np.mean([len(x) for x in d]):.2f}/f detected", flush=True)
        print(f"wrote ep{ep['episode_id']} ({clip.shape[1]} agents)", flush=True)
print("DONE ->", out, flush=True)

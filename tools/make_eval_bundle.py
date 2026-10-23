"""Build a slim, portable checkpoint bundle for the agent-count DF eval.

Copies exactly what jobs/eval_agentcount_df_20261023 reads, with optimizer state
stripped (eval never touches it; it is ~2/3 of every training checkpoint):

  <out>/pretrained/<group>_stream/vae-*.ckpt              (3 frozen VAEs)
  <out>/output/agentcount_df_20260809/<group>/<run>/checkpoints/
        epoch_XXXX.pt        the lowest-val epoch checkpoint (what --checkpoint
                             default picks: it is the only epoch file shipped)
        step_00075000.pt     the step-matched checkpoint

The layout mirrors $FLOCKWORLD_STORAGE_DIR, so on another machine pointing
FLOCKWORLD_STORAGE_DIR at the bundle is the only path change needed. Writes
MANIFEST.tsv (path, bytes, sha256, global_step, val_loss).

Usage:
  uv run python tools/make_eval_bundle.py --out /scratch1/$USER/flockworld_eval_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

SWEEP = "agentcount_df_20260809"
GROUPS = {
    "color_agent": "color_agent_stream/vae-1976-0.0018.ckpt",
    "color_bg": "color_bg_stream/vae-6928-0.0034.ckpt",
    "random_hue": "random_hue_stream/vae-2433-0.0015.ckpt",
}
RUN_KEYS = ["agents02", "agents05", "agents08", "agents10", "agents20", "agents30", "floor"]
STEP_CKPT = "step_00075000.pt"
MODEL_KEEP = ("model", "epoch", "global_step", "train_loss", "val_loss", "config")
VAE_KEEP = ("state_dict", "hyper_parameters", "hparams_name", "epoch", "global_step",
            "pytorch-lightning_version")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_slim(src: Path, dst: Path, keep) -> dict:
    state = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: state[k] for k in keep if k in state}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(slim, tmp)
    tmp.replace(dst)
    return slim


def _best_epoch(ckpt_dir: Path) -> Path:
    """Same rule as eval_flock_dit.find_best_checkpoint (lowest stored val_loss)."""
    best, best_val = None, float("inf")
    for p in sorted(ckpt_dir.glob("epoch_*.pt")):
        v = torch.load(p, map_location="cpu", weights_only=False, mmap=True).get("val_loss")
        if v is not None and float(v) < best_val:
            best, best_val = p, float(v)
    if best is None:
        raise FileNotFoundError(f"no epoch checkpoint with val_loss in {ckpt_dir}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--storage", default=os.environ.get("FLOCKWORLD_STORAGE_DIR",
                                                        "/project2/ustun_1726/flockworld"))
    args = ap.parse_args()
    storage, out = Path(args.storage), Path(args.out)

    rows = []

    def record(dst: Path, slim: dict):
        rows.append((str(dst.relative_to(out)), dst.stat().st_size, _sha256(dst),
                     slim.get("global_step"), slim.get("val_loss")))
        print(f"  {rows[-1][0]}  {rows[-1][1] / 1e6:.0f} MB  step={rows[-1][3]}  val={rows[-1][4]}",
              flush=True)

    for group, vae_rel in GROUPS.items():
        print(f"[{group}]", flush=True)
        dst = out / "pretrained" / vae_rel
        record(dst, _save_slim(storage / "pretrained" / vae_rel, dst, VAE_KEEP))
        for key in RUN_KEYS:
            runs = sorted((storage / "output" / SWEEP / group).glob(f"{group}-{key}-*"))
            if len(runs) != 1:
                raise SystemExit(f"expected one run dir for {group}/{key}, got {runs}")
            src_dir = runs[0] / "checkpoints"
            dst_dir = out / "output" / SWEEP / group / runs[0].name / "checkpoints"
            for src in (_best_epoch(src_dir), src_dir / STEP_CKPT):
                dst = dst_dir / src.name
                record(dst, _save_slim(src, dst, MODEL_KEEP))

    with open(out / "MANIFEST.tsv", "w") as f:
        f.write("path\tbytes\tsha256\tglobal_step\tval_loss\n")
        for r in rows:
            f.write("\t".join("" if x is None else str(x) for x in r) + "\n")
    total = sum(r[1] for r in rows)
    print(f"wrote {len(rows)} files, {total / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()

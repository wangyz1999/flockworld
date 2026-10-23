#!/usr/bin/env python3
"""Pod entry for running the agent-count DF eval on one Greenland node (8x H200).

Shipped by submit_greenland.sh through greenlandtest/gpujob/submit.sh. Stdlib only:
it runs under the stock DLC image's python3, before the project env exists.

  1. pull the code tarball ($JOB_CODE_S3) and the checkpoint bundle ($JOB_BUNDLE_S3)
     onto node-local NVMe, and build the env with `uv sync --frozen`;
  2. smoke-test one eval (color_agent, step75k, N=2, 1 episode) and abort on failure;
  3. run every (mode, group, N) eval of eval_agentcount_df.job as its own task, one per
     GPU, largest N first so the long runs don't trail;
  4. after each task, and at the end, sync eval_results/ (JSON + logs) to
     $S3_OUTPUT_PREFIX/$RUN_NAME/; finally render the plots/tables and sync again.

Knobs (env): MODES, EVAL_GROUPS, AGENT_COUNTS, EPISODES, SECONDS_, STEPS, SKIP_SMOKE=1.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

RUN = os.environ["RUN_NAME"]
S3_OUT = f"{os.environ['S3_OUTPUT_PREFIX']}/{RUN}"
WORK = Path("/tmp/instance_storage/work") / RUN
CODE = WORK / "flockworld"
BUNDLE = WORK / "bundle"
EVAL_ROOT = CODE / "eval_results/agentcount_df_20260809"
JOB = CODE / "jobs/eval_agentcount_df_20261023/eval_agentcount_df.job"

MODES = os.environ.get("MODES", "best step75k").split()
GROUPS = os.environ.get("EVAL_GROUPS", "color_agent color_bg random_hue").split()
COUNTS = [int(n) for n in os.environ.get("AGENT_COUNTS", "2 5 8 10 20 30").split()]

_sync_lock = threading.Lock()


def log(msg):
    print(f"[entry {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(*cmd, **kw):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def sync_results():
    with _sync_lock:
        subprocess.run(["aws", "s3", "sync", str(CODE / "eval_results"), f"{S3_OUT}/eval_results/",
                        "--only-show-errors"], check=False)


def setup():
    WORK.mkdir(parents=True, exist_ok=True)
    sh("aws", "s3", "cp", os.environ["JOB_CODE_S3"], WORK / "code.tgz", "--only-show-errors")
    CODE.mkdir(exist_ok=True)
    sh("tar", "xzf", WORK / "code.tgz", "-C", CODE)
    sh("aws", "s3", "sync", os.environ["JOB_BUNDLE_S3"], BUNDLE, "--only-show-errors")
    # Verify the bundle against its manifest (path, bytes, sha256, ...), skipping the header.
    sh("bash", "-c", "tail -n +2 MANIFEST.tsv | awk -F'\\t' '{print $3\"  \"$1}' | sha256sum -c --quiet",
       cwd=BUNDLE)

    os.environ.update(UV_CACHE_DIR=str(WORK / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(WORK / "uv_python"),
                      UV_LINK_MODE="copy")
    if not shutil.which("uv"):
        sh(sys.executable, "-m", "pip", "install", "--quiet", "--no-input", "uv")
    sh("uv", "sync", "--frozen", cwd=CODE)
    # Every eval below is `uv run`; the env is built once here, so don't re-resolve per task.
    os.environ.update(UV_FROZEN="1", UV_NO_SYNC="1")
    sh("uv", "run", "python", "-c",
       "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
       "'devices', torch.cuda.device_count(), torch.cuda.get_device_name(0))", cwd=CODE)

    paths = CODE / "config/local_paths.conf"
    paths.write_text(f'export FLOCKWORLD_CODE_DIR="{CODE}"\n'
                     f'export FLOCKWORLD_STORAGE_DIR="{BUNDLE}"\n'
                     f'export FLOCKWORLD_VAE_CHECKPOINT="{BUNDLE}/pretrained/color_agent_stream/vae-1976-0.0018.ckpt"\n')
    os.environ["FLOCKWORLD_PATH_CONFIG"] = str(paths)


def run_eval(gpu, mode, group, n, **extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GROUP=group, MODES=mode, AGENT_COUNTS=str(n),
               **extra)
    t0 = time.time()
    rc = subprocess.run(["bash", str(JOB)], cwd=CODE, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    log(f"{'OK' if rc == 0 else 'FAILED'}: {mode} {group} N={n} on GPU {gpu} ({time.time() - t0:.0f}s)")
    return rc == 0


def smoke():
    root = WORK / "smoke"
    ok = run_eval(0, "step75k", "color_agent", 2, EPISODES="1", EVAL_ROOT=str(root))
    for f in sorted((root / "logs").glob("*.log")):
        log(f"--- tail {f.name}")
        print("".join(f.read_text(errors="replace").splitlines(True)[-30:]), flush=True)
    subprocess.run(["aws", "s3", "sync", str(root), f"{S3_OUT}/smoke/", "--only-show-errors"], check=False)
    if not ok:
        log("smoke eval failed; aborting before the full sweep")
        sys.exit(1)


def main():
    setup()
    if os.environ.get("SKIP_SMOKE") != "1":
        smoke()

    n_gpus = int(os.environ.get("NUM_GPUS") or subprocess.run(
        ["bash", "-c", "nvidia-smi -L | wc -l"], capture_output=True, text=True).stdout.strip())
    tasks = queue.Queue()
    order = sorted(((m, g, n) for m in MODES for g in GROUPS for n in COUNTS), key=lambda t: -t[2])
    for t in order:
        tasks.put(t)
    log(f"{len(order)} evals on {n_gpus} GPUs")

    failed = []

    def worker(gpu):  # one thread per GPU, each pulling the next eval off the shared queue
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                return
            if not run_eval(gpu, *task):
                failed.append(task)
            sync_results()

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for mode in MODES:
        for group in GROUPS:
            view = EVAL_ROOT / "_view" / mode / group
            view.mkdir(parents=True, exist_ok=True)
            link = view / "agentcount"
            if not link.exists():
                link.symlink_to(EVAL_ROOT / mode / group)
            subprocess.run(["uv", "run", "python", "plot_eval_sweep.py", "--results-dir", str(view),
                            "--plots-dir", str(EVAL_ROOT / "plots" / mode / group),
                            "--tables-dir", str(EVAL_ROOT / "tables" / mode / group)], cwd=CODE, check=False)
    sync_results()
    log(f"done: {len(order) - len(failed)}/{len(order)} OK; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

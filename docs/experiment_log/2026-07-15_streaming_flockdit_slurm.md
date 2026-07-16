# Streaming FlockDiT Slurm Experiments

- Recorded: 2026-07-15 19:17:20 -07:00
- Slurm account: `ustun_1726`
- Partition: `gpu`
- GPU constraint: `a40|a100`
- Allocation per task: 1 GPU, 26 CPUs, 90 GB RAM, 48 hours
- Submission unit: seven-task Slurm array
- Job definition: `jobs/flockdit_streaming_20260715/flockdit_streaming.job`

## Shared Training Settings

Every experiment uses on-the-fly simulation data with `data.streaming.enabled=true`.
Training workers derive each episode seed from the base seed, worker ID, and a
monotonically increasing per-worker episode counter. Consequently, exact training
windows do not repeat within an uninterrupted stage. Validation intentionally reuses
a fixed 64-window set. A new process resets the episode counters, so the second stage
of a two-stage run can revisit underlying simulation seeds from stage 1.

| Setting | Value |
|---|---|
| Base seed | 42 |
| Multi-agent views | 10 |
| Context / future latent frames | 2 / 4 |
| Pixel frames per streamed clip | 21 |
| Batch size | 4 |
| Clips per epoch | 20,000 |
| Epoch limit | Unlimited (`train.epochs=-1`) |
| Step checkpoint interval | 1,000 optimizer steps |
| Epoch checkpoint interval | Every 2 epochs |
| DiT width / depth / heads | 512 / 8 / 8 |
| Learning rate | 0.0002 |
| Mixed precision | Enabled |
| W&B project | `flock-world-model` |

## Experiment Matrix

Experiment 5, the smaller-world density experiment, was removed. Original IDs are
preserved so configuration comments and historical discussion remain unambiguous.

| ID | Slurm array index | Configuration | Schedule | Spatial coordinates | Action conditioning | Agent identity | Noise timesteps | Training and validation loss |
|---:|---:|---|---|---|---|---|---|---|
| 1 | 0 | `config/train_flockdit_latent_multi_stream.yaml` | 47h40m multi-agent from scratch | Standard per-view RoPE coordinates | Each view receives its own action | Learned agent embedding | Independent per agent and future frame; context fixed clean | Future frames only |
| 2 | 1 | `config/train_flockdit_latent_multi_stream_tiled.yaml` | 47h40m multi-agent from scratch | Ten views tiled in a 2x5 RoPE grid | All ten tagged actions are combined and broadcast | Agent embedding disabled | One timestep per frame shared across all views; context fixed clean | Future frames only |
| 3 | 2 | `config/train_flockdit_latent_multi_stream_df.yaml` | 47h40m multi-agent from scratch | Standard per-view RoPE coordinates | Each view receives its own action | Learned agent embedding | Independent per agent and frame, including context | All context and future frames |
| 4 | 3 | `config/train_flockdit_latent_multi_stream_twostage.yaml` | 11h50m single-agent, then 35h50m multi-agent | Standard RoPE in both stages | Own action in both stages | Multi-agent embedding initialized in stage 2 | Independent; context fixed clean | Future frames only |
| 6 | 4 | `config/train_flockdit_latent_multi_stream_tiled_df.yaml` | 47h40m multi-agent from scratch | Ten views tiled in a 2x5 RoPE grid | All ten tagged actions are combined and broadcast | Agent embedding disabled | One timestep per frame shared across views, including context | All context and future frames |
| 7 | 5 | `config/train_flockdit_latent_multi_stream_triple.yaml` | 11h50m single-agent DF, then 35h50m tiled multi-agent DF | Standard stage 1, tiled 2x5 stage 2 | Own action stage 1, broadcast actions stage 2 | Disabled in stage 2 | DF in both stages; shared across views in stage 2 | All context and future frames |
| 8 | 6 | `config/train_flockdit_latent_multi_stream_twostage_tiled.yaml` | 11h50m single-agent, then 35h50m tiled multi-agent | Standard stage 1, tiled 2x5 stage 2 | Own action stage 1, broadcast actions stage 2 | Disabled in stage 2 | Context fixed clean; shared across views in stage 2 | Future frames only |

The tiled setting is a bundled intervention: tiled RoPE, broadcast actions, disabled
agent embeddings, and shared per-frame timesteps all change together.

## Recorded Metrics

| Destination | Metrics and artifacts |
|---|---|
| W&B | Resolved configuration, parameter count, instantaneous step loss, rolling step loss, epoch train loss, and epoch validation loss |
| Local output | `resolved_config.yaml`, optimizer/model checkpoints, checkpoint train/validation loss, and any enabled videos |
| Slurm logs | Complete stdout and stderr under `logs/flockdit_streaming_20260715/` |

Diffusion-forcing runs compute loss on all frames, while non-DF runs compute loss only
on future frames. Their raw train and validation losses are therefore not directly
comparable. Final model comparisons should use a common rollout evaluation protocol.

## Two-Stage Formulation

Two-stage tasks execute both stages sequentially inside one Slurm allocation. Stage 1
trains the single-agent model, receives `SIGTERM` after 11h50m, finishes its current
optimizer step, writes a final step checkpoint, closes W&B, and exits. The job verifies
that a checkpoint exists before starting stage 2. Stage 2 selects the checkpoint with
the largest global step and loads model weights only. Optimizer state, epoch, and global
step restart from zero; multi-agent-only parameters retain their fresh initialization.

## Graceful Time Limit

Single-stage tasks receive `SIGTERM` from GNU `timeout` after 47h40m. Two-stage tasks
use 11h50m and 35h50m limits, totaling 47h40m. The trainer handles `SIGTERM` and
`SIGINT`, completes the current optimizer step, atomically saves a final checkpoint,
and calls `wandb.finish()`. GNU `timeout` allows up to five additional minutes before
forcing termination. The Slurm allocation retains at least approximately 15 minutes
for final cleanup even in that worst case, and Slurm also requests a warning signal ten
minutes before the allocation limit as a safety fallback.

## Submission

Run from any location inside the repository checkout:

```bash
bash jobs/flockdit_streaming_20260715/submit_all.sh
```

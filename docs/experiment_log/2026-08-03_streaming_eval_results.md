# Streaming FlockDiT Rollout Evaluation

- Recorded: 2026-08-03 10:26:00 -07:00
- Companion to: `docs/experiment_log/2026-07-15_streaming_flockdit_slurm.md` (training setup / experiment matrix)
- Eval tooling: `compare_experiments.py`, `eval_flock_multi.py`, `modeling/eval/streaming_eval_dataset.py`
- Scope: experiments 1, 2, 3, 6 (finished training). Experiments 4, 7, 8 (two-stage) lost
  their stage-2 fine-tune to the DataLoader-worker crash fixed in `b242c22` and are
  currently retraining -- not included here yet.

## Why a new eval path was needed

`eval_flock_multi.py` only worked against the old disk-recorded / precomputed-latent
pipeline. Streaming-trained checkpoints have no `data.root`, no precomputed latent
cache, and no recorded GT trajectories on disk, so none of that eval path applied.
Built to close that gap:

- `SimClipGenerator.return_gt_positions` (additive, off by default): also emits every
  simulated boid's true position, not just the camera agents', for GT-anchored metrics.
- `modeling/eval/streaming_eval_dataset.py`: generates a fixed set of held-out long
  episodes straight from the sim, VAE-encodes them, exposes GT positions -- the
  streaming analogue of `FlockingLatentMultiDataset`.
- `compare_experiments.py`: runs the same held-out episodes through every
  experiment's own checkpoint and prints a single comparison table.

**Normalization caveat (important, affects absolute numbers slightly):** streaming
checkpoints never persist their own training-time latent mean/std (`flow_trainer.py`
fits them fresh from 8 batches at the start of each run and discards them). Each
column below is normalized using its OWN checkpoint's exactly-replayed calibration
(`fit_train_stats`, verified bit-for-bit against that run's logged
`mean~/std~`), not an approximation -- see the module docstring for the full story
(a naive shared refit was measured ~25% off on std before this fix).

## Eval protocol used for the results below

| Setting | Value |
|---|---|
| Held-out episodes | 6, `--eval-seed 0` (distinct seed namespace from train/val) |
| Rollout length | 10s (75 latent frames) |
| Euler steps per window | 50 |
| `ceiling` | Real GT latents, VAE-decoded, no rollout -- isolates VAE reconstruction loss from rollout drift |
| `floor` | Independent single-agent model (exp04's stage-1 checkpoint), each of the 10 camera agents rolled out separately, no cross-attention |
| Command | `uv run python compare_experiments.py --include exp01_baseline exp02_tiled exp03_diffusion_forcing exp06_tiled_df --episodes 6 --seconds 10 --steps 50` |

**Floor caveat:** exp04's stage-1 checkpoint got only ~11h50m of training (by design --
it's stage 1 of a two-stage recipe, not a standalone artifact), versus ~47h40m for
every other column here. Its val_loss was still oscillating (0.347-0.381 over its last
15 epochs), not cleanly converged. Treat `floor` as a weak lower-bound sanity check,
not a rigorous "architecture vs. architecture" bar -- beating it is a low bar; losing
to it would be a real red flag.

## Results (6 episodes x 10s, steps=50)

### Tier A -- per-view fidelity vs GT

| metric | ceiling | floor | exp01_baseline | exp02_tiled | exp03_df | exp06_tiled_df |
|---|---:|---:|---:|---:|---:|---:|
| Detection rate (up) | 0.972 | 0.153 | 0.264 | 0.182 | 0.262 | 0.164 |
| Position error px (down) | 1.490 | 2.783 | 1.968 | 3.330 | 1.964 | 2.692 |
| Detections / frame | 5.011 | 7.524 | 6.723 | 16.788 | 5.096 | 6.731 |
| GT boids visible / frame (volume) | 5.139 | 5.139 | 5.139 | 5.139 | 5.139 | 5.139 |

### Cross-view consistency -- GT-free (agents vs each other, not vs reality)

| metric | ceiling | floor | exp01_baseline | exp02_tiled | exp03_df | exp06_tiled_df |
|---|---:|---:|---:|---:|---:|---:|
| Reciprocity (up) | 0.404 | 0.141 | 0.127 | 0.375 | 0.100 | 0.123 |
| Displacement error px (down) | 3.284 | 66.363 | 65.516 | 66.174 | 34.130 | 36.536 |
| Motion error px/frame (down) | 1.409 | 22.903 | 24.880 | 53.056 | 6.520 | 9.610 |
| White boid match (up) | 0.926 | 0.068 | 0.105 | 0.094 | 0.347 | 0.369 |
| PSNR dB (up) | 27.000 | 13.028 | 12.130 | 9.791 | 15.451 | 14.706 |
| SSIM (up) | 0.946 | 0.666 | 0.602 | 0.405 | 0.718 | 0.690 |
| Sightings / pair-frame (volume) | 0.0534 | 0.2820 | 0.2010 | 0.6375 | 0.0789 | 0.0665 |
| n_reciprocal events (sample size behind the 5 rows above) | 876 | 1632 | 1039 | 9595 | 238 | 302 |

## Per-experiment read (visual + metrics)

**1) exp01 -- one-stage baseline.** Extreme overrendering of boids in a couple of the
POVs, with boids other than the camera agent taking on different colors. Best-or-tied
per-view fidelity of the four. Moderate overrender vs GT (6.7 vs 5.1/frame), matching
the visual. Cross-view agreement is weak but on a modest sample (~1000 events) -- not a
clean read either way.

**2) exp02 -- tiled (RoPE tile grid + broadcast actions + no agent embed + shared
timesteps).** Even more extreme overrendering than baseline; all POVs go chaotic
toward the end of the 10s rollout. Worst per-view fidelity of the four. By far the
most extreme overrender (16.8 vs 5.1/frame, 3x+), matching "chaotic" directly. Its
reciprocity looks second-best after ceiling, but sightings/pair-frame is 12x ceiling's
rate (~9600 events) -- almost certainly agreement-by-density, not real consistency.

**3) exp03 -- diffusion forcing.** Seems to improve rollout stability, a bit better
than baseline, but some overrendering remains. Tied-best per-view fidelity with
baseline. The most volume-accurate of all four -- barely any overrender (5.1 vs GT's
5.1). Best on every consistency metric, but on a much smaller sample than
baseline/floor/tiled (~240 vs ~1000-9600 events) -- suggestive, not confirmed.

**6) exp06 -- tiled+DF.** Stable movement, but collapses all boids to white (loses hue
identity). Worst detection rate of the four -- consistent with still detecting shapes
but failing to hue-match them. Position error middling. Still draws roughly the right
number of shapes despite the color collapse (moderate overrender, not an empty world).
Same small-sample caveat as exp03 applies to its consistency numbers.

## Confirmed VAE finding (affects `ceiling` too, not just rollouts)

Checked whether a camera agent's own identity color stays stable across a 10s episode
by tracking detected hue every 20 latent frames, on the pure `ceiling` path (real GT
latents, VAE-decoded, zero model/rollout involved). Several agents show real hue
drift/jumps mid-episode -- e.g. agent index 10 continuously bounced between hue
~0.65-0.9 with no stable value; agent 2 cleanly jumped from its correct hue (~0.10) to
a wrong one (~0.29) and stayed there. Verified this is not a data-pipeline indexing bug
(the tracked world position is smooth/continuous through the jump, i.e. the crop is
following one physical boid throughout, not silently swapping identities) -- it's a
genuine VAE color-reconstruction instability. It correlates with which POV shows the
worst overrendering in exp01's rollout (the worst-behaved agent in `ceiling` = the
worst-behaved tile in the baseline rollout), though the correlation isn't clean across
all 10 agents on the one spot-check done so far. Not yet quantified across all
episodes/agents.

## Open threads / TODO when adding exp04, exp07, exp08

- Add each to `EXPERIMENTS` in `compare_experiments.py` once their stage-2 retrain
  finishes (checkpoint dir + config, same pattern as the existing four).
- Re-run with `--include exp01_baseline exp02_tiled exp03_diffusion_forcing exp06_tiled_df exp04_two_stage exp07_... exp08_...`
  for a full 8-column table (9 with floor).
- Consider quantifying the VAE per-agent hue-instability finding across all 6 episodes
  x 10 agents (currently a 1-episode spot check) before leaning on it further.
- Consider breaking exp03/exp06's detections into self / correctly-hued-other / white
  buckets to explain why their cross-view sightings volume is so much lower than
  baseline's despite near-identical overall detection volume (exp03 especially: 5.096
  vs GT's 5.139, essentially exact).

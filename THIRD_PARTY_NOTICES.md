# Third-party notices

FlockWorld incorporates and adapts source code from the projects below. For
each one this file records what was taken, where it lives here, and how it was
changed. Full license texts are in [`licenses/`](licenses/). The affected
source files carry the same information in a header comment.

This file covers code. Methods reimplemented from published descriptions,
without using the authors' source, are credited in the README acknowledgments
and cited in the accompanying paper instead.

## Solaris — Apache License 2.0

- Upstream: https://github.com/solaris-wm/solaris (NYU VisionX), commit `68e0ed3`
- License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

| File in this repository | Derived from |
|---|---|
| `modeling/models/flock_dit.py` | `src/models/singleplayer/world_model.py`, `src/models/multiplayer/world_model.py`, `src/models/transformer_utils.py`, `src/models/transformer.py` |
| `modeling/flow_matching.py` | `src/runners/trainer_sp.py` |

Both files have been modified.

`modeling/models/flock_dit.py` is a PyTorch reimplementation of a model written
in JAX/Flax-nnx. It preserves the factorized 3D RoPE split (`rope_params` /
`rope_apply` / `apply_rope_mp`), the `WanRMSNorm` normalization, the
`sinusoidal_embedding_1d` construction, the adaLN-zero DiT block with its six
modulation parameters and
`1/sqrt(dim)` initialization, block-causal attention with a per-frame block
size of `P*S`, the patchify/unpatchify Conv3d scheme, and per-frame token
interleaving across agents. It removes the CLIP
image-to-video cross-attention and replaces the mouse/keyboard action module
with a 2D steering acceleration injected through adaLN. Tiled RoPE, broadcast
action conditioning, the per-agent action tag, and the optional per-layer agent
embedding are additions that are not part of the original.

`modeling/flow_matching.py` reimplements the rectified-flow forward process,
velocity target, and `[0, 1000]` timestep range in PyTorch, and adds per-frame
Diffusion-Forcing timesteps and a context-preserving Euler sampler.

## Wan 2.2 — Apache License 2.0

- Upstream: https://github.com/Wan-Video/Wan2.2
- Paper: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
- License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

`modeling/models/wan_vae.py` is the Wan causal 3D video VAE, vendored with its
copyright header intact. The file has been modified: it is reduced to the
encoder/decoder and the `Wan2_2_VAE` wrapper this project uses. The model is
trained from scratch on this environment at a smaller configuration (8 latent
channels, 16x spatial and 4x temporal compression) rather than loading released
weights.

## cubedhuang/boids — MIT License

- Upstream: https://github.com/cubeDhuang/boids ([live demo](https://boids.dan.onl))
- Copyright (c) 2023 Daniel Huang
- License text: [`licenses/MIT-cubedhuang-boids.txt`](licenses/MIT-cubedhuang-boids.txt)

| File in this repository | Derived from |
|---|---|
| `flockworld/core/boids.py` | the JS flocking update loop |
| `flockworld/rendering/primitives.py` (`sd_js_boid_batch`) | the PixiJS five-point boid geometry |
| `flockworld/rendering/renderer.py` (HSV helper) | the reference color mapping |

These files have been modified. The flocking algorithm is reimplemented in JAX
as a batched, vectorized update, preserving the single vision radius, the
velocity-dot-product alignment bias, Reynolds-style steering clamped by
`max_force` for all three rules, velocity drag, random heading noise, and
min/max speed clamping. One deliberate behavioral change: this arena reflects
at the walls, while the reference wraps positions at the canvas edge.

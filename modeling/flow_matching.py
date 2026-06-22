"""Flow-matching (rectified-flow) objective and sampler for FlockDiT.

Faithful to Solaris (``solaris/src/runners/trainer_sp.py``):

* forward / noising:  ``x_t = (1 - sigma) * x0 + sigma * eps``, ``sigma = t/1000``
* velocity target:    ``v = eps - x0``
* loss:               ``MSE(model(x_t, t, a), v)``  (over future frames only)

Context conditioning uses **Diffusion Forcing**: per-frame timesteps with
context frames pinned to ``t=0`` (clean, ``sigma=0``) and only future frames
noised. The Euler sampler integrates ``sigma: 1 -> 0`` while holding context
frames clean, which supports autoregressive rollout.

All helpers are framework-level (no model coupling) and work for both the
single-agent ``(B, F, ...)`` and multi-agent ``(B, P, F, ...)`` layouts -- the
per-frame timestep tensor just carries the matching leading dims.
"""

from __future__ import annotations

import torch

T_MAX = 1000  # Solaris timestep range is [0, 1000]; sigma = t / T_MAX


def add_noise(x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow forward process.

    Args:
        x0: clean sample, shape ``(..., F, C, H, W)``.
        t:  per-frame timestep in ``[0, T_MAX]``, shape ``(..., F)`` (broadcasts
            over the trailing C, H, W dims).
    Returns ``(x_t, eps)`` where ``eps`` is the sampled noise.
    """
    eps = torch.randn_like(x0)
    sigma = (t / T_MAX).reshape(*t.shape, 1, 1, 1)  # (..., F, 1, 1, 1)
    x_t = (1.0 - sigma) * x0 + sigma * eps
    return x_t, eps


def velocity_target(eps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Flow-matching target velocity ``v = eps - x0``."""
    return eps - x0


def sample_timesteps(
    leading: tuple[int, ...],
    num_frames: int,
    context_len: int,
    device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-frame timesteps for Diffusion Forcing.

    Context frames ``[:context_len]`` are pinned to ``0`` (clean); each future
    frame gets an independent ``t ~ Uniform{0..T_MAX}`` (matching Solaris's
    ``randint(0, 1001)``; independent per frame is the Diffusion-Forcing variant
    used for causal rollout).

    Args:
        leading: leading dims before the frame axis, e.g. ``(B,)`` single-agent
            or ``(B, P)`` multi-agent.
    Returns timestep tensor of shape ``(*leading, num_frames)`` (float).
    """
    t = torch.randint(
        0, T_MAX + 1, (*leading, num_frames), device=device, generator=generator
    ).float()
    t[..., :context_len] = 0.0
    return t


@torch.no_grad()
def euler_rollout(
    model,
    context_frames: torch.Tensor,
    actions: torch.Tensor,
    num_future: int,
    num_steps: int = 50,
) -> torch.Tensor:
    """Generate ``num_future`` frames given clean context via Euler ODE.

    Integrates ``sigma: 1 -> 0`` for the future frames while holding context
    frames clean at every step. Single-agent only (``x`` is ``(B, F, C, H, W)``);
    the multi-agent rollout follows the same recipe with a player axis.

    Args:
        context_frames: ``(B, Fc, C, H, W)`` clean context in ``[-1, 1]``.
        actions:        ``(B, Fc+num_future, A)`` per-frame actions for the whole clip.
        num_future:     number of frames to generate.
    Returns the full clip ``(B, Fc+num_future, C, H, W)``.
    """
    model.eval()
    b, fc, c, h, w = context_frames.shape
    f = fc + num_future
    device = context_frames.device

    x = torch.cat(
        [context_frames, torch.randn(b, num_future, c, h, w, device=device)], dim=1
    )

    # sigma schedule from 1 -> 0; context frames stay at sigma=0 throughout.
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        d_sigma = sigma_next - sigma  # negative
        t = torch.zeros(b, f, device=device)
        t[:, fc:] = sigma * T_MAX
        v = model(x, t, actions)  # predicted velocity (B, F, C, H, W)
        # dx/dsigma = v  =>  x <- x + d_sigma * v   (Euler step)
        x[:, fc:] = x[:, fc:] + d_sigma * v[:, fc:]
    return x

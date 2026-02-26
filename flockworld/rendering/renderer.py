"""Compose a full frame image from boid arrays using JAX SDF rendering.

The renderer works in normalised coordinates: positions in pixel space are
mapped to a [-1, 1] (or aspect-corrected) UV space for the SDF shader, then
the result is returned as an (H, W, 3) float32 array in [0, 1].
"""

import jax
import jax.numpy as jnp

from flockworld.rendering.primitives import render_boid_at


# ── coordinate helpers ───────────────────────────────────────────────────

def build_uv_grid(width: int, height: int) -> jnp.ndarray:
    """Create an (H, W, 2) UV grid in normalised coordinates."""
    aspect = width / height
    x = jnp.linspace(-aspect, aspect, width)
    y = jnp.linspace(1.0, -1.0, height)
    xv, yv = jnp.meshgrid(x, y)
    return jnp.stack([xv, yv], axis=-1)


def pixel_to_uv(positions, width, height):
    """Convert pixel positions (N, 2) to UV coordinates."""
    aspect = width / height
    uv_x = (positions[:, 0] / width) * 2.0 * aspect - aspect
    uv_y = 1.0 - (positions[:, 1] / height) * 2.0
    return jnp.stack([uv_x, uv_y], axis=-1)


def pixel_size_to_uv(size, height):
    """Convert a pixel-space size scalar to UV-space size."""
    return size / height * 2.0


# ── per-pixel rendering ─────────────────────────────────────────────────

def _render_pixel_for_boid(uv, boid_uv, heading, size_uv, color, aa_blur):
    _, mask = render_boid_at(uv, boid_uv, heading, size_uv, color, aa_blur)
    return color * mask, mask


def _composite_pixel(uv, boid_uvs, headings, size_uv, colors, bg_color, aa_blur):
    """Render one pixel: background + all boids composited back-to-front."""
    all_colors, all_masks = jax.vmap(
        _render_pixel_for_boid, in_axes=(None, 0, 0, None, 0, None)
    )(uv, boid_uvs, headings, size_uv, colors, aa_blur)

    def _blend(carry, x):
        c, m = x
        return carry * (1.0 - m) + c * m, None

    color, _ = jax.lax.scan(_blend, bg_color, (all_colors, all_masks))
    return color


# ── full frame (JIT-compiled) ────────────────────────────────────────────

@jax.jit
def render_frame(
    positions, headings,
    uv_grid,
    width, height,
    agent_size, agent_color, controlled_color, background_color,
    aa_blur,
):
    """Render a complete (H, W, 3) float32 frame.

    All arguments are JAX arrays / scalars — no Python objects.
    """
    boid_uvs = pixel_to_uv(positions, width, height)
    size_uv = pixel_size_to_uv(agent_size, height)

    # Negate headings for rendering: in pixel space +Y is downward, but
    # in UV space +Y is upward.  The heading is computed as arctan2(vx, vy)
    # in pixel space, so we negate it so the triangle nose points in the
    # screen-space velocity direction.
    render_headings = -headings

    n_agents = positions.shape[0]
    colors = jnp.broadcast_to(agent_color, (n_agents, 3))
    colors = colors.at[0].set(controlled_color)

    render_row = jax.vmap(
        _composite_pixel, in_axes=(0, None, None, None, None, None, None)
    )
    render_image = jax.vmap(
        render_row, in_axes=(0, None, None, None, None, None, None)
    )

    frame = render_image(
        uv_grid, boid_uvs, render_headings, size_uv, colors, background_color, aa_blur,
    )
    return jnp.clip(frame, 0.0, 1.0)

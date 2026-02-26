"""Compose a full frame image from BoidState using JAX SDF rendering.

The renderer works in normalised coordinates: positions in pixel space are
mapped to a [-1, 1] (or aspect-corrected) UV space for the SDF shader, then
the result is returned as an (H, W, 3) float32 array in [0, 1].
"""

import jax
import jax.numpy as jnp

from flockworld.rendering.primitives import render_triangle_at


# ── coordinate helpers ───────────────────────────────────────────────────

def build_uv_grid(width: int, height: int) -> jnp.ndarray:
    """Create an (H, W, 2) UV grid in normalised coordinates.

    x ∈ [-aspect, aspect], y ∈ [-1, 1] with y increasing upward.
    """
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
    """Colour contribution of one boid at one pixel."""
    _, mask = render_triangle_at(uv, boid_uv, heading, size_uv, color, aa_blur)
    return color * mask, mask


def _render_pixel_all_boids(uv, boid_uvs, headings, size_uv, colors, aa_blur):
    """Composite all boids onto a single pixel.

    Uses a vmap over boids, then composites back-to-front.
    """
    all_colors, all_masks = jax.vmap(
        _render_pixel_for_boid, in_axes=(None, 0, 0, None, 0, None)
    )(uv, boid_uvs, headings, size_uv, colors, aa_blur)
    # all_colors: (N, 3), all_masks: (N,)
    return all_colors, all_masks


def _composite_pixel(uv, boid_uvs, headings, size_uv, colors, bg_color, aa_blur):
    """Render one pixel: background + all boids composited."""
    all_colors, all_masks = _render_pixel_all_boids(
        uv, boid_uvs, headings, size_uv, colors, aa_blur,
    )
    color = bg_color
    # Back-to-front compositing (boid 0 drawn last = on top)
    def _blend(carry, x):
        c, m = x
        return carry * (1.0 - m) + c * m, None

    color, _ = jax.lax.scan(_blend, color, (all_colors, all_masks))
    return color


# ── full frame ───────────────────────────────────────────────────────────

def render_frame(
    boids,
    uv_grid,
    width: int,
    height: int,
    agent_size: float,
    agent_color,
    controlled_color,
    background_color,
    aa_blur: float,
):
    """Render a complete (H, W, 3) float32 frame.

    Parameters
    ----------
    boids : BoidState
    uv_grid : (H, W, 2) pre-built UV grid
    width, height : canvas pixel dimensions
    agent_size : triangle size in pixels
    agent_color : (3,) default boid colour
    controlled_color : (3,) colour for agent 0
    background_color : (3,) background colour
    aa_blur : anti-aliasing softness
    """
    boid_uvs = pixel_to_uv(boids.positions, width, height)
    size_uv = pixel_size_to_uv(agent_size, height)

    n_agents = boids.positions.shape[0]
    colors = jnp.broadcast_to(agent_color, (n_agents, 3))
    colors = colors.at[0].set(controlled_color)

    bg = jnp.array(background_color, dtype=jnp.float32)

    # vmap over H and W dimensions
    render_row = jax.vmap(
        _composite_pixel, in_axes=(0, None, None, None, None, None, None)
    )
    render_image = jax.vmap(
        render_row, in_axes=(0, None, None, None, None, None, None)
    )

    frame = render_image(
        uv_grid, boid_uvs, boids.headings, size_uv, colors, bg, aa_blur,
    )
    return jnp.clip(frame, 0.0, 1.0)

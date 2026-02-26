"""Compose a full frame image from boid arrays using JAX SDF rendering.

Each boid is a plain filled triangle in a single flat colour.  The
controlled agent (index 0) uses ``controlled_color``; all others use
``agent_color``.  No border, no gradient, no colour changes over time.
"""

import jax
import jax.numpy as jnp

from flockworld.rendering.primitives import render_boid_simple, render_boid_fancy


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
    return size / height * 2.0


# ── wing flapping helper ────────────────────────────────────────────────

def _wave(t):
    """Oscillating wave for wing flapping (ported from Processing)."""
    base = 0.5 * jnp.sin(t)
    sharp = 2.5 * jnp.sin(2.0 * t) * jnp.exp(-5.0 * (t - jnp.pi / 2) ** 2)
    return base + sharp


def _wing_openness(step_count, phase_offsets, fps=30.0):
    """Per-boid wing openness in [open_min, open_max]."""
    open_min, open_max = 0.618, 2.218
    t = step_count / fps * 0.4 - phase_offsets
    w = _wave(t)
    return jnp.clip((w - (-0.5)) / (1.38 - (-0.5)) * (open_max - open_min) + open_min,
                     open_min, open_max)


# ── per-pixel rendering (simple — plain filled triangle) ────────────────

def _render_pixel_simple(uv, boid_uv, heading, size_uv, color, aa_blur):
    return render_boid_simple(uv, boid_uv, heading, size_uv, color, aa_blur)


def _composite_pixel_simple(uv, boid_uvs, headings, size_uv,
                            colors, bg_color, aa_blur):
    all_colors, all_masks = jax.vmap(
        _render_pixel_simple,
        in_axes=(None, 0, 0, None, 0, None),
    )(uv, boid_uvs, headings, size_uv, colors, aa_blur)

    def _blend(carry, x):
        c, m = x
        return carry * (1.0 - m) + c * m, None

    color, _ = jax.lax.scan(_blend, bg_color, (all_colors, all_masks))
    return color


# ── per-pixel rendering (fancy — split wings, no outline) ───────────────

def _render_pixel_fancy_flat(uv, boid_uv, heading, size_uv, wing_open,
                             color, aa_blur):
    """Fancy wing shape but filled with a single flat colour, no stroke."""
    return render_boid_fancy(
        uv, boid_uv, heading, size_uv, wing_open,
        color, color, color, aa_blur,
    )


def _composite_pixel_fancy_flat(uv, boid_uvs, headings, size_uv, wing_opens,
                                colors, bg_color, aa_blur):
    all_colors, all_masks = jax.vmap(
        _render_pixel_fancy_flat,
        in_axes=(None, 0, 0, None, 0, 0, None),
    )(uv, boid_uvs, headings, size_uv, wing_opens, colors, aa_blur)

    def _blend(carry, x):
        c, m = x
        return carry * (1.0 - m) + c * m, None

    color, _ = jax.lax.scan(_blend, bg_color, (all_colors, all_masks))
    return color


# ── full frame (JIT-compiled) ────────────────────────────────────────────

def _vmap_image(composite_fn, in_axes):
    """Double-vmap a per-pixel composite function over (H, W)."""
    render_row = jax.vmap(composite_fn, in_axes=in_axes)
    return jax.vmap(render_row, in_axes=in_axes)


@jax.jit
def _render_frame_simple(
    boid_uvs, render_headings, size_uv, colors,
    uv_grid, background_color, aa_blur,
):
    axes = (0, None, None, None, None, None, None)
    render_image = _vmap_image(_composite_pixel_simple, axes)
    return render_image(
        uv_grid, boid_uvs, render_headings, size_uv,
        colors, background_color, aa_blur,
    )


@jax.jit
def _render_frame_fancy_flat(
    boid_uvs, render_headings, size_uv, wing_opens, colors,
    uv_grid, background_color, aa_blur,
):
    axes = (0, None, None, None, None, None, None, None)
    render_image = _vmap_image(_composite_pixel_fancy_flat, axes)
    return render_image(
        uv_grid, boid_uvs, render_headings, size_uv, wing_opens,
        colors, background_color, aa_blur,
    )


def render_frame(
    positions, velocities, phase_offsets, step_count,
    uv_grid,
    width, height,
    agent_size, agent_color, controlled_color, background_color,
    aa_blur,
    fancy_shape, flap_wings,
):
    """Render a complete (H, W, 3) float32 frame."""
    boid_uvs = pixel_to_uv(positions, width, height)
    size_uv = pixel_size_to_uv(agent_size, height)
    n_agents = positions.shape[0]

    render_headings = jnp.arctan2(-velocities[:, 0], -velocities[:, 1])

    colors = jnp.broadcast_to(agent_color[None, :], (n_agents, 3))
    colors = colors.at[0].set(controlled_color)

    # Move controlled agent (index 0) to the end so it renders on top
    order = jnp.concatenate([jnp.arange(1, n_agents), jnp.array([0])])
    boid_uvs = boid_uvs[order]
    render_headings = render_headings[order]
    colors = colors[order]

    if fancy_shape:
        wing_opens = _wing_openness(step_count, phase_offsets) if flap_wings \
            else jnp.ones(n_agents)
        wing_opens = wing_opens[order]
        frame = _render_frame_fancy_flat(
            boid_uvs, render_headings, size_uv, wing_opens, colors,
            uv_grid, background_color, aa_blur,
        )
    else:
        frame = _render_frame_simple(
            boid_uvs, render_headings, size_uv, colors,
            uv_grid, background_color, aa_blur,
        )

    return jnp.clip(frame, 0.0, 1.0)

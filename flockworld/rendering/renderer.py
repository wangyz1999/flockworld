"""Compose a full frame image from boid arrays using JAX SDF rendering.

Ports the visual style from the Processing boid sketch: two-tone split
wings, wing flapping, and outline stroke.
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


# ── colour helpers ────────────────────────────────────────────────────────

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


def _controlled_color_gradient(positions, canvas_h):
    """Warm pink-to-purple gradient based on Y position (like Processing leader)."""
    palette = jnp.array([
        [1.0, 0.502, 0.447],    # salmon  rgb(255,128,114)
        [0.941, 0.478, 0.631],  # pink    rgb(240,122,161)
        [0.906, 0.459, 0.816],  # orchid  rgb(231,117,208)
        [0.867, 0.435, 1.0],    # violet  rgb(221,111,255)
    ])
    portion = positions[:, 1] / canvas_h  # (N,)
    idx_f = portion * (palette.shape[0] - 1)
    idx_lo = jnp.clip(jnp.floor(idx_f).astype(jnp.int32), 0, palette.shape[0] - 2)
    frac = (idx_f - idx_lo.astype(jnp.float32))[..., None]
    colors = palette[idx_lo] * (1.0 - frac) + palette[idx_lo + 1] * frac
    return colors  # (N, 3)


# ── per-pixel rendering (simple) ─────────────────────────────────────────

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


# ── per-pixel rendering (fancy) ─────────────────────────────────────────

def _render_pixel_fancy(uv, boid_uv, heading, size_uv, wing_open,
                        color_left, color_right, stroke_color, aa_blur):
    return render_boid_fancy(
        uv, boid_uv, heading, size_uv, wing_open,
        color_left, color_right, stroke_color, aa_blur,
    )


def _composite_pixel_fancy(uv, boid_uvs, headings, size_uv, wing_opens,
                           colors_left, colors_right, stroke_colors, bg_color, aa_blur):
    all_colors, all_masks = jax.vmap(
        _render_pixel_fancy,
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, None),
    )(uv, boid_uvs, headings, size_uv, wing_opens,
      colors_left, colors_right, stroke_colors, aa_blur)

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
def _render_frame_fancy(
    boid_uvs, render_headings, size_uv, wing_opens,
    colors_left, colors_right, stroke_colors,
    uv_grid, background_color, aa_blur,
):
    axes = (0, None, None, None, None, None, None, None, None, None)
    render_image = _vmap_image(_composite_pixel_fancy, axes)
    return render_image(
        uv_grid, boid_uvs, render_headings, size_uv, wing_opens,
        colors_left, colors_right, stroke_colors, background_color, aa_blur,
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

    # Nose is at local +Y, so after rotation by θ the nose points at
    # (-sin θ, cos θ) in UV space.  We need that to equal the UV velocity
    # direction (vx, -vy).  Solving: θ = arctan2(-vx, -vy).
    render_headings = jnp.arctan2(-velocities[:, 0], -velocities[:, 1])

    # Base colours
    base_fill = jnp.broadcast_to(
        jnp.array([0.706, 0.831, 1.0]),
        (n_agents, 3),
    )
    base_stroke = jnp.broadcast_to(
        jnp.array([0.525, 0.714, 0.965]),
        (n_agents, 3),
    )

    ctrl_colors = _controlled_color_gradient(positions, height)
    base_fill = base_fill.at[0].set(jnp.clip(ctrl_colors[0] * 1.1, 0, 1))
    base_stroke = base_stroke.at[0].set(ctrl_colors[0])

    # Move controlled agent (index 0) to the end so it renders on top
    order = jnp.concatenate([jnp.arange(1, n_agents), jnp.array([0])])
    boid_uvs = boid_uvs[order]
    render_headings = render_headings[order]
    base_fill = base_fill[order]
    base_stroke = base_stroke[order]

    if fancy_shape:
        wing_opens = _wing_openness(step_count, phase_offsets) if flap_wings \
            else jnp.ones(n_agents)
        wing_opens = wing_opens[order]
        frame = _render_frame_fancy(
            boid_uvs, render_headings, size_uv, wing_opens,
            base_fill, base_fill, base_stroke,
            uv_grid, background_color, aa_blur,
        )
    else:
        frame = _render_frame_simple(
            boid_uvs, render_headings, size_uv, base_fill,
            uv_grid, background_color, aa_blur,
        )

    return jnp.clip(frame, 0.0, 1.0)

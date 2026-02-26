"""Compose a full frame image from boid arrays using JAX SDF rendering.

Ports the visual style from the Processing boid sketch: two-tone split
wings, wing flapping, light-source shading, and outline stroke.
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
    return size / height * 2.0


# ── light & colour helpers ───────────────────────────────────────────────

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


def _compute_light_shading(positions, headings, light_pos, canvas_diag):
    """Per-boid light factor: angle-based left/right split + distance fade."""
    to_light = light_pos[None, :] - positions  # (N, 2)
    vel_dir = jnp.stack([jnp.sin(headings), jnp.cos(headings)], axis=-1)

    # Cross product sign → which side is lit
    cross = vel_dir[:, 0] * to_light[:, 1] - vel_dir[:, 1] * to_light[:, 0]
    angle_factor = jnp.tanh(cross / (canvas_diag * 0.1 + 1e-6))  # in [-1, 1]

    dist = jnp.sqrt(jnp.sum(to_light ** 2, axis=-1) + 1e-8)
    brightness = jnp.clip(1.0 - dist / (canvas_diag * 1.2), 0.15, 1.0)

    return angle_factor, brightness  # (N,), (N,)


# ── per-pixel rendering ─────────────────────────────────────────────────

def _render_pixel_for_boid(uv, boid_uv, heading, size_uv, wing_open,
                           color_left, color_right, stroke_color, aa_blur):
    color, mask = render_boid_at(
        uv, boid_uv, heading, size_uv, wing_open,
        color_left, color_right, stroke_color, aa_blur,
    )
    return color, mask


def _composite_pixel(uv, boid_uvs, headings, size_uv, wing_opens,
                     colors_left, colors_right, stroke_colors, bg_color, aa_blur):
    all_colors, all_masks = jax.vmap(
        _render_pixel_for_boid,
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, None),
    )(uv, boid_uvs, headings, size_uv, wing_opens,
      colors_left, colors_right, stroke_colors, aa_blur)

    def _blend(carry, x):
        c, m = x
        return carry * (1.0 - m) + c * m, None

    color, _ = jax.lax.scan(_blend, bg_color, (all_colors, all_masks))
    return color


# ── full frame (JIT-compiled) ────────────────────────────────────────────

@jax.jit
def render_frame(
    positions, headings, phase_offsets, step_count,
    uv_grid,
    width, height,
    agent_size, agent_color, controlled_color, background_color,
    aa_blur,
):
    """Render a complete (H, W, 3) float32 frame with Processing-style visuals."""
    boid_uvs = pixel_to_uv(positions, width, height)
    size_uv = pixel_size_to_uv(agent_size, height)

    n_agents = positions.shape[0]
    render_headings = -headings

    # Wing flapping
    wing_opens = _wing_openness(step_count, phase_offsets)

    # Moving light source (figure-eight pattern like Processing)
    t_light = step_count / 30.0
    light_x = width / 2.0 + 0.3 * width * jnp.sin(2.0 * t_light) * jnp.cos(2.0 * t_light)
    light_y = height / 2.0 + 0.3 * height * jnp.sin(t_light)
    light_pos = jnp.array([light_x, light_y])
    canvas_diag = jnp.sqrt(width ** 2 + height ** 2)

    angle_factor, brightness = _compute_light_shading(
        positions, headings, light_pos, canvas_diag,
    )

    # Base colours: soft blue for agents, gradient for controlled
    base_fill = jnp.broadcast_to(
        jnp.array([0.706, 0.831, 1.0]),  # rgb(180, 212, 255)
        (n_agents, 3),
    )
    base_stroke = jnp.broadcast_to(
        jnp.array([0.525, 0.714, 0.965]),  # rgb(134, 182, 246)
        (n_agents, 3),
    )

    # Controlled agent: warm gradient
    ctrl_colors = _controlled_color_gradient(positions, height)
    base_fill = base_fill.at[0].set(ctrl_colors[0] * 1.1)
    base_stroke = base_stroke.at[0].set(ctrl_colors[0])

    # Light-based left/right shading
    lighten = 0.08
    af = angle_factor[:, None]  # (N, 1)
    br = brightness[:, None]    # (N, 1)

    colors_left  = jnp.clip(base_fill * br + lighten * jnp.clip(af, 0, 1), 0, 1)
    colors_right = jnp.clip(base_fill * br + lighten * jnp.clip(-af, 0, 1), 0, 1)
    stroke_colors = jnp.clip(base_stroke * br, 0, 1)

    render_row = jax.vmap(
        _composite_pixel,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
    render_image = jax.vmap(
        render_row,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )

    frame = render_image(
        uv_grid, boid_uvs, render_headings, size_uv, wing_opens,
        colors_left, colors_right, stroke_colors, background_color, aa_blur,
    )
    return jnp.clip(frame, 0.0, 1.0)

"""Compose full-frame images from boid arrays using JAX renderers."""

from functools import partial

import jax
import jax.numpy as jnp

from flockworld.rendering.primitives import (
    render_boid_simple,
    render_boid_fancy,
    sd_js_boid_batch,
    smoothstep,
)


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

def _hsv_to_rgb(h, s, v):
    """Vectorized equivalent of the JS hsv(h, s, v) helper, normalized to 0..1."""
    i = jnp.floor(h * 6.0).astype(jnp.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    imod = jnp.mod(i, 6)

    r = jnp.select(
        [imod == 0, imod == 1, imod == 2, imod == 3, imod == 4],
        [v, q, p, p, t],
        default=v,
    )
    g = jnp.select(
        [imod == 0, imod == 1, imod == 2, imod == 3, imod == 4],
        [t, v, v, q, p],
        default=p,
    )
    b = jnp.select(
        [imod == 0, imod == 1, imod == 2, imod == 3, imod == 4],
        [p, p, t, v, v],
        default=q,
    )
    return jnp.stack([r, g, b], axis=-1)


@partial(jax.jit, static_argnames=("height", "width", "color_mode"))
def _render_frame_js_dart(
    positions, velocities, agent_color, background_color,
    height: int, width: int,
    max_speed, alpha, aa_blur,
    color_mode: str,
):
    """Render the original JS PIXI dart with a per-boid patch rasterizer."""
    image = jnp.broadcast_to(background_color, (height, width, 3)).copy()
    speed = jnp.sqrt(jnp.sum(velocities ** 2, axis=-1))
    if color_mode == "fixed":
        colors = jnp.broadcast_to(agent_color[None, :], (positions.shape[0], 3))
    else:
        hue = jnp.clip(speed / (max_speed * 2.0), 0.0, 1.0)
        colors = _hsv_to_rgb(hue, 1.0, 1.0)
    headings = jnp.arctan2(velocities[:, 1], velocities[:, 0])

    radius = 9
    offsets_1d = jnp.arange(-radius, radius + 1)
    off_x, off_y = jnp.meshgrid(offsets_1d, offsets_1d)
    offsets = jnp.stack(
        [off_x.reshape(-1), off_y.reshape(-1)], axis=-1,
    ).astype(jnp.float32)

    def draw_one(img, inputs):
        pos, heading, color = inputs
        cx = jnp.rint(pos[0]).astype(jnp.int32)
        cy = jnp.rint(pos[1]).astype(jnp.int32)
        xs = cx + offsets[:, 0].astype(jnp.int32)
        ys = cy + offsets[:, 1].astype(jnp.int32)

        pixel_pos = jnp.stack(
            [xs.astype(jnp.float32), ys.astype(jnp.float32)], axis=-1,
        )
        rel = pixel_pos - pos
        c = jnp.cos(-heading)
        s = jnp.sin(-heading)
        local = jnp.stack(
            [rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c],
            axis=-1,
        )

        d = sd_js_boid_batch(local)
        mask = (1.0 - smoothstep(0.0, aa_blur, d)) * alpha
        old = img.at[ys, xs].get(mode="fill", fill_value=0.0)
        new = old * (1.0 - mask[:, None]) + color * mask[:, None]
        img = img.at[ys, xs].set(new, mode="drop")
        return img, None

    image, _ = jax.lax.scan(draw_one, image, (positions, headings, colors))
    return jnp.clip(image, 0.0, 1.0)

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
    max_speed, boid_alpha,
    agent_shape, color_mode, flap_wings,
):
    """Render a complete (H, W, 3) float32 frame."""
    if agent_shape == "js":
        image_h, image_w = uv_grid.shape[:2]
        return _render_frame_js_dart(
            positions, velocities, agent_color, background_color,
            image_h, image_w,
            max_speed, boid_alpha, aa_blur,
            color_mode,
        )

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

    if agent_shape == "fancy":
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

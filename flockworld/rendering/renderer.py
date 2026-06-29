"""Compose full-frame images from boid arrays using JAX renderers."""

from functools import partial

import jax
import jax.numpy as jnp

from flockworld.rendering.primitives import (
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


@partial(
    jax.jit,
    static_argnames=("height", "width", "patch_radius", "color_mode", "border_width"),
)
def _render_frame_js_dart(
    positions, velocities, agent_color, boid_colors, background_image,
    height: int, width: int,
    agent_size, max_speed, alpha, aa_blur,
    patch_radius: int,
    color_mode: str,
    border_width: int,
    border_color,
):
    """Render the original JS PIXI dart with a batched patch rasterizer.

    The original scan-based implementation matched painter's-order alpha
    compositing exactly, but it forced one tiny scatter update per boid. This
    version builds every boid-patch pixel at once and reduces them into the
    target frame with large scatter-add operations, which gives XLA a much more
    GPU-friendly workload. Overlapping boids are composited with an
    order-independent alpha union.
    """
    speed = jnp.sqrt(jnp.sum(velocities ** 2, axis=-1))
    if color_mode == "fixed":
        colors = jnp.broadcast_to(agent_color[None, :], (positions.shape[0], 3))
    elif color_mode == "agent_id":
        colors = boid_colors  # (N, 3) precomputed per-boid identity colors
    else:  # "speed"
        hue = jnp.clip(speed / (max_speed * 2.0), 0.0, 1.0)
        colors = _hsv_to_rgb(hue, 1.0, 1.0)
    headings = jnp.arctan2(velocities[:, 1], velocities[:, 0])

    offsets_1d = jnp.arange(-patch_radius, patch_radius + 1)
    off_x, off_y = jnp.meshgrid(offsets_1d, offsets_1d)
    offsets = jnp.stack(
        [off_x.reshape(-1), off_y.reshape(-1)], axis=-1,
    ).astype(jnp.float32)

    centers = jnp.rint(positions).astype(jnp.int32)
    xs = centers[:, 0:1] + offsets[None, :, 0].astype(jnp.int32)
    ys = centers[:, 1:2] + offsets[None, :, 1].astype(jnp.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs_safe = jnp.clip(xs, 0, width - 1)
    ys_safe = jnp.clip(ys, 0, height - 1)

    pixel_pos = jnp.stack(
        [xs.astype(jnp.float32), ys.astype(jnp.float32)], axis=-1,
    )
    # Use the rounded center (not float position) so the dart's visual centroid
    # is locked to its integer pixel — keeps the focal agent stable inside the
    # integer-aligned partial crop.
    rel = pixel_pos - centers.astype(jnp.float32)[:, None, :]
    c = jnp.cos(-headings)[:, None]
    s = jnp.sin(-headings)[:, None]
    local = jnp.stack(
        [rel[..., 0] * c - rel[..., 1] * s, rel[..., 0] * s + rel[..., 1] * c],
        axis=-1,
    )

    d = sd_js_boid_batch(local.reshape((-1, 2)), agent_size).reshape(xs.shape)
    mask = (1.0 - smoothstep(0.0, aa_blur, d)) * alpha
    mask = jnp.where(valid, mask, 0.0)
    mask = jnp.clip(mask, 0.0, 1.0 - 1e-6)

    flat_idx = (ys_safe * width + xs_safe).reshape(-1)
    flat_mask = mask.reshape(-1)
    flat_rgb = (colors[:, None, :] * mask[..., None]).reshape((-1, 3))

    n_pixels = height * width
    alpha_sum = jnp.zeros((n_pixels,), dtype=jnp.float32).at[flat_idx].add(flat_mask)
    rgb_sum = jnp.zeros((n_pixels, 3), dtype=jnp.float32).at[flat_idx].add(flat_rgb)
    log_trans = jnp.log1p(-flat_mask)
    trans = jnp.exp(
        jnp.zeros((n_pixels,), dtype=jnp.float32).at[flat_idx].add(log_trans)
    )

    avg_rgb = rgb_sum / jnp.maximum(alpha_sum[:, None], 1e-6)
    bg_flat = background_image.reshape((height * width, 3))  # per-pixel bg (gradient or constant)
    image = (
        bg_flat * trans[:, None]
        + avg_rgb * (1.0 - trans[:, None])
    )
    image = jnp.clip(image.reshape((height, width, 3)), 0.0, 1.0)
    if border_width > 0:
        bw = min(border_width, min(height, width) // 2)
        ys = jnp.arange(height)[:, None]
        xs = jnp.arange(width)[None, :]
        on_border = (ys < bw) | (ys >= height - bw) | (xs < bw) | (xs >= width - bw)
        image = jnp.where(on_border[..., None], border_color[None, None, :], image)
    return image

def render_frame(
    positions, velocities,
    uv_grid,
    agent_size, agent_render_radius,
    agent_color, boid_colors, background_image,
    aa_blur,
    max_speed, boid_alpha,
    color_mode,
    border_width, border_color,
):
    """Render a complete (H, W, 3) float32 frame."""
    image_h, image_w = uv_grid.shape[:2]
    return _render_frame_js_dart(
        positions, velocities, agent_color, boid_colors, background_image,
        image_h, image_w,
        agent_size, max_speed, boid_alpha, aa_blur,
        agent_render_radius,
        color_mode,
        int(border_width), border_color,
    )

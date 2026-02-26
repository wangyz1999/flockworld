"""Low-level SDF and math primitives for JAX rendering."""

import jax.numpy as jnp


def smoothstep(edge0, edge1, x):
    """Hermite interpolation between 0 and 1."""
    t = jnp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rotate_2d(p, angle):
    """Rotate a 2D point by *angle* radians."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def sd_triangle(p, p0, p1, p2):
    """Signed distance from *p* to the triangle (p0, p1, p2)."""
    e0, e1, e2 = p1 - p0, p2 - p1, p0 - p2
    v0, v1, v2 = p - p0, p - p1, p - p2

    pq0 = v0 - e0 * jnp.clip(jnp.dot(v0, e0) / jnp.dot(e0, e0), 0.0, 1.0)
    pq1 = v1 - e1 * jnp.clip(jnp.dot(v1, e1) / jnp.dot(e1, e1), 0.0, 1.0)
    pq2 = v2 - e2 * jnp.clip(jnp.dot(v2, e2) / jnp.dot(e2, e2), 0.0, 1.0)

    s = jnp.sign(e0[0] * e2[1] - e0[1] * e2[0])

    d = jnp.min(jnp.array([
        jnp.dot(pq0, pq0), jnp.dot(pq1, pq1), jnp.dot(pq2, pq2)
    ]))

    dist = -jnp.sqrt(d) * jnp.sign(jnp.min(jnp.array([
        s * (v0[0] * e0[1] - v0[1] * e0[0]),
        s * (v1[0] * e1[1] - v1[1] * e1[0]),
        s * (v2[0] * e2[1] - v2[1] * e2[0]),
    ])))
    return dist


def render_boid_at(uv, position, heading, size, wing_open,
                   color_left, color_right, stroke_color, aa_blur):
    """Render a two-tone paper-plane boid with wing flapping and outline.

    The shape mirrors the Processing sketch: nose at top, two swept wings
    whose spread is controlled by *wing_open*, and a V-notch tail.

    Returns (blended_color, mask) for compositing.
    """
    local_p = rotate_2d(uv - position, -heading)

    # Geometry matching the Processing boid (in local space, nose = +Y)
    nose  = jnp.array([0.0, size * 2.0])
    lwing = jnp.array([-size * wing_open, -size * 2.0])
    rwing = jnp.array([ size * wing_open, -size * 2.0])
    notch = jnp.array([0.0, -size * 0.472])

    # Left half: nose → lwing → notch
    d_left = sd_triangle(local_p, nose, lwing, notch)
    # Right half: nose → notch → rwing
    d_right = sd_triangle(local_p, nose, notch, rwing)

    mask_left  = 1.0 - smoothstep(0.0, aa_blur, d_left)
    mask_right = 1.0 - smoothstep(0.0, aa_blur, d_right)

    # Outline: thin ring around the full shape
    d_full = jnp.minimum(d_left, d_right)
    stroke_width = aa_blur * 3.0
    outline_mask = (1.0 - smoothstep(0.0, aa_blur, d_full)) * smoothstep(-stroke_width, -stroke_width * 0.3, d_full)

    # Composite: left fill, right fill, then outline on top
    fill_color = color_left * mask_left + color_right * mask_right
    total_mask = jnp.clip(mask_left + mask_right, 0.0, 1.0)

    color = fill_color * (1.0 - outline_mask) + stroke_color * outline_mask

    return color, total_mask

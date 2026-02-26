"""Low-level SDF and math primitives for JAX rendering."""

import jax.numpy as jnp


def smoothstep(edge0: float, edge1: float, x):
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


def triangle_vertices(heading, size):
    """Return the three local-space vertices of an isoceles boid triangle.

    The tip points in the *heading* direction.  Returns (p0, p1, p2) each
    shape (2,).
    """
    tip = rotate_2d(jnp.array([0.0, size]), heading)
    left = rotate_2d(jnp.array([-size * 0.5, -size * 0.5]), heading)
    right = rotate_2d(jnp.array([size * 0.5, -size * 0.5]), heading)
    return tip, left, right


def render_triangle_at(uv, position, heading, size, color, aa_blur=0.005):
    """Compute the color contribution of a single boid triangle at *uv*.

    *uv* is a single 2D point in pixel-normalised coordinates.
    Returns (rgb, mask) where mask ∈ [0, 1].
    """
    local_uv = uv - position
    tip, left, right = triangle_vertices(heading, size)
    d = sd_triangle(local_uv, tip, left, right)
    mask = 1.0 - smoothstep(0.0, aa_blur, d)
    return color, mask

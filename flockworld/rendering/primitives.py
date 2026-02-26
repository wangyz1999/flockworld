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


def sd_paper_plane(p, heading, size):
    """Signed distance for a paper-plane shape (outer hull minus tail notch).

    The nose points in the *heading* direction.  The shape is an elongated
    triangle with a V-shaped cutout at the tail, giving a folded-paper look.

    Local geometry (before rotation, nose pointing +Y):
        nose:       (0, size)
        left wing:  (-size*0.45, -size*0.6)
        right wing: ( size*0.45, -size*0.6)
        notch tip:  (0, -size*0.15)   (cuts into the tail)
    """
    local_p = rotate_2d(p, -heading)

    # Outer hull triangle
    nose  = jnp.array([0.0,          size])
    lwing = jnp.array([-size * 0.45, -size * 0.6])
    rwing = jnp.array([ size * 0.45, -size * 0.6])
    d_outer = sd_triangle(local_p, nose, lwing, rwing)

    # Tail notch triangle (subtracted)
    notch_tip = jnp.array([0.0, -size * 0.15])
    d_notch = sd_triangle(local_p, notch_tip, rwing, lwing)

    # Boolean subtraction: outer AND NOT notch
    return jnp.maximum(d_outer, -d_notch)


def render_boid_at(uv, position, heading, size, color, aa_blur=0.005):
    """Compute the color contribution of a single paper-plane boid at *uv*.

    Returns (rgb, mask) where mask is in [0, 1].
    """
    local_uv = uv - position
    d = sd_paper_plane(local_uv, heading, size)
    mask = 1.0 - smoothstep(0.0, aa_blur, d)
    return color, mask

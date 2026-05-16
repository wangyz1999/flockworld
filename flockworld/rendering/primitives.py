"""Low-level SDF and math primitives for JAX rendering."""

import jax.numpy as jnp


def smoothstep(edge0, edge1, x):
    """Hermite interpolation between 0 and 1."""
    t = jnp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sd_triangle_batch(p, p0, p1, p2):
    """Signed distance from points ``p`` to triangle ``(p0, p1, p2)``."""
    e0, e1, e2 = p1 - p0, p2 - p1, p0 - p2
    v0, v1, v2 = p - p0, p - p1, p - p2

    pq0 = v0 - e0 * jnp.clip(
        jnp.sum(v0 * e0, axis=-1, keepdims=True) / jnp.dot(e0, e0), 0.0, 1.0,
    )
    pq1 = v1 - e1 * jnp.clip(
        jnp.sum(v1 * e1, axis=-1, keepdims=True) / jnp.dot(e1, e1), 0.0, 1.0,
    )
    pq2 = v2 - e2 * jnp.clip(
        jnp.sum(v2 * e2, axis=-1, keepdims=True) / jnp.dot(e2, e2), 0.0, 1.0,
    )

    s = jnp.sign(e0[0] * e2[1] - e0[1] * e2[0])
    d = jnp.minimum(
        jnp.minimum(jnp.sum(pq0 * pq0, axis=-1), jnp.sum(pq1 * pq1, axis=-1)),
        jnp.sum(pq2 * pq2, axis=-1),
    )
    edge_sign = jnp.minimum(
        jnp.minimum(
            s * (v0[..., 0] * e0[1] - v0[..., 1] * e0[0]),
            s * (v1[..., 0] * e1[1] - v1[..., 1] * e1[0]),
        ),
        s * (v2[..., 0] * e2[1] - v2[..., 1] * e2[0]),
    )
    return -jnp.sqrt(d) * jnp.sign(edge_sign)


def sd_js_boid_batch(p, size):
    """Signed distance for the JS dart shape in pixel units.

    ``size=10`` matches the original source points; other values scale the
    shape uniformly.
    """
    scale = jnp.maximum(size, 1e-6) / 10.0
    p = p / scale
    nose = jnp.array([6.0, 0.0])
    tail_top = jnp.array([-6.0, -4.0])
    notch = jnp.array([-4.0, 0.0])
    tail_bottom = jnp.array([-6.0, 4.0])
    d_top = sd_triangle_batch(p, nose, tail_top, notch)
    d_bottom = sd_triangle_batch(p, nose, notch, tail_bottom)
    return jnp.minimum(d_top, d_bottom) * scale

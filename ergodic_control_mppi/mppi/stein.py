"""Analytic Gaussian-mixture scores and RBF Stein interactions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ergodic_control_mppi.parameters import GMMParams, SteinParams

# Design constant: dimensionless floor gating the over-coverage field off as the
# total relative excess vanishes. Small against the O(1) excesses of normal
# operation, so it only acts in the (near-)fully-under-covered regime.
ACTIVITY_FLOOR = 1e-3


def component_logpdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Return component log densities with shape ``(..., M)``."""
    delta = position[..., None, :2] - params.means
    quadratic = jnp.einsum("...mi,mij,...mj->...m", delta, params.covariance_inverse, delta)
    return params.log_normalizers - 0.5 * quadratic


def logpdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the log Gaussian-mixture density at ``(..., 2)`` positions."""
    return logsumexp(params.log_weights + component_logpdf(position, params), axis=-1)


def pdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the Gaussian-mixture density at ``(..., 2)`` positions."""
    return jnp.exp(logpdf(position, params))


def score_pdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the analytic score ``grad(log p)`` with shape ``(..., 2)``."""
    delta = position[..., None, :2] - params.means
    component_scores = -jnp.einsum("mij,...mj->...mi", params.covariance_inverse, delta)
    logits = params.log_weights + component_logpdf(position, params)
    responsibilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("...m,...mi->...i", responsibilities, component_scores)


def kernel(x: jax.Array, y: jax.Array, bandwidth: jax.Array) -> jax.Array:
    """Evaluate ``exp(-||x-y||^2 / bandwidth)``."""
    delta = x - y
    return jnp.exp(-jnp.sum(delta * delta, axis=-1) / bandwidth)


def kernel_gradient(x: jax.Array, y: jax.Array, bandwidth: jax.Array) -> jax.Array:
    """Return the analytic RBF gradient with respect to ``x``."""
    return (-2.0 / bandwidth) * (x - y) * kernel(x, y, bandwidth)[..., None]


def stein_gradient(
    positions: jax.Array,
    particles: jax.Array,
    gmm: GMMParams,
    stein: SteinParams,
    bandwidth: jax.Array,
) -> jax.Array:
    """Compute target-attractive Stein flow for each queried position.

    Args:
        positions: Query positions with shape ``(Q, 2)``.
        particles: Target particles with shape ``(P, 2)``.
        gmm: Precomputed target-density terms.
        stein: Rotation and interaction parameters.
        bandwidth: Positive RBF bandwidth.

    Returns:
        Mean Stein flow with shape ``(Q, 2)``.
    """
    particle = particles[None, :, :]
    query = positions[:, None, :]
    values = kernel(particle, query, bandwidth)[..., None]
    flow = values * (score_pdf(particles, gmm) @ stein.rotation.T)[None, :, :]
    flow += kernel_gradient(particle, query, bandwidth) @ stein.rotation.T
    return jnp.mean(flow, axis=1)


def stein_repulsion(
    positions: jax.Array,
    particles: jax.Array,
    weights: jax.Array,
    stein: SteinParams,
    bandwidth: jax.Array,
) -> jax.Array:
    """Weighted-mean RBF repulsion pushing each query away from ``particles``.

    Used to repel the plan from a bounded memory of recently visited positions
    (the curl-augmented "fading memory" coverage feedback). Same curl rotation
    as ``stein_gradient``; no target-density term. Per-particle ``weights``
    carry the recency decay and density-ratio; the mean is normalized by their
    sum so the direction is well-defined regardless of the weighting.

    Args:
        positions: Query positions with shape ``(Q, 2)``.
        particles: Memory positions with shape ``(P, 2)``.
        weights: Non-negative per-particle weights with shape ``(P,)``.
        stein: Rotation parameters.
        bandwidth: Positive RBF bandwidth.

    Returns:
        Weighted-mean repulsion with shape ``(Q, 2)``.
    """
    repulsion = kernel_gradient(particles[None, :, :], positions[:, None, :], bandwidth)
    rotated = repulsion @ stein.rotation.T
    return jnp.sum(rotated * weights[None, :, None], axis=1) / jnp.maximum(jnp.sum(weights), 1e-12)


def smoothed(gmm: GMMParams, bandwidth: jax.Array) -> GMMParams:
    """Convolve the target with the normalized kernel ``kappa_h / (pi h)``.

    That kernel is ``N(0, (h/2) I)``, so the convolution is again a Gaussian
    mixture with every covariance inflated by ``(h/2) I`` -- closed form, no
    quadrature. Comparing occupancy at scale ``h`` against ``smoothed(gmm, h)``
    instead of the raw target removes the bandwidth-dependent bias of comparing
    a smoothed density to an unsmoothed one.
    """
    covariance = gmm.covariance + 0.5 * bandwidth * jnp.eye(2)
    return GMMParams(
        means=gmm.means,
        covariance=covariance,
        covariance_inverse=jnp.linalg.inv(covariance),
        log_weights=gmm.log_weights,
        log_normalizers=-0.5
        * (2 * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(covariance)[1]),
    )


def multiscale_memory_flow(
    positions: jax.Array,
    memory: jax.Array,
    recency: jax.Array,
    gmm: GMMParams,
    stein: SteinParams,
    density_floor: jax.Array,
) -> jax.Array:
    """Normalized multiscale over-coverage repulsion from the memory buffer.

    At each of ``memory_scales`` log-spaced bandwidths between
    ``fine_bandwidth`` and ``coarse_bandwidth``, blend two *separately
    normalized fields* by ``memory_balance`` in ``[0, 1]``: the fading trail
    (0), weighted by ``omega_i``, and the relative over-coverage (1), weighted
    by ``omega_i e_i`` with
    ``e_i = [o_h(m_i) - p_h(m_i)]_+ / (p_h(m_i) + density_floor)``. Because
    :func:`stein_repulsion` divides by the weight sum, blending the two fields
    makes ``memory_balance`` a genuine interpolation coefficient -- unlike
    blending the weights themselves, whose effective mixing coefficient drifts
    with time, bandwidth and the instantaneous total surplus.

    The trail term is a true probability-weighted average; the excess term is
    deliberately a *sub*-probability one, its deficit being the activity gate
    described inline below.

    Each scale is divided by ``max_r |grad kappa_h| = sqrt(2/h) e^{-1/2}`` so the
    scales are commensurate and carry no separate gains.

    Args:
        positions: Query positions with shape ``(Q, 2)``.
        memory: Executed-position buffer with shape ``(P, 2)``.
        recency: Non-negative fading weights with shape ``(P,)``.
        gmm: Target density terms.
        stein: Bandwidth endpoints, scale count, and balance.
        density_floor: Positive density scale stabilizing the relative excess.

    Returns:
        Scale-averaged repulsion with shape ``(Q, 2)``.
    """
    scales = jnp.geomspace(
        stein.fine_bandwidth, stein.coarse_bandwidth, stein.memory_scales
    )
    total = jnp.zeros_like(positions)
    for index in range(stein.memory_scales):
        bandwidth = scales[index]
        occupancy = (kernel(memory[:, None, :], memory[None, :, :], bandwidth) @ recency) / (
            jnp.sum(recency) * jnp.pi * bandwidth
        )
        target = pdf(memory, smoothed(gmm, bandwidth))
        excess = jnp.maximum(occupancy - target, 0.0) / (target + density_floor)
        # Activity gate. The normalized excess field is scale-invariant in the excess
        # (multiplying every e_i by c leaves it unchanged), so it does NOT tend to zero
        # as over-coverage disappears -- it would jump to zero only at the weight-sum
        # floor. Multiplying by S/(S+eps_S), with S the recency-mean excess, makes the
        # excess component genuinely continuous at S=0 while acting as the identity
        # whenever S >> eps_S, i.e. in normal operation.
        activity = jnp.sum(recency * excess) / jnp.sum(recency)
        field = (1.0 - stein.memory_balance) * stein_repulsion(
            positions, memory, recency, stein, bandwidth
        ) + stein.memory_balance * (activity / (activity + ACTIVITY_FLOOR)) * stein_repulsion(
            positions, memory, recency * excess, stein, bandwidth
        )
        total += jnp.sqrt(0.5 * jnp.e * bandwidth) * field
    return total / stein.memory_scales

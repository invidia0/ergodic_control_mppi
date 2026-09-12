"""
The reference field: analytic scores, KDE repulsion, and the service gate.
"""

from dataclasses import replace
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ergodic_control_mppi.parameters import GMMParams, FieldParams

# Design constant: dimensionless floor gating the over-coverage field off as the
# total relative excess vanishes. Small against the O(1) excesses of normal
# operation, so it only acts in the (near-)fully-under-covered regime.
ACTIVITY_FLOOR = 1e-3


def component_logpdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Return component log densities with shape ``(..., M)``."""
    delta = position[..., None, :] - params.means
    quadratic = jnp.einsum("...mi,mij,...mj->...m", delta, params.covariance_inverse, delta)
    return params.log_normalizers - 0.5 * quadratic


def logpdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the log Gaussian-mixture density at ``(..., d)`` positions."""
    return logsumexp(params.log_weights + component_logpdf(position, params), axis=-1)


def pdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the Gaussian-mixture density at ``(..., d)`` positions."""
    return jnp.exp(logpdf(position, params))


def score_pdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the analytic score ``grad(log p)`` with shape ``(..., d)``."""
    delta = position[..., None, :] - params.means
    component_scores = -jnp.einsum("mij,...mj->...mi", params.covariance_inverse, delta)
    logits = params.log_weights + component_logpdf(position, params)
    responsibilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("...m,...mi->...i", responsibilities, component_scores)


def responsibility_gaps(params: GMMParams) -> jax.Array:
    """Log-odds margin each component holds at its own centre, shape ``(J,)``."""
    logits = params.log_weights + component_logpdf(params.means, params)
    own = jnp.diagonal(logits)
    rival = jnp.max(jnp.where(jnp.eye(logits.shape[0], dtype=bool), -jnp.inf, logits), axis=-1)
    return own - rival


def kernel(x: jax.Array, y: jax.Array, bandwidth: jax.Array) -> jax.Array:
    """Evaluate ``exp(-||x-y||^2 / bandwidth)``."""
    delta = x - y
    return jnp.exp(-jnp.sum(delta * delta, axis=-1) / bandwidth)


def kernel_gradient(x: jax.Array, y: jax.Array, bandwidth: jax.Array) -> jax.Array:
    """Return the analytic RBF gradient with respect to ``x``."""
    return (-2.0 / bandwidth) * (x - y) * kernel(x, y, bandwidth)[..., None]


def kde_repulsion(
    positions: jax.Array,
    particles: jax.Array,
    weights: jax.Array,
    bandwidth: jax.Array,
) -> jax.Array:
    """
    Weighted-mean RBF repulsion pushing each query away from ``particles``.
    
    Args:
        positions: Query positions with shape ``(Q, d)``.
        particles: Source positions with shape ``(P, d)``.
        weights: Non-negative per-particle weights with shape ``(P,)``.
        bandwidth: Positive RBF bandwidth.
    Returns:
        Weighted-mean repulsion with shape ``(Q, d)``.
    """
    repulsion = kernel_gradient(particles[None, :, :], positions[:, None, :], bandwidth)
    return jnp.sum(repulsion * weights[None, :, None], axis=1) / jnp.maximum(
        jnp.sum(weights), 1e-12
    )


def kde_potential(
    positions: jax.Array,
    particles: jax.Array,
    weights: jax.Array,
    bandwidth: jax.Array,
) -> jax.Array:
    """The scalar whose negative gradient is :func:`kde_repulsion`, shape ``(Q,)``."""
    values = kernel(particles[None, :, :], positions[:, None, :], bandwidth)
    return jnp.sum(values * weights[None, :], axis=1) / jnp.maximum(jnp.sum(weights), 1e-12)


def smoothed(gmm: GMMParams, bandwidth: jax.Array) -> GMMParams:
    """Convolve the target with the normalized kernel ``kappa_h / (pi h)``."""
    covariance = gmm.covariance + 0.5 * bandwidth * jnp.eye(gmm.means.shape[-1])
    d = gmm.means.shape[-1]
    return GMMParams(
        means=gmm.means,
        covariance=covariance,
        covariance_inverse=jnp.linalg.inv(covariance),
        log_weights=gmm.log_weights,
        log_normalizers=-0.5
        * (d * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(covariance)[1]),
    )


def memory_weights(
    memory: jax.Array,
    recency: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    density_floor: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the ``(trail, excess, gate)`` weightings of the memory term."""
    bandwidth = field.fine_bandwidth
    occupancy = (kernel(memory[:, None, :], memory[None, :, :], bandwidth) @ recency) / (
        jnp.sum(recency) * (jnp.pi * bandwidth) ** (0.5 * memory.shape[-1])
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
    return recency, recency * excess, activity / (activity + ACTIVITY_FLOOR)


def memory_repulsion(
    positions: jax.Array,
    memory: jax.Array,
    recency: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    density_floor: jax.Array,
) -> jax.Array:
    """Normalized over-coverage repulsion from the memory buffer.

    Args:
        positions: Query positions with shape ``(Q, d)``.
        memory: Executed-position buffer with shape ``(P, d)``.
        recency: Non-negative fading weights with shape ``(P,)``.
        gmm: Target density terms.
        field: Bandwidth and balance.
        density_floor: Positive density scale stabilizing the relative excess.

    Returns:
        Gauge-normalized repulsion with shape ``(Q, d)``.
    """
    bandwidth = field.fine_bandwidth
    trail, excess, gate = memory_weights(memory, recency, gmm, field, density_floor)
    blended = (1.0 - field.memory_balance) * kde_repulsion(
        positions, memory, trail, bandwidth
    ) + field.memory_balance * gate * kde_repulsion(positions, memory, excess, bandwidth)
    return jnp.sqrt(0.5 * jnp.e * bandwidth) * blended


memory_flow = memory_repulsion


def responsibilities(position: jax.Array, gmm: GMMParams) -> jax.Array:
    """GMM responsibilities at one position, shape ``(J,)``."""
    return jax.nn.softmax(component_logpdf(position, gmm))


def service_ratio(memory: jax.Array, recency: jax.Array, gmm: GMMParams) -> jax.Array:
    """
    Return how well-served the mode the vehicle currently sits in has been.
    
    Args:
            memory: Executed-position buffer with shape ``(P, 2)``; oldest first.
            recency: Non-negative fading weights with shape ``(P,)``.
            gmm: Target density terms.
    Returns:
            Scalar service ratio at the newest buffer entry.
    """
    weights = jax.nn.softmax(
        jax.vmap(component_logpdf, in_axes=(0, None))(memory, gmm), axis=-1
    )                                                    # (P, J)
    mass = recency @ weights                             # (J,)
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    target = jnp.exp(gmm.log_weights)
    return jnp.sum(weights[-1] * share / jnp.maximum(target, 1e-12))


def service_ratio_from_mass(mass: jax.Array, position: jax.Array,
                            gmm: GMMParams) -> jax.Array:
    """Service ratio from an exponentially-weighted mass accumulator."""
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    weights = jnp.exp(gmm.log_weights)
    return jnp.sum(responsibilities(position, gmm) * share / jnp.maximum(weights, 1e-12))


def scheduled_speed(
    queries: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    sigma: jax.Array,
) -> jax.Array:
    """Scalar speed schedule ``v(z, s)``, shape ``(Q,)``."""
    peak = jnp.max(pdf(gmm.means, gmm))
    share = jnp.clip(pdf(queries, gmm) / jnp.maximum(peak, 1e-12), 0.0, 1.0)
    floor = field.service_floor
    served = jnp.maximum(sigma - 1.0, 0.0)
    release = jnp.where(floor > 0, served / (served + jnp.maximum(floor, 1e-12)), 0.0)
    hold = 1.0 - release
    return field.reference_speed * jnp.power(
        field.transit_speedup, 1.0 - share * hold
    ) / jnp.power(field.dwell_slowdown, share * hold)


def deficit_weighted(mass: jax.Array, gmm: GMMParams, floor: jax.Array) -> GMMParams:
    """Re-weight the mixture toward components the path has under-served."""
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    weights = jnp.exp(gmm.log_weights)
    deficit = jnp.maximum(1.0 - share / jnp.maximum(weights, 1e-12), 0.0)
    bent = weights * (floor + deficit)
    return replace(gmm, log_weights=jnp.log(bent / jnp.maximum(jnp.sum(bent), 1e-12)))


def per_mode_weighted(mass: jax.Array, gmm: GMMParams, floor: jax.Array,
                      release_ratio: float) -> GMMParams:
    """
    `deficit_weighted` plus a demotion that releases every mode at the same over-service.
    """
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    weights = jnp.exp(gmm.log_weights)
    deficit = jnp.maximum(1.0 - share / jnp.maximum(weights, 1e-12), 0.0)
    bent = weights * (floor + deficit)
    ratio = share / jnp.maximum(weights, 1e-12)
    # release_ratio -> 1 demands an unbounded penalty (release exactly at fair share); the
    # config floor keeps the divisor away from zero.
    penalty = responsibility_gaps(gmm) / max(release_ratio - 1.0, 1e-6)
    # A component with no rival has an infinite margin, and no penalty can release it --
    # there is nowhere to go. Demoting it is meaningless, and `inf * 0` at exactly fair
    # share is NaN, which would take the whole field down on a unimodal target.
    penalty = jnp.where(jnp.isfinite(penalty), penalty, 0.0)
    log_bent = jnp.log(jnp.maximum(bent, 1e-30)) - penalty * jnp.maximum(ratio - 1.0, 0.0)
    return replace(gmm, log_weights=log_bent - logsumexp(log_bent))


def attraction_target(gmm: GMMParams, field: FieldParams,
                      service_mass: jax.Array | None) -> GMMParams:
    """The mixture the score attraction is taken of."""
    if service_mass is None:
        return gmm
    ceiling = jnp.maximum(field.deficit_ceiling, 1e-12)
    bent = (per_mode_weighted(service_mass, gmm, ceiling, field.release_ratio)
            if field.release_ratio > 0
            else deficit_weighted(service_mass, gmm, ceiling))
    return jax.tree.map(
        lambda b, true: jnp.where(field.deficit_ceiling > 0, b, true), bent, gmm
    )


def potential(
    positions: jax.Array,
    memory: jax.Array,
    recency: jax.Array,
    plan: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    density_floor: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """
    The scalar ``Phi`` whose gradient is the pre-gauge reference field, shape ``(Q,)``.
    """
    bandwidth = field.fine_bandwidth
    gauge = jnp.sqrt(0.5 * jnp.e * bandwidth)
    trail, excess, gate = memory_weights(memory, recency, gmm, field, density_floor)
    phi = logpdf(positions, attraction_target(gmm, field, service_mass))
    phi -= field.memory_gain * gauge * (
        (1.0 - field.memory_balance) * kde_potential(positions, memory, trail, bandwidth)
        + field.memory_balance * gate * kde_potential(positions, memory, excess, bandwidth)
    )
    phi -= field.plan_gain * gauge * kde_potential(
        positions, plan, jnp.ones((plan.shape[0],), dtype=positions.dtype), bandwidth
    )
    return phi

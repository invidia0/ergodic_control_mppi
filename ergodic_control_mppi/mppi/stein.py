"""Analytic Gaussian-mixture scores and RBF Stein interactions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ergodic_control_mppi.parameters import GMMParams, SteinParams


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
        positions: Query positions with shape ``(T, 2)``.
        particles: Target particles with shape ``(P, 2)``.
        gmm: Precomputed target-density terms.
        stein: Rotation and interaction parameters.
        bandwidth: Positive RBF bandwidth.

    Returns:
        Mean Stein flow with shape ``(T, 2)``.
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
    stein: SteinParams,
) -> jax.Array:
    """Compute pure rotated kernel-gradient interaction with cross particles."""
    gradients = kernel_gradient(
        particles[None, :, :], positions[:, None, :], stein.cross_bandwidth
    )
    return jnp.mean(gradients @ stein.rotation.T, axis=1)

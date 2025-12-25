from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SteinParams:
    # GMM in R^2
    means: jnp.ndarray # (M,2)
    cov_inv: jnp.ndarray # (M,2,2)
    log_weights: jnp.ndarray # (M,)
    log_norm: jnp.ndarray # (M,)

    # drift matrices
    D: jnp.ndarray # (2,2)
    S: jnp.ndarray # (2,2)
    gamma: float

    # kernel
    h: float

    # overall weight (set to 0.0 to disable)
    weight: float
    weight_pdf: float # maybe this one can go somewhere else


def component_logpdf(x, p: SteinParams):
    dz = x[:2] - p.means
    quad = jnp.einsum("ki,kij,kj->k", dz, p.cov_inv, dz)
    return p.log_norm - 0.5 * quad


def logpdf(x, p: SteinParams):
    log_comps = p.log_weights + component_logpdf(x, p)
    return logsumexp(log_comps)


def score_pdf(x, p: SteinParams):
    return jax.grad(logpdf, argnums=0)(x, p)


def drift(x, p: SteinParams):
    s = score_pdf(x, p)
    r = p.gamma * (p.S @ s.T).T
    return p.D @ s + r


def pdf(x, p: SteinParams):
    return jnp.exp(logpdf(x, p))


def kernel(x, y, p: SteinParams):
    diff = x - y
    return jnp.exp(-jnp.dot(diff, diff) / p.h)


def stein_grad_unit(x1, x2, p: SteinParams):
    ker_grad = jax.grad(kernel, argnums=0)(x2, x1, p)
    val = kernel(x2, x1, p) * drift(x2, p) + ker_grad
    return val


def stein_grad_state(x, x_traj, p: SteinParams):
    """
    Here we compute the Stein gradient for a single particle x against all particles in x_traj
    and then average the results.
    """
    stein_grads = jax.vmap(stein_grad_unit, in_axes=(None, 0, None))(x, x_traj, p)
    return jnp.mean(stein_grads, axis=0)


def stein_grad_traj(x_traj, p: SteinParams):
    """
    Compute the Stein gradient for each particle in x_traj against all particles in x_traj.
    """
    stein_grads = jax.vmap(stein_grad_state, in_axes=(0, None, None))(x_traj, x_traj, p)
    return stein_grads
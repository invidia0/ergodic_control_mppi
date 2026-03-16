import dataclasses
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

    # preconditioning matrix  A = R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
    # θ ∈ [0°, 90°): mixing angle between score-descent (0°) and pure rotation (90°)
    A: jnp.ndarray # (2,2)

    # kernel
    h: float
    h_cross: float  # cross-term bandwidth (fixed); sqrt(h_cross) ≈ separation radius in metres

    # overall weight (set to 0.0 to disable)
    weight: float
    weight_pdf: float # maybe this one can go somewhere else
    cross_alpha: float  # relative weight of inter-robot h_cross term (0 = single-robot)


def component_logpdf(x, p: SteinParams):
    dz = x[:2] - p.means
    quad = jnp.einsum("ki,kij,kj->k", dz, p.cov_inv, dz)
    return p.log_norm - 0.5 * quad


def logpdf(x, p: SteinParams):
    log_comps = p.log_weights + component_logpdf(x, p)
    return logsumexp(log_comps)


def score_pdf(x, p: SteinParams):
    return jax.grad(logpdf, argnums=0)(x, p)


def pdf(x, p: SteinParams):
    return jnp.exp(logpdf(x, p))


def kernel(x, y, p: SteinParams):
    diff = x - y
    return jnp.exp(-jnp.dot(diff, diff) / p.h)


def stein_grad_unit(x1, x2, p: SteinParams):
    """
    h_A(x1) contribution from particle x2:
      k(x2,x1) * A @ score(x2)  +  A @ ∇_{x2} k(x2,x1)
    where ker_grad = ∇_{x2} k(x2,x1).
    A = R(θ) is the precomputed rotation matrix stored in p.
    """
    # gradient w.r.t. the first argument of kernel, evaluated at (x2, x1)
    ker_grad = jax.grad(kernel, argnums=0)(x2, x1, p)

    val = kernel(x2, x1, p) * (p.A @ score_pdf(x2, p)) + (p.A @ ker_grad)
    return val


def stein_grad_state(x, x_traj, p: SteinParams):
    """
    Here we compute the Stein gradient for a single particle x against all particles in x_traj
    and then average the results.
    """
    stein_grads = jax.vmap(stein_grad_unit, in_axes=(None, 0, None))(x, x_traj, p)
    return jnp.mean(stein_grads, axis=0)

def stein_repulsion_unit(x1, x2, p: SteinParams):
    """
    Pure inter-robot repulsion: only the kernel gradient term.
      A @ ∇_{x2} k(x2, x1)
    Drops the score term k(x2,x1) * A @ score(x2) entirely,
    removing the redundant mode-attraction bias from cross-robot particles.
    """
    ker_grad = jax.grad(kernel, argnums=0)(x2, x1, p)
    return p.A @ ker_grad


def stein_repulsion_state(x, x_traj, p: SteinParams):
    """
    Average pure-repulsion contribution over all particles in x_traj.
    Uses h_cross (fixed separation bandwidth) instead of h (median-adapted self bandwidth).
    """
    p_cross = dataclasses.replace(p, h=p.h_cross)
    repulsions = jax.vmap(stein_repulsion_unit, in_axes=(None, 0, None))(x, x_traj, p_cross)
    return jnp.mean(repulsions, axis=0)


def stein_grad_traj(x_traj, p: SteinParams):
    """
    Compute the Stein gradient for each particle in x_traj against all particles in x_traj.
    """
    stein_grads = jax.vmap(stein_grad_state, in_axes=(0, None, None))(x_traj, x_traj, p)
    return stein_grads
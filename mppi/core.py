import dataclasses
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from models import double_integrator as model

from jax.scipy.special import logsumexp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObstacleParams:
    xyr: jnp.ndarray  # (num_obstacles, 3) (x, y, r)
    weight: float = field(metadata={"static": True})
    safe_distance: float = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Params:
    K: int = field(metadata={"static": True})
    T: int = field(metadata={"static": True})
    dim_u: int = field(metadata={"static": True})
    dim_x: int = field(metadata={"static": True})
    num_robots: int = field(metadata={"static": True})

    means: jnp.ndarray # (M,2)
    cov_inv: jnp.ndarray # (M,2,2)
    log_weights: jnp.ndarray # (M,)
    log_norm: jnp.ndarray # (M,)

    gamma_uld: float
    w_uld: float
    w_mag: float
    w_rate: float
    alpha_irrev: float  # single derived quantity for core logic

    lam: float
    alpha: float
    gamma: float

    use_nominal: jnp.ndarray  # (K,) bool - dynamic field

    Sigma: jnp.ndarray # (dim_u, dim_u)
    Sigma_inv: jnp.ndarray # (dim_u, dim_u)

    map_x_limits: jnp.ndarray   # (2,)
    map_y_limits: jnp.ndarray   # (2,)
    resolution: float
    oom_cost: float

    delta_t: float

    model_params: model.DoubleIntegratorParams

    obstacle_params: ObstacleParams

    ess_target: float   # target ESS fraction, e.g. 0.3
    lam_min: float      # lower clamp for lam
    lam_max: float      # upper clamp for lam

    seed: int

    steps: int


def sample_epsilon(key: jax.Array, params: Params) -> tuple[jnp.ndarray, jax.Array]:
    """
    eps ~ N(0, Sigma) with shape (K, T, dim_u)
    """
    key, sub = jax.random.split(key)
    eps = jax.random.multivariate_normal(
        sub,
        mean=jnp.zeros((params.dim_u,), dtype=jnp.float32),
        cov=params.Sigma,
        shape=(params.K, params.T),
        dtype=jnp.float32,
    )
    return eps, key


def _is_collided(x: jnp.ndarray, obs_params: ObstacleParams) -> bool:
    """
    Check if position x is in collision with any obstacle.
    """
    def check_obs(o):
        dist = jnp.linalg.norm(x - o[:2])
        return dist <= o[2] + obs_params.safe_distance

    collided = jax.vmap(check_obs)(obs_params.xyr)
    return jnp.any(collided)


def component_logpdf(x, p: Params):
    dz = x[:2] - p.means
    quad = jnp.einsum("ki,kij,kj->k", dz, p.cov_inv, dz)
    return p.log_norm - 0.5 * quad


def logpdf(x, p: Params):
    log_comps = p.log_weights + component_logpdf(x, p)
    return logsumexp(log_comps)


def score_pdf(x, p: Params):
    return jax.grad(logpdf, argnums=0)(x, p)


def _state_velocity(x: jnp.ndarray, params: Params) -> jnp.ndarray:
    """Extract generalized velocity components for known state layouts."""
    if params.dim_x == 4 and params.dim_u == 2:
        # Pure second-order model: x = [px, py, vx, vy]
        return x[2:4]
    if params.dim_x == 6 and params.dim_u == 3:
        # Legacy yaw-inclusive model: x = [px, py, vx, vy, yaw, yaw_rate]
        return jnp.array([x[2], x[3], x[5]], dtype=x.dtype)

    # Fallback for generic second-order-like layouts where position leads the state.
    return x[2:2 + params.dim_u]

_J2 = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=jnp.float32)
def stage_cost(x: jnp.ndarray, u: jnp.ndarray, u_prev: jnp.ndarray, params: Params) -> float:
    """
    MPPI stage cost combining safety constraints with Underdamped Langevin
    Dynamics (ULD) ergodic tracking.
    """
    # ==========================================
    # 1. Hard Constraints (Safety & Boundaries)
    # ==========================================
    collided = _is_collided(x[:2], params.obstacle_params)
    px, py = x[0], x[1]
    oom = (
        (px < params.map_x_limits[0]) | (px > params.map_x_limits[1]) |
        (py < params.map_y_limits[0]) | (py > params.map_y_limits[1])
    )
    safety_cost = (
        jnp.where(collided, params.obstacle_params.weight, 0.0)
        + jnp.where(oom, params.oom_cost, 0.0)
    )

    # ==========================================
    # 2. ULD Ergodic Tracking Cost
    # ==========================================
    vel = _state_velocity(x, params)
    score = score_pdf(x, params)[:2]
    I = jnp.eye(2, dtype=x.dtype)
    # rot_score = (I + params.c_orbit * _J2) @ score
    # gyro_damping = params.gamma_uld * vel - params.omega_mag * (_J2 @ vel)

    # core.py — stage_cost becomes clean
    rot_score    = score - params.gamma_uld * vel
    gyro_damping = params.alpha_irrev * _J2 @ (score + vel)
    a_ref        = rot_score + gyro_damping

    if params.dim_u > 2:
        # Keep non-translational channels neutral while matching translational drift.
        u_ref = jnp.concatenate(
            [a_ref, jnp.zeros((params.dim_u - 2,), dtype=x.dtype)],
            axis=0,
        )
    else:
        u_ref = a_ref
    uld_cost = params.w_uld * jnp.sum(jnp.square(u - u_ref))

    # ==========================================
    # 3. Control Regularization
    # ==========================================
    control_cost = params.w_mag * jnp.sum(jnp.square(u))

    rate_cost = params.w_rate * jnp.sum(jnp.square(u - u_prev))

    total_cost = safety_cost + uld_cost + control_cost + rate_cost
    return total_cost


def terminal_cost(x: jnp.ndarray, params: Params) -> float:
    """
    Quadratic terminal cost for double integrator.
    x: (dim_x,)
    """
    cost_x = 0.0
    return cost_x

def _moving_average_filter(xx: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """apply moving average filter for smoothing input sequence
    Handles both 1D and 2D inputs. For 2D, applies filter per column (control dimension).
    """
    def apply_filter_1d(x_1d: jnp.ndarray) -> jnp.ndarray:
        b = jnp.ones(window_size) / window_size
        # Use 'same' mode which pads and returns output of same length
        return jnp.convolve(x_1d, b, mode="same")
    
    if xx.ndim == 1:
        return apply_filter_1d(xx)
    elif xx.ndim == 2:
        # Apply filter to each column (control dimension)
        return jax.vmap(apply_filter_1d, in_axes=1, out_axes=1)(xx)
    else:
        raise ValueError(f"Expected 1D or 2D input, got shape {xx.shape}")

def batched_rollouts(params: Params,
                     x0: jnp.ndarray,              # (dim_x,)
                     U_prev: jnp.ndarray,          # (T, dim_u)
                     eps: jnp.ndarray,             # (K, T, dim_u)
                     use_nominal: jnp.ndarray):    # (K,) bool
    """
    Batched rollouts with one scan over time.

    Returns:
      S: (K,)
      V: (K, T, dim_u)
      pos_traj: (K, T, 2)
    """
    T, K, _ = eps.shape

    x = jnp.broadcast_to(x0, (K, params.dim_x))              # (K, dim_x)
    S = jnp.zeros((K,), dtype=jnp.float32)                   # (K,)

    def step(carry, inputs):
        x, S, u_prev = carry
        U_t, e_t = inputs                                    # U_t: (dim_u,), e_t: (K, dim_u)

        # Apply nominal per rollout
        v_t = jnp.where(use_nominal[:, None], U_t[None, :] + e_t, e_t)  # (K, dim_u)
        u_t = jax.vmap(model.clamp, in_axes=(0, None))(v_t, params.model_params)  # (K, dim_u)

        # Cross term (vectorized over K)
        cross = params.gamma * (U_t[None, :] @ params.Sigma_inv * v_t).sum(axis=1)  # (K,)

        # Stage cost vectorized over K
        sc = jax.vmap(stage_cost, in_axes=(0, 0, 0, None))(x, u_t, u_prev, params)    # (K,)

        S = S + sc + cross
        x = jax.vmap(model.step, in_axes=(0, 0, None))(x, u_t, params.model_params) # (K, dim_x)

        pos = x[:, :2]                                       # (K, 2)
        return (x, S, u_t), (u_t, pos)

    u_prev0 = jnp.broadcast_to(U_prev[0], (params.K, params.dim_u))
    (xT, S, _), (V, pos_traj) = jax.lax.scan(step, (x, S, u_prev0), (U_prev, eps))
    # V: (T, K, dim_u) -> (K, T, dim_u)
    V = jnp.swapaxes(V, 0, 1)
    pos_traj = jnp.swapaxes(pos_traj, 0, 1)

    # Terminal cost per rollout (if any)
    tc = jax.vmap(terminal_cost, in_axes=(0, None))(xT, params)  # (K,)
    S = S + tc

    return S, V, pos_traj


def shift_U(U: jnp.ndarray) -> jnp.ndarray:
    U_next = jnp.roll(U, shift=-1, axis=0)
    U_next = U_next.at[-1].set(U[-1])
    return U_next


@jax.jit
def mppi_step(
params: Params,
U_prev: jnp.ndarray,
x0: jnp.ndarray,
key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    One MPPI step.

    Returns:
        u0:             (dim_u,)      first control to apply
        U_next:         (T, dim_u)   shifted control trajectory for next step
        key:            updated PRNG key
        trajs:          (K, T, 2)    sampled position trajectories
        opt_traj:       (T, dim_x)   optimal (weighted) full-state trajectory
        w:              (K,)         normalized importance weights (for ESS adaptation)
    """
    eps, key = sample_epsilon(key, params)  # (K,T,dim_u)

    eps_T = jnp.swapaxes(eps, 0, 1)  # (T, K, dim_u)
    S, V, trajs = batched_rollouts(
    params=params,
    x0=x0,
    U_prev=U_prev,
    eps=eps_T,                    # (K, T, dim_u)
    use_nominal=params.use_nominal,  # (K,)
    )

    rho = jnp.min(S)
    w_unnorm = jnp.exp(-(S - rho) / params.lam)
    w = w_unnorm / (jnp.sum(w_unnorm) + 1e-12)

    delta = V - U_prev[None, :, :]
    w_eps = jnp.einsum("k,ktu->tu", w, delta)
    # w_eps = _moving_average_filter(w_eps, window_size=5)  # Smooth the control updates
    U = U_prev + w_eps

    def opt_step(x, u):
        x = model.step(x, u, params.model_params)
        return x, x

    _, opt_traj = jax.lax.scan(opt_step, x0, U)

    u0 = U.at[0].get()
    U_next = shift_U(U)
    return u0, U_next, key, trajs, opt_traj, w
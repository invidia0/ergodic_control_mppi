from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from models.jax import double_integrator as model
from typing import NamedTuple
from mppi.jax.stein import SteinParams, stein_grad_traj, logpdf


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObstacleParams:
    xyr: jnp.ndarray  # (num_obstacles, 3) (x, y, r)
    weight: float = field(metadata={"static": True})
    safe_distance: float = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MPPIParams:
    K: int = field(metadata={"static": True})
    T: int = field(metadata={"static": True})
    dim_u: int = field(metadata={"static": True})
    dim_x: int = field(metadata={"static": True})

    # MPPI scalars
    lam: float
    alpha: float
    gamma: float

    use_nominal: jnp.ndarray = field(metadata={"static": True})

    Sigma: jnp.ndarray # (dim_u, dim_u)
    Sigma_inv: jnp.ndarray # (dim_u, dim_u)

    map_x_limits: jnp.ndarray   # (2,)
    map_y_limits: jnp.ndarray   # (2,)
    resolution: float
    oom_cost: float

    delta_t: float

    stein: SteinParams

    model_params: model.DoubleIntegratorParams

    obstacle_params: ObstacleParams


def sample_epsilon(key: jax.Array, params: MPPIParams) -> tuple[jnp.ndarray, jax.Array]:
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


def stage_cost(x: jnp.ndarray, u: jnp.ndarray, params: MPPIParams) -> float:
    collided = _is_collided(x[:2], params.obstacle_params)
    return jnp.where(collided, params.obstacle_params.weight, 0.0)


def terminal_cost(x: jnp.ndarray, params: MPPIParams) -> float:
    """
    Quadratic terminal cost for double integrator.
    x: (dim_x,)
    """
    cost_x = 0.0
    return cost_x


def single_rollout(params: MPPIParams, x0: jnp.ndarray, U_prev: jnp.ndarray, eps_k: jnp.ndarray, use_nominal: jnp.ndarray):
    """
    Rollout one sample.
    eps_k: (T, dim_u)
    use_nominal: scalar bool (JAX bool)
    """
    def scan_step(carry, inputs):
        x, S = carry
        U_t, e_t = inputs

        v_t = jnp.where(use_nominal, U_t + e_t, e_t)
        u_t = model.clamp(v_t, params.model_params)

        cross = params.gamma * (U_t @ params.Sigma_inv @ v_t)
        S = S + stage_cost(x, u_t, params) + cross
        x = model.step(x, u_t, params.model_params)
        return (x, S), (u_t, x)

    (x, S), (V, traj) = jax.lax.scan(scan_step, (x0, jnp.array(0.0, jnp.float32)), (U_prev, eps_k))

    S = S + terminal_cost(x, params)
    return S, V, traj


def shift_U(U: jnp.ndarray) -> jnp.ndarray:
    U_next = jnp.roll(U, shift=-1, axis=0)
    U_next = U_next.at[-1].set(U[-1])
    return U_next


@jax.jit
def mppi_step(
params: MPPIParams,
U_prev: jnp.ndarray,
x0: jnp.ndarray,
key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray, jnp.ndarray]:
    eps, key = sample_epsilon(key, params)  # (K,T,dim_u)

    S, V, trajs = jax.vmap(
        lambda e, un: single_rollout(params, x0, U_prev, e, un),
        in_axes=(0, 0),
        out_axes=(0, 0, 0),
    )(eps, params.use_nominal)

    spatial_trajs = trajs[:, :, :2]  # (K,T,2)
    spatial_median = jnp.median(spatial_trajs, axis=0)  # (T,2)
    h_target = stein_grad_traj(spatial_median, params.stein)  # (T,2)

    S_flow = -jnp.einsum('ktn,tn->k', spatial_trajs, h_target)
    S = S + params.stein.weight * S_flow

    logp = jax.vmap(
        lambda traj_pos: jax.vmap(
            lambda x: logpdf(x, params.stein)
        )(traj_pos)
    )(spatial_trajs)

    # Equivalent to stage-wise cost
    S_pdf = -jnp.sum(logp, axis=1)
    S = (
        S
        + params.stein.weight * S_flow
        + params.stein.weight_pdf  * S_pdf
    )

    rho = jnp.min(S)
    w_unnorm = jnp.exp(-(S - rho) / params.lam)
    w = w_unnorm / (jnp.sum(w_unnorm) + 1e-12)

    delta = V - U_prev[None, :, :]
    w_eps = jnp.einsum("k,ktu->tu", w, delta)
    U = U_prev + w_eps

    def opt_step(x, u):
        x = model.step(x, u, params.model_params)
        return x, x

    _, opt_traj = jax.lax.scan(opt_step, x0, U)

    u0 = U.at[0].get()
    U_next = shift_U(U)
    return u0, U_next, key, trajs, opt_traj
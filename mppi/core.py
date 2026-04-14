import dataclasses
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from models import double_integrator as model
from mppi.stein import SteinParams, stein_grad_state, stein_repulsion_state, logpdf



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

    lam: float
    control_cost_coeff: float

    use_nominal: jnp.ndarray  # (K,) bool - dynamic field

    Sigma: jnp.ndarray # (dim_u, dim_u)
    Sigma_inv: jnp.ndarray # (dim_u, dim_u)

    map_x_limits: jnp.ndarray   # (2,)
    map_y_limits: jnp.ndarray   # (2,)
    resolution: float
    oom_cost: float

    stein: SteinParams

    model_params: model.DoubleIntegratorParams

    obstacle_params: ObstacleParams

    ess_target: float   # target ESS fraction, e.g. 0.3
    lam_min: float      # lower clamp for lam
    lam_max: float      # upper clamp for lam

    history_len: int = field(metadata={"static": True})  # size of position history buffer
    history: jnp.ndarray  # (history_len, 2)  rolling buffer of executed (x, y) positions

    cross_particles_len: int = field(metadata={"static": True})  # (R-1)*T for multi, 0 for single
    cross_particles: jnp.ndarray  # (cross_particles_len, 2)  other robots' planning trajectories



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
    px, py = x[0], x[1]
    oom = (
        (px < params.map_x_limits[0]) | (px > params.map_x_limits[1]) |
        (py < params.map_y_limits[0]) | (py > params.map_y_limits[1])
    )
    return (
        jnp.where(collided, params.obstacle_params.weight, 0.0)
        + jnp.where(oom, params.oom_cost, 0.0)
    )


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

        cross = params.control_cost_coeff * (U_t @ params.Sigma_inv @ v_t)
        S = S + stage_cost(x, u_t, params) + cross
        x = model.step(x, u_t, params.model_params)
        return (x, S), (u_t, x)

    (x, S), (V, traj) = jax.lax.scan(scan_step, (x0, jnp.array(0.0, jnp.float32)), (U_prev, eps_k))

    S = S + terminal_cost(x, params)
    return S, V, traj



def batched_rollouts(params: MPPIParams,
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
    T, K, dim_u = eps.shape

    x = jnp.broadcast_to(x0, (K, params.dim_x))              # (K, dim_x)
    S = jnp.zeros((K,), dtype=jnp.float32)                   # (K,)

    def step(carry, inputs):
        x, S = carry
        U_t, e_t = inputs                                    # U_t: (dim_u,), e_t: (K, dim_u)

        # Apply nominal per rollout
        v_t = jnp.where(use_nominal[:, None], U_t[None, :] + e_t, e_t)  # (K, dim_u)
        u_t = jax.vmap(model.clamp, in_axes=(0, None))(v_t, params.model_params)  # (K, dim_u)

        # Cross term (vectorized over K)
        cross = params.control_cost_coeff * (U_t[None, :] @ params.Sigma_inv * v_t).sum(axis=1)  # (K,)

        # Stage cost vectorized over K
        sc = jax.vmap(stage_cost, in_axes=(0, 0, None))(x, u_t, params)             # (K,)

        S = S + sc + cross
        x = jax.vmap(model.step, in_axes=(0, 0, None))(x, u_t, params.model_params) # (K, dim_x)

        pos = x[:, :2]                                       # (K, 2)
        return (x, S), (u_t, pos)

    (xT, S), (V, pos_traj) = jax.lax.scan(step, (x, S), (U_prev, eps))
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
params: MPPIParams,
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
        trajectory_surrogate: (T, 2) temporal median of sampled position trajectories,
                                     used as a trajectory surrogate in the flow objective.
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

    spatial_trajs = trajs[:, :, :2]  # (K,T,2)
    trajectory_surrogate = jnp.median(spatial_trajs, axis=0)  # (T,2)

    # Median heuristic for kernel bandwidth: ell = max(median(||xi-xj||²), ell_floor)
    diffs = trajectory_surrogate[:, None, :] - trajectory_surrogate[None, :, :]  # (T,T,2)
    sq_dists = jnp.sum(diffs ** 2, axis=-1)                          # (T,T)
    ell_adapted = jnp.maximum(jnp.median(sq_dists), params.stein.ell_self)   # floor = YAML ell_self
    stein_adapted = dataclasses.replace(params.stein, ell_self=ell_adapted)
    params_ell = dataclasses.replace(params, stein=stein_adapted)

    # ell_self: repel from own planning trajectory + own position history
    self_particles = jnp.concatenate([trajectory_surrogate, params_ell.history], axis=0)  # (T+H, 2)
    ell_self = jax.vmap(stein_grad_state, in_axes=(0, None, None))(
        trajectory_surrogate, self_particles, params_ell.stein
    )  # (T, 2)

    # ell_cross: repel from other robots' planning trajectories.
    # cross_particles_len is static → this branch is resolved at Python/trace time:
    # single-robot (len=0) compiles the else; multi-robot compiles the if.
    if params.cross_particles_len > 0:
        ell_cross = jax.vmap(stein_repulsion_state, in_axes=(0, None, None))( # ← was stein_grad_state
            trajectory_surrogate, params_ell.cross_particles, params_ell.stein
        )
        h_target = ell_self + params_ell.stein.alpha_cross * ell_cross
    else:
        h_target = ell_self

    S_flow = -jnp.einsum('ktn,tn->k', spatial_trajs, h_target)

    # logp = jax.vmap(
    #     lambda traj_pos: jax.vmap(
    #         lambda x: logpdf(x, params_h.stein)
    #     )(traj_pos)
    # )(spatial_trajs)

    # # Equivalent to stage-wise cost
    # S_pdf = -jnp.sum(logp, axis=1)
    S = (
        S
        + params.stein.weight_stein * S_flow
        # + params.stein.weight_pdf  * S_pdf
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
    return u0, U_next, key, trajs, opt_traj, trajectory_surrogate, w

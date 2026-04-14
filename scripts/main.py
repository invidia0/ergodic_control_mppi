import dataclasses
import sys
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
from configs import params_loader

from mppi.core import mppi_step
from models import double_integrator as model
from mppi.stein import pdf

_ESS_RATE = 0.05  # multiplicative adaptation step size for lambda


def _adapt_lam(lam: jnp.ndarray, w: jnp.ndarray, params) -> jnp.ndarray:
    """ESS-based multiplicative adaptation of the MPPI temperature lambda."""
    ess_frac = 1.0 / (jnp.sum(w ** 2) * params.K)
    lam_new = lam * jnp.exp(_ESS_RATE * (params.ess_target - ess_frac))
    return jnp.clip(lam_new, params.lam_min, params.lam_max)


# Check for CUDA availability
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
    print(f"[INFO] CUDA device found: {gpu}")
except (RuntimeError, ValueError, IndexError) as exc:
    gpu = cpu
    logging.warning(
        "[INFO] No CUDA device found or CUDA initialization failed, using CPU.",
        exc_info=exc,
    )


# ---------------------------------------------------------------------------
# Single-robot closed loop (original behaviour, backward compatible)
# ---------------------------------------------------------------------------

def closed_loop(params, x0, U0, key, N: int):
    init_trajs = jnp.zeros((params.K, params.T, 2), dtype=jnp.float32)
    init_opt_traj = jnp.zeros((params.T, params.dim_x), dtype=jnp.float32)
    init_surrogate = jnp.zeros((params.T, 2), dtype=jnp.float32)
    init_lam = jnp.asarray(params.lam, dtype=jnp.float32)
    # Seed history with start position so the robot is repelled from it from step 0
    init_history = jnp.broadcast_to(x0[:2], (params.history_len, 2))  # (H, 2)

    def one_step(carry, _):
        x, U_prev, key, lam, history, _, _, _ = carry
        p = dataclasses.replace(params, lam=lam, history=history)
        u0, U_next, key_next, trajs, opt_traj, traj_surrogate, w = mppi_step(p, U_prev, x, key)
        lam_next = _adapt_lam(lam, w, params)
        history_next = jnp.roll(history, shift=-1, axis=0).at[-1].set(x[:2])
        x_next = model.step(x, u0, params.model_params)
        return (x_next, U_next, key_next, lam_next, history_next, trajs, opt_traj, traj_surrogate), x_next

    (_, _, _, _, _, last_trajs, last_opt_traj, _), path = jax.lax.scan(
        one_step,
        (x0, U0, key, init_lam, init_history, init_trajs, init_opt_traj, init_surrogate),
        xs=None,
        length=N,
    )
    return path, last_trajs, last_opt_traj


closed_loop_jit = jax.jit(closed_loop, static_argnames=("N",))


# ---------------------------------------------------------------------------
# Multi-robot closed loop
# ---------------------------------------------------------------------------

def multi_robot_closed_loop(params, x0_all, U0_all, key_all, num_robots: int, N: int):
    """
    Decentralized multi-robot ergodic control loop.

    Each robot runs its own MPPI instance independently. A trajectory
    surrogate is kept in the scan carry and exchanged as Stein particles for
    decentralized cross-robot repulsion.

    Args:
        params:     shared MPPIParams (same map, density, and model for all robots)
        x0_all:     (R, dim_x)      initial states
        U0_all:     (R, T, dim_u)   initial control trajectories
        key_all:    (R, 2)          per-robot PRNG keys
        num_robots: R  — static, used to unroll the inner Python for-loop at
                         JAX trace time (no runtime overhead)
        N:          number of control steps — static for lax.scan

    Returns:
        paths_all:       (N, R, dim_x)  executed state history for every robot
        last_opt_trajs:  (R, T, 2)      final optimal position trajectories
        last_surrogates: (R, T, 2)      final shared trajectory surrogates
    """
    # Initialise carry fields that are not yet meaningful but must have the
    # correct shapes so lax.scan can verify the carry structure.
    surrogates_all = jnp.zeros((num_robots, params.T, 2), dtype=jnp.float32)
    opt_trajs_all = jnp.zeros((num_robots, params.T, 2), dtype=jnp.float32)
    lam_all = jnp.full((num_robots,), params.lam, dtype=jnp.float32)
    # Per-robot history buffers seeded with each robot's start position
    histories_all = jnp.stack([
        jnp.broadcast_to(x0_all[i, :2], (params.history_len, 2))
        for i in range(num_robots)
    ])  # (R, H, 2)

    # cross_particles_len is static so dataclasses.replace inside the scan only
    # changes the dynamic leaf (cross_particles), not the pytree structure.
    cross_len = (num_robots - 1) * (params.T + params.history_len)
    params_base = dataclasses.replace(
        params,
        cross_particles_len=cross_len,
        cross_particles=jnp.zeros((cross_len, 2), dtype=jnp.float32),
    )

    def one_step(carry, _):
        x_all, U_prev_all, key_all, lam_all, histories_all, surrogates_all, _opt_trajs_all = carry

        new_x = []
        new_U = []
        new_keys = []
        new_lams = []
        new_histories = []
        new_surrogates = []
        new_opt_trajs = []

        for i in range(num_robots):
            # Build cross_particles: other robots' planning trajectories + trail histories.
            # At step 0 surrogates/histories are zeros — acceptable 1-step transient.
            other_flat = jnp.concatenate([
                surrogates_all[:i].reshape(-1, 2), # planning trajectories: robots 0..i-1
                histories_all[:i].reshape(-1, 2), # trail histories: robots 0..i-1
                surrogates_all[i+1:].reshape(-1, 2), # planning trajectories: robots i+1..R-1
                histories_all[i+1:].reshape(-1, 2), # trail histories: robots i+1..R-1
            ], axis=0) # ((R-1)*(T+H), 2)

            p = dataclasses.replace(params_base,
                lam=lam_all[i],
                history=histories_all[i],
                cross_particles=other_flat,
            )
            u0, U_next, key_next, _, opt_traj, traj_surrogate, w = mppi_step(
                p, U_prev_all[i], x_all[i], key_all[i]
            )
            x_next = model.step(x_all[i], u0, params.model_params)

            new_x.append(x_next)
            new_U.append(U_next)
            new_keys.append(key_next)
            new_lams.append(_adapt_lam(lam_all[i], w, params))
            new_histories.append(jnp.roll(histories_all[i], shift=-1, axis=0).at[-1].set(x_all[i, :2]))
            new_surrogates.append(traj_surrogate)  # (T, 2)
            new_opt_trajs.append(opt_traj[:, :2]) # (T, 2) — position only

        new_x_all = jnp.stack(new_x, axis=0)  # (R, dim_x)
        new_U_all = jnp.stack(new_U, axis=0)  # (R, T, dim_u)
        new_key_all = jnp.stack(new_keys, axis=0)  # (R, 2)
        new_lam_all = jnp.stack(new_lams, axis=0)  # (R,)
        new_histories_all = jnp.stack(new_histories, axis=0)  # (R, H, 2)
        new_surrogates_all = jnp.stack(new_surrogates, axis=0)  # (R, T, 2)
        new_opt_trajs_all = jnp.stack(new_opt_trajs, axis=0)  # (R, T, 2)

        carry_out = (new_x_all, new_U_all, new_key_all, new_lam_all, new_histories_all, new_surrogates_all, new_opt_trajs_all)
        return carry_out, new_x_all  # output stacked by scan → (N, R, dim_x)

    init_carry = (x0_all, U0_all, key_all, lam_all, histories_all, surrogates_all, opt_trajs_all)
    (_, _, _, _, _, last_surrogates, last_opt_trajs), paths_all = jax.lax.scan(
        one_step, init_carry, xs=None, length=N
    )
    return paths_all, last_opt_trajs, last_surrogates  # (N,R,dim_x), (R,T,2), (R,T,2)


multi_robot_closed_loop_jit = jax.jit(
    multi_robot_closed_loop,
    static_argnames=("num_robots", "N"),
)


def _pdf_background(params):
    n_x = int((params.map_x_limits[1] - params.map_x_limits[0]) / params.resolution)
    n_y = int((params.map_y_limits[1] - params.map_y_limits[0]) / params.resolution)
    gx, gy = jnp.meshgrid(
        jnp.linspace(params.map_x_limits[0], params.map_x_limits[1], n_x),
        jnp.linspace(params.map_y_limits[0], params.map_y_limits[1], n_y),
    )
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
    gp = jax.vmap(pdf, in_axes=(0, None))(pts, params.stein).reshape(gx.shape)
    return np.array(gx), np.array(gy), np.array(gp)


def setup_canvas(fig, ax, params):
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(params.map_x_limits[0], params.map_x_limits[1])
    ax.set_ylim(params.map_y_limits[0], params.map_y_limits[1])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig, ax


def _draw_obstacles(ax, params):
    for (ox, oy, r) in params.obstacle_params.xyr:
        body = Circle((ox, oy), r, facecolor="tab:red", edgecolor="none", alpha=0.20, zorder=3)
        ax.add_patch(body)
        edge = Circle((ox, oy), r, facecolor="none", edgecolor="tab:red",
                      linewidth=1.5, alpha=0.9, zorder=4)
        edge.set_path_effects([
            pe.Stroke(linewidth=3.0, foreground="white", alpha=0.5),
            pe.Normal(),
        ])
        ax.add_patch(edge)


def visualize(params, path=None, trajs=None, opt_traj=None):
    """Single-robot visualization (original)."""
    gx, gy, gp = _pdf_background(params)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig, ax = setup_canvas(fig, ax, params)
    ax.contourf(gx, gy, gp, cmap='Blues', alpha=0.8)
    _draw_obstacles(ax, params)

    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color='black', alpha=0.8, linewidth=2,
                zorder=4, label='Path')
        ax.scatter(path[-1, 0], path[-1, 1], color='tab:red', marker='.', s=100,
                   label='End', zorder=5)

    if trajs is not None:
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1, zorder=3)

    if opt_traj is not None:
        ax.plot(opt_traj[:, 0], opt_traj[:, 1], color='tab:red', alpha=1,
                linewidth=2, zorder=4, label='Opt. Trajectory')

    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_multi(params, paths_all, last_opt_trajs, num_robots):
    """
    Multi-robot visualization.

    Args:
        params:          MPPIParams
        paths_all:       (N, R, dim_x)  executed trajectories for all robots
        last_opt_trajs:  (R, T, 2)      final optimal position trajectories
        num_robots:      R
    """
    gx, gy, gp = _pdf_background(params)
    cmap = plt.cm.get_cmap('tab10', num_robots)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig, ax = setup_canvas(fig, ax, params)
    ax.contourf(gx, gy, gp, cmap='Blues', alpha=0.8)
    _draw_obstacles(ax, params)

    paths_np = np.array(paths_all)     # (N, R, dim_x)
    opt_np = np.array(last_opt_trajs) # (R, T, 2)

    for i in range(num_robots):
        color  = cmap(i)
        path_i = paths_np[:, i, :]       # (N, dim_x)

        ax.plot(path_i[:, 0], path_i[:, 1],
                color=color, alpha=0.85, linewidth=1.5, zorder=4,
                label=f'Robot {i}')

        # Start marker (star)
        ax.scatter(path_i[0, 0], path_i[0, 1],
                   color=color, marker='*', s=150, zorder=6,
                   edgecolors='white', linewidths=0.5)

        # End marker (circle)
        ax.scatter(path_i[-1, 0], path_i[-1, 1],
                   color=color, marker='o', s=60, zorder=6,
                   edgecolors='white', linewidths=0.5)

        # Final optimal trajectory (dashed)
        ax.plot(opt_np[i, :, 0], opt_np[i, :, 1],
                color=color, alpha=0.5, linewidth=2, linestyle='--', zorder=5)

    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"Multi-Robot Ergodic Coverage ({num_robots} robots)")
    plt.tight_layout()
    plt.show()


def _random_state(key, params):
    """Sample a random initial state within the map bounds."""
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    px = jax.random.uniform(k1, minval=params.map_x_limits[0], maxval=params.map_x_limits[1], shape=())
    py = jax.random.uniform(k2, minval=params.map_y_limits[0], maxval=params.map_y_limits[1], shape=())
    vx = jax.random.uniform(k3, minval=-1.0, maxval=1.0, shape=())
    vy = jax.random.uniform(k4, minval=-1.0, maxval=1.0, shape=())
    yaw = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi, shape=())
    yaw_rate = jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=())
    return jnp.array([px, py, vx, vy, yaw, yaw_rate], dtype=jnp.float32)


def main():
    config_path = "configs/mppi_params.yaml"
    params = params_loader.load_mppi_params(config_path)
    run_cfg = params_loader.load_run_config(config_path)

    num_robots = run_cfg.num_robots
    N = run_cfg.steps
    key = jax.random.PRNGKey(run_cfg.seed)

    if num_robots == 1:
        # ---- Single-robot path (original behaviour) ----
        key, subkey = jax.random.split(key)
        x0 = _random_state(subkey, params)
        U_prev = jnp.zeros((params.T, params.dim_u), dtype=jnp.float32)

        x0 = jax.device_put(x0, gpu)
        U_prev = jax.device_put(U_prev, gpu)
        key = jax.device_put(key, gpu)
        params = jax.device_put(params, gpu)

        print("Running single-robot closed-loop simulation...")
        path, trajs, opt_traj = closed_loop_jit(params, x0, U_prev, key, N=N)
        print("Done.")
        visualize(params, path, trajs, opt_traj)

    else:
        # ---- Multi-robot path ----
        # Give each robot a unique PRNG key and random start state.
        robot_keys = jax.random.split(key, num_robots) # (R, 2)
        x0_list = [_random_state(robot_keys[i], params) for i in range(num_robots)]
        x0_all = jnp.stack(x0_list, axis=0) # (R, dim_x)
        U0_all = jnp.zeros((num_robots, params.T, params.dim_u), dtype=jnp.float32)

        # Fresh independent keys for the simulation (split again to avoid reuse)
        sim_keys = jax.random.split(jax.random.fold_in(key, 1), num_robots)  # (R, 2)

        x0_all = jax.device_put(x0_all, gpu)
        U0_all = jax.device_put(U0_all, gpu)
        sim_keys = jax.device_put(sim_keys, gpu)
        params = jax.device_put(params, gpu)

        print(f"Running multi-robot closed-loop simulation ({num_robots} robots)...")
        paths_all, last_opt_trajs, last_surrogates = multi_robot_closed_loop_jit(
            params, x0_all, U0_all, sim_keys, num_robots=num_robots, N=N
        )
        print("Done.")
        visualize_multi(params, paths_all, last_opt_trajs, num_robots)


if __name__ == "__main__":
    main()

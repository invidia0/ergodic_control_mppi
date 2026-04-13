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

from mppi.core import logpdf, mppi_step
from models import double_integrator as model

_ESS_RATE = 0.05

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

def closed_loop(params, x0, U0, key, N: int):
    init_trajs = jnp.zeros((params.K, params.T, 2), dtype=jnp.float32)
    init_opt_traj = jnp.zeros((params.T, params.dim_x), dtype=jnp.float32)
    init_lam = jnp.asarray(params.lam, dtype=jnp.float32)

    def one_step(carry, _):
        x, U_prev, key, lam, _, _ = carry
        # p = dataclasses.replace(params, lam=lam)
        u0, U_next, key_next, trajs, opt_traj, w = mppi_step(params, U_prev, x, key)
        lam_next = _adapt_lam(lam, w, params)
        x_next = model.step(x, u0, params.model_params)
        return (x_next, U_next, key_next, lam_next, trajs, opt_traj), x_next

    (_, _, _, _, last_trajs, last_opt_traj), path = jax.lax.scan(
        one_step,
        (x0, U0, key, init_lam, init_trajs, init_opt_traj),
        xs=None,
        length=N,
    )
    return path, last_trajs, last_opt_traj


closed_loop_jit = jax.jit(closed_loop, static_argnames=("N",))


def _pdf_background(params):
    n_x = int((params.map_x_limits[1] - params.map_x_limits[0]) / params.resolution)
    n_y = int((params.map_y_limits[1] - params.map_y_limits[0]) / params.resolution)
    gx, gy = jnp.meshgrid(
        jnp.linspace(params.map_x_limits[0], params.map_x_limits[1], n_x),
        jnp.linspace(params.map_y_limits[0], params.map_y_limits[1], n_y),
    )
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
    gp = jnp.exp(jax.vmap(logpdf, in_axes=(0, None))(pts, params)).reshape(gx.shape)
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
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], color='black', alpha=0.8, linewidth=2,
                zorder=4, label='Path')
        ax.scatter(path[-1, 0], path[-1, 1], color='tab:red', marker='.', s=100,
                   label='End', zorder=5)

    if trajs is not None:
        trajs = np.asarray(trajs)
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1, zorder=3)

    if opt_traj is not None:
        opt_traj = np.asarray(opt_traj)
        ax.plot(opt_traj[:, 0], opt_traj[:, 1], color='tab:red', alpha=1,
                linewidth=2, zorder=4, label='Opt. Trajectory')

    ax.legend()
    plt.tight_layout()
    plt.show()

# def _random_state(key, params):
#     """Sample a random initial state within the map bounds."""
#     key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
#     px = jax.random.uniform(k1, minval=params.map_x_limits[0], maxval=params.map_x_limits[1], shape=())
#     py = jax.random.uniform(k2, minval=params.map_y_limits[0], maxval=params.map_y_limits[1], shape=())
#     vx = jax.random.uniform(k3, minval=-1.0, maxval=1.0, shape=())
#     vy = jax.random.uniform(k4, minval=-1.0, maxval=1.0, shape=())
#     yaw = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi, shape=())
#     yaw_rate = jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=())
#     return jnp.array([px, py, vx, vy, yaw, yaw_rate], dtype=jnp.float32)
def _random_state(key, params):
    """Sample a random initial state within the map bounds."""
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    px = jax.random.uniform(k1, minval=params.map_x_limits[0], maxval=params.map_x_limits[1], shape=())
    py = jax.random.uniform(k2, minval=params.map_y_limits[0], maxval=params.map_y_limits[1], shape=())
    vx = jax.random.uniform(k3, minval=-1.0, maxval=1.0, shape=())
    vy = jax.random.uniform(k4, minval=-1.0, maxval=1.0, shape=())
    return jnp.array([px, py, vx, vy], dtype=jnp.float32)


def main():
    params = params_loader.load_params("configs/mppi_params.yaml")

    num_robots = params.num_robots

    N = params.steps
    key = jax.random.PRNGKey(params.seed)

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


if __name__ == "__main__":
    main()

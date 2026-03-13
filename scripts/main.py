import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from configs import params_loader

from mppi.core import mppi_step
from models import double_integrator as model
from mppi.stein import pdf

import matplotlib.patheffects as pe
from matplotlib.patches import Circle

# Check for CUDA availability
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
    print(f"[INFO] CUDA device found: {gpu}")
except:
    gpu = cpu
    print("[INFO] No CUDA device found, using CPU.")


def closed_loop(params, x0, U0, key, N: int):
    init_trajs = jnp.zeros((params.K, params.T, 2), dtype=jnp.float32)
    init_opt_traj = jnp.zeros((params.T, params.dim_x), dtype=jnp.float32)

    def one_step(carry, _):
        x, U_prev, key, _, _ = carry
        u0, U_next, key_next, trajs, opt_traj = mppi_step(params, U_prev, x, key)
        x_next = model.step(x, u0, params.model_params)
        return (x_next, U_next, key_next, trajs, opt_traj), x_next

    (_, _, _, last_trajs, last_opt_traj), path = jax.lax.scan(
        one_step,
        (x0, U0, key, init_trajs, init_opt_traj),
        xs=None,
        length=N,
    )

    return path, last_trajs, last_opt_traj


closed_loop_jit = jax.jit(
    closed_loop,
    static_argnames=("N",)
)

def setup_canvas(fig, ax, params):
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    x_limits = params.map_x_limits
    y_limits = params.map_y_limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig, ax


def visualize(params, path=None, trajs=None, opt_traj=None):
    n_x = int((params.map_x_limits[1] - params.map_x_limits[0]) / params.resolution)
    n_y = int((params.map_y_limits[1] - params.map_y_limits[0]) / params.resolution)
   
    grids_x, grids_y = jnp.meshgrid(
        jnp.linspace(params.map_x_limits[0], params.map_x_limits[1], n_x),
        jnp.linspace(params.map_y_limits[0], params.map_y_limits[1], n_y),
    )

    grids_jax = jnp.array(jnp.stack([grids_x.ravel(), grids_y.ravel()], axis=1))
    pdf_grids = jax.vmap(pdf, in_axes=(0, None))(grids_jax, params.stein)
    pdf_grids = pdf_grids.reshape(grids_x.shape)

    grids_x_np = np.array(grids_x)
    grids_y_np = np.array(grids_y)
    pdf_grids_np = np.array(pdf_grids)

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    fig, ax = setup_canvas(fig, ax, params)
    cs = ax.contourf(grids_x_np, grids_y_np, pdf_grids_np, cmap='Blues', alpha=0.8)

    for (ox, oy, r) in params.obstacle_params.xyr:
        body = Circle(
            (ox, oy), r,
            facecolor="tab:red",
            edgecolor="none",
            alpha=0.20,
            zorder=3,
        )
        ax.add_patch(body)

        edge = Circle(
            (ox, oy), r,
            facecolor="none",
            edgecolor="tab:red",
            linewidth=1.5,
            alpha=0.9,
            zorder=4,
        )
        edge.set_path_effects([
            pe.Stroke(linewidth=3.0, foreground="white", alpha=0.5),  # halo
            pe.Normal()
        ])
        ax.add_patch(edge)

    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color='black', alpha=0.8, linewidth=2, zorder=4, label='Path')
        xs = path[-1, :2]
        ax.scatter(xs[0], xs[1], color='tab:red', marker='.', s=100, label='End', zorder=5)

    if trajs is not None:
        # Last sampled trajectory batch from the final closed-loop step: (K, T, 2)
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1, zorder=3)
        
    
    if opt_traj is not None:
        ax.plot(opt_traj[:, 0], opt_traj[:, 1], color='tab:red', alpha=1, linewidth=2, zorder=4, label='Opt. Trajectory')

    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    params = params_loader.load_mppi_params("configs/mppi_params.yaml")

    N = params.steps

    key = jax.random.PRNGKey(params.seed)
    
    px = jax.random.uniform(
        key,
        shape=(),
        minval=params.map_x_limits[0],
        maxval=params.map_x_limits[1],
    )
    key, subkey = jax.random.split(key)
    py = jax.random.uniform(
        subkey,
        shape=(),
        minval=params.map_y_limits[0],
        maxval=params.map_y_limits[1],
    )
    key, subkey = jax.random.split(key)
    vx = jax.random.uniform(
        subkey,
        shape=(),
        minval=-1.0,
        maxval=1.0,
    )
    key, subkey = jax.random.split(key)
    vy = jax.random.uniform(
        subkey,
        shape=(),
        minval=-1.0,
        maxval=1.0,
    )
    key, subkey = jax.random.split(key)
    yaw = jax.random.uniform(
        subkey,
        shape=(),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    key, subkey = jax.random.split(key)
    yaw_rate = jax.random.uniform(
        subkey,
        shape=(),
        minval=-1.0,
        maxval=1.0,
    )
    x0 = jnp.array(
        [px, py, vx, vy, yaw, yaw_rate],
        dtype=jnp.float32,
    )
    U_prev = jnp.zeros((params.T, params.dim_u), dtype=jnp.float32)

    x0 = jax.device_put(x0, gpu)
    U_prev = jax.device_put(U_prev, gpu)
    key = jax.device_put(key, gpu)
    params = jax.device_put(params, gpu)

    print("Running closed-loop simulation...")
    path, trajs, opt_traj = closed_loop_jit(
        params,
        x0,
        U_prev,
        key,
        N=N,
    )
    print("Done.")
    visualize(params, path, trajs, opt_traj)

if __name__ == "__main__":
    main()
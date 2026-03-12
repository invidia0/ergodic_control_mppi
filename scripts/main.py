import os
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
# from mppi.stein import drift

from dataclasses import replace

import matplotlib.patheffects as pe
from matplotlib.patches import Circle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def closed_loop(params, x0, U0, key, N: int):
    def one_step(carry, _):
        x, U_prev, key = carry
        key, subkey = jax.random.split(key)
        # Donate U_prev and x to save memory
        u0, U_next, _, trajs, opt_traj = jax.jit(mppi_step)(params, U_prev, x, subkey)
        x_next = model.step(x, u0, params.model_params)
        return (x_next, U_next, key), (x_next, u0, trajs, opt_traj)

    (xN, UN, keyN), (xs, us, trajs_all, opt_trajs_all) = jax.lax.scan(
        one_step,
        (x0, U0, key),
        xs=None,
        length=N,
    )

    return xs, us, UN, keyN, trajs_all, opt_trajs_all


closed_loop_jit = jax.jit(closed_loop, static_argnames=("N",))


def setup_canvas(fig, ax, params):
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    # Update every font to serif
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for
    x_limits = params.map_x_limits
    y_limits = params.map_y_limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    # ax.set_title("MPPI Double Integrator Benchmark")
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig, ax


def visualize(params, xs=None, trajs_all=None, opt_trajs_all=None):

    # grid
    n_x = int((params.map_x_limits[1] - params.map_x_limits[0]) / params.resolution)
    n_y = int((params.map_y_limits[1] - params.map_y_limits[0]) / params.resolution)

    grids_x, grids_y = jnp.meshgrid(
        jnp.linspace(params.map_x_limits[0], params.map_x_limits[1], n_x),
        jnp.linspace(params.map_y_limits[0], params.map_y_limits[1], n_y),
    )

    grids_jax = jnp.array(jnp.stack([grids_x.ravel(), grids_y.ravel()], axis=1))
    pdf_grids = jax.vmap(pdf, in_axes=(0, None))(grids_jax, params.stein)
    pdf_grids = pdf_grids.reshape(grids_x.shape)

    # fs = jax.vmap(lambda x:drift(x, params.stein))(grids_jax)
    # fs_x = fs[:, 0].reshape(grids_x.shape)
    # fs_y = fs[:, 1].reshape(grids_y.shape)

    grids_x_np = np.array(grids_x)
    grids_y_np = np.array(grids_y)
    pdf_grids_np = np.array(pdf_grids)

    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    # Update every font to serif
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for
    # IEEE_column_size = 3.5  # in inches
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    fig, ax = setup_canvas(fig, ax, params)
    cs = ax.contourf(grids_x_np, grids_y_np, pdf_grids_np, cmap='Blues', alpha=0.8)

    for (ox, oy, r) in params.obstacle_params.xyr:
        # Filled body (soft)
        body = Circle(
            (ox, oy), r,
            facecolor="tab:red",
            edgecolor="none",
            alpha=0.20,
            zorder=3,
        )
        ax.add_patch(body)

        # Crisp boundary + subtle halo for readability
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

    if trajs_all is not None:
        for traj in trajs_all[-1, :, :, :2]:
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1, zorder=3)
    
    if opt_trajs_all is not None:
        ax.plot(opt_trajs_all[-1][:, 0], opt_trajs_all[-1][:, 1], color='tab:red', alpha=1, linewidth=2, zorder=4, label='Opt. Trajectory')
    # Soft black
    
    color ="#424242"
    if xs is not None:
        ax.plot(xs[:, 0], xs[:, 1], label="Trajectory", color=color)
        ax.scatter(xs[-1, 0], xs[-1, 1], color='tab:red', marker='.', s=100, label='End', zorder=5)
        ax.scatter(xs[0, 0], xs[0, 1], color='tab:green', marker='.', s=100, label='Start', zorder=5)


    # # Save trajectory as numpy file
    # np.save(os.path.join(os.getcwd(), "scripts/test_sensitivity_params/N{}L{}SEED{}.npy".format(params.K, params.lam, params.seed)), xs)
    np.save(os.path.join(os.getcwd(), "scripts/video_material/experiment2_{}_{}_{}.npy".format(params.stein.D[0,0]*2, params.stein.gamma, params.seed)), xs)
    # Fancy legend
    ax.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), "scripts/", "plots/mppi_double_integrator_benchmark.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main_nosim(N=2000):
    params = params_loader.load_mppi_params("configs/mppi_params.yaml")

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


    print("Running closed-loop simulation...")
    xs, us, U_prev, key, trajs_all, opt_trajs_all = closed_loop_jit(
        params,
        x0,
        U_prev,
        key,
        N=N,
    )
    visualize(params, xs, trajs_all, opt_trajs_all)
    print("Done.")

if __name__ == "__main__":
    N = 5000
    main_nosim(N=N)
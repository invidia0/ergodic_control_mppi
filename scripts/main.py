import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from configs import params_loader

from mppi.mppi_core import mppi_step
from models import double_integrator as model
from mppi.stein import pdf
from mppi.stein import drift


def closed_loop(params, x0, U0, key, N: int):
    def one_step(carry, _):
        x, U_prev, key = carry
        key, subkey = jax.random.split(key)
        u0, U_next, _, trajs, opt_traj = mppi_step(params, U_prev, x, subkey)
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
    x_limits = params.map_x_limits
    y_limits = params.map_y_limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("MPPI Double Integrator Benchmark")
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

    fs = jax.vmap(lambda x:drift(x, params.stein))(grids_jax)
    fs_x = fs[:, 0].reshape(grids_x.shape)
    fs_y = fs[:, 1].reshape(grids_y.shape)

    grids_x_np = np.array(grids_x)
    grids_y_np = np.array(grids_y)
    pdf_grids_np = np.array(pdf_grids)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig, ax = setup_canvas(fig, ax, params)
    cs = ax.contourf(grids_x_np, grids_y_np, pdf_grids_np, cmap='Blues', alpha=0.8)
    # fs_y_np = np.array(fs_y)
    # fs_x_np = np.array(fs_x)
    # ax.streamplot(grids_x_np, grids_y_np, fs_x_np, fs_y_np, color='k', density=1.5, linewidth=0.3, arrowsize=1, arrowstyle='-|>')

    for obs in params.obstacle_params.xyr:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
        ax.add_artist(circle)

    if trajs_all is not None:
        for traj in trajs_all[-1, :, :, :2]:
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1)
    
    if opt_trajs_all is not None:
        ax.plot(opt_trajs_all[-1][:, 0], opt_trajs_all[-1][:, 1], color='green', alpha=1)
    
    if xs is not None:
        ax.plot(xs[:, 0], xs[:, 1], label="Trajectory", color='k')
        ax.scatter(xs[-1, 0], xs[-1, 1], color='red', marker='x', label='End')
    
    ax.legend()
    plt.show()


def main(N=2000):
    params, seed = params_loader.load_mppi_params("configs/mppi_params.yaml")

    x0 = jnp.array(
        [0.0, 0.0, 1.0, 1.0, jnp.deg2rad(45.0), 0.0],
        dtype=jnp.float32,
    )
    U_prev = jnp.zeros((params.T, params.dim_u), dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)

    print("Running closed-loop simulation...")
    xs, us, U_prev, key, trajs_all, opt_trajs_all = closed_loop_jit(
        params,
        x0,
        U_prev,
        key,
        N=N,
    )
    print("Done.")
    visualize(params, xs, trajs_all, opt_trajs_all)

    # Simple alternative just for carry on in versions
    # x = [px, py, vx, vy, yaw, yaw_rate]
    # x = jnp.array([0.0, 0.0, 1,1, jnp.deg2rad(45), 0.0], dtype=jnp.float32)
    # step_jit = jax.jit(lambda x, u: model.step(x, u, mppi_params.model_params))

    # for _ in tqdm(range(2000)):
    #     out = controller.step(x)
    #     x = step_jit(x, out.u0)

if __name__ == "__main__":
    N = 5000
    main(N=N)
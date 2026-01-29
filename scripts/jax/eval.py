import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
from configs.jax import params_loader

from mppi.jax.mppi_core import mppi_step
from models.jax import double_integrator as model
from mppi.jax.stein import pdf
from mppi.jax.stein import drift
from utils.evaluator import Evaluator


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


def visualize(params, xs=None, trajs_all=None, opt_trajs_all=None, ergodic_metric=None):

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
    # cs = ax.contourf(grids_x_np, grids_y_np, pdf_grids_np, cmap='Blues', alpha=0.8)
    ax.pcolormesh(grids_x_np, grids_y_np, pdf_grids_np, cmap='Blues', shading='auto', alpha=0.8)
    # fs_y_np = np.array(fs_y)
    # fs_x_np = np.array(fs_x)
    # ax.streamplot(grids_x_np, grids_y_np, fs_x_np, fs_y_np, color='k', density=1.5, linewidth=0.3, arrowsize=1, arrowstyle='-|>')

    for obs in params.obstacle_params.xyr:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='black', alpha=0.5)
        ax.add_artist(circle)

    if trajs_all is not None:
        for traj in trajs_all[-1, :, :, :2]:
            ax.plot(traj[:, 0], traj[:, 1], color='k', alpha=0.1)
    
    if opt_trajs_all is not None:
        ax.plot(opt_trajs_all[-1][:, 0], opt_trajs_all[-1][:, 1], color='green', alpha=1)
    
    if xs is not None:
        if ergodic_metric is None:
            ax.plot(xs[:, 0], xs[:, 1], label="Trajectory", color='k')
            ax.scatter(xs[-1, 0], xs[-1, 1], color='red', marker='x', label='End')
        else:
            points = np.column_stack([xs[:, 0], xs[:, 1]])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            norm = LogNorm(vmin=np.min(ergodic_metric), vmax=np.max(ergodic_metric))
            lc = LineCollection(segments, cmap='YlOrRd', norm=norm)
            lc.set_array(ergodic_metric)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax, label='Ergodic Metric')
    ax.legend()
    plt.show()


def main(N=2000):
    params, seed = params_loader.load_mppi_params("configs/mppi_params.yaml")
    N_EPISODES = 1
    ergodic_metric_hist = np.zeros((N_EPISODES, N))
    trajs_hist = np.zeros((N_EPISODES, N, 2))
    key = jax.random.PRNGKey(seed)
    for ep in range(N_EPISODES):
        pos = params.map_x_limits[0] + (params.map_x_limits[1] - params.map_x_limits[0]) * jax.random.uniform(key, shape=(2,))
        print(f"Ep: {ep} | Starting pos: {pos}")
        x0 = jnp.array(
            [pos[0], pos[1], 0.0, 0.0, jnp.deg2rad(45.0), 0.0],
            dtype=jnp.float32,
        )
        U_prev = jnp.zeros((params.T, params.dim_u), dtype=jnp.float32)

        print("Running closed-loop simulation...")
        # timer_start = time.perf_counter()
        xs, us, U_prev, key, trajs_all, opt_trajs_all = closed_loop_jit(
            params,
            x0,
            U_prev,
            key,
            N=N,
        )
        trajs_hist[ep, :, :] = np.array(xs[:, 0:2])
        # timer_end = time.perf_counter()
        # print(f"Episode {ep + 1} completed in {timer_end - timer_start:.2f} seconds.")
        # avg_time = (timer_end - timer_start) / N
        # print(f"Average time per step [MPPI]: {avg_time * 1000:.3f} [ms]")
        # print("Done.")
        eval = Evaluator(params, xs)
        ergodic_metric_hist[ep, :] = eval.get_ergodic_metric()
        print("Final ergodic metric: ", ergodic_metric_hist[ep, -1])
        
        visualize(params, xs, trajs_all, opt_trajs_all)

    ergodic_metric = np.mean(ergodic_metric_hist, axis=0)
    lower_bound = np.percentile(ergodic_metric_hist, 5, axis=0)   # 5th percentile (lower)
    upper_bound = np.percentile(ergodic_metric_hist, 95, axis=0)  # 95th percentile (upper)
    np.save(os.path.join(str(Path(__file__).parent.parent.parent), 'results', 'e_mppi.npy'), ergodic_metric_hist)
    np.save(os.path.join(str(Path(__file__).parent.parent.parent), 'results', 'xs_mppi.npy'), trajs_hist)
    fig, ax = plt.subplots()
    ax.plot(np.arange(N) * params.delta_t, ergodic_metric)
    ax.fill_between(
        np.arange(N) * params.delta_t,
        lower_bound,
        upper_bound,
        color='blue',
        alpha=0.2,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Ergodic Metric Average")
    ax.set_title("Ergodic Metric over Time")
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.show()

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
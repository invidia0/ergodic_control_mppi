"""Interactive and file-based visualization of simulation results."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from ergodic_control_mppi.config import AppConfig
from ergodic_control_mppi.metrics.ergodicity import (
    compute_cumulative_team_ergodic_error,
    compute_team_occupancy_grid,
)
from ergodic_control_mppi.mppi.stein import pdf
from ergodic_control_mppi.simulation import SimulationResult


def plot_simulation(
    config: AppConfig,
    result: SimulationResult,
    output: str | Path | None = None,
    show: bool = True,
):
    """Render coverage, occupancy, and cumulative ergodic error panels.

    Args:
        config: Configuration used for the simulation.
        result: Normalized simulation result.
        output: Optional image path.
        show: Whether to display the Matplotlib window.

    Returns:
        The created Matplotlib figure.
    """
    params = config.controller
    workspace = params.workspace
    nx = max(2, int((workspace.x_limits[1] - workspace.x_limits[0]) / config.run.resolution))
    ny = max(2, int((workspace.y_limits[1] - workspace.y_limits[0]) / config.run.resolution))
    x = jnp.linspace(workspace.x_limits[0], workspace.x_limits[1], nx)
    y = jnp.linspace(workspace.y_limits[0], workspace.y_limits[1], ny)
    grid_x, grid_y = jnp.meshgrid(x, y)
    density = np.asarray(pdf(jnp.stack((grid_x, grid_y), axis=-1), params.gmm))
    target = density / density.sum()
    limits_x = tuple(map(float, workspace.x_limits))
    limits_y = tuple(map(float, workspace.y_limits))
    occupancy = compute_team_occupancy_grid(result.paths, limits_x, limits_y, (nx, ny))
    errors = compute_cumulative_team_ergodic_error(
        result.paths, target, limits_x, limits_y, (nx, ny)
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].contourf(grid_x, grid_y, density, cmap="Blues", alpha=0.8)
    colors = plt.colormaps["tab10"]
    for robot in range(result.paths.shape[1]):
        path = result.paths[:, robot]
        axes[0].plot(path[:, 0], path[:, 1], color=colors(robot), label=f"Robot {robot + 1}")
        axes[0].plot(
            result.optimal_trajectories[robot, :, 0],
            result.optimal_trajectories[robot, :, 1],
            "--",
            color=colors(robot),
        )
    for ox, oy, radius in np.asarray(workspace.obstacles):
        axes[0].add_patch(Circle((ox, oy), radius, color="tab:red", alpha=0.3))
    axes[0].legend()
    axes[0].set_title("Coverage")
    axes[1].imshow(occupancy, origin="lower", extent=(*limits_x, *limits_y), cmap="magma")
    axes[1].set_title("Empirical density")
    axes[2].plot(np.arange(1, len(errors) + 1), errors)
    axes[2].set_yscale("log")
    axes[2].set_title("Ergodic error")
    for axis in axes[:2]:
        axis.set(xlim=limits_x, ylim=limits_y, aspect="equal", xlabel="X [m]", ylabel="Y [m]")
    axes[2].set(xlabel="Step", ylabel="Occupancy MSE")
    figure.tight_layout()
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    return figure

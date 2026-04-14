from __future__ import annotations

import numpy as np


def _binary_visited_cells(
    robot_paths: np.ndarray,
    robot_idx: int,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> np.ndarray:
    xy = np.asarray(robot_paths, dtype=np.float64)[:, robot_idx, :2]
    hist, _, _ = np.histogram2d(
        xy[:, 1],
        xy[:, 0],
        bins=bins,
        range=[
            [map_y_limits[0], map_y_limits[1]],
            [map_x_limits[0], map_x_limits[1]],
        ],
    )
    return (hist > 0).astype(np.float64)


def compute_redundancy_metric(
    robot_paths: np.ndarray,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] = (40, 40),
) -> float:
    """
    Lower is better.
    Cell-wise overlap redundancy in [0, 1]:
      0 => no shared visited cells
      1 => all robots repeatedly visit same covered cells
    """
    paths = np.asarray(robot_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("robot_paths must have shape (steps, robots, state_dim)")
    num_robots = paths.shape[1]
    if num_robots <= 1:
        return 0.0

    visited = np.stack(
        [
            _binary_visited_cells(paths, i, map_x_limits, map_y_limits, bins)
            for i in range(num_robots)
        ],
        axis=0,
    )  # (robots, ny, nx)
    per_cell_count = visited.sum(axis=0)
    covered = per_cell_count > 0
    if not np.any(covered):
        return 0.0

    redundancy_per_cell = (per_cell_count[covered] - 1.0) / (num_robots - 1.0)
    return float(np.mean(redundancy_per_cell))


from __future__ import annotations

import numpy as np


def _normalize_density(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("target density grid must have positive total mass")
    return arr / total


def _team_occupancy(
    robot_paths: np.ndarray,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> np.ndarray:
    xy = np.asarray(robot_paths, dtype=np.float64)[..., :2]
    flat = xy.reshape(-1, 2)
    hist, _, _ = np.histogram2d(
        flat[:, 1],  # y-axis
        flat[:, 0],  # x-axis
        bins=bins,
        range=[
            [map_y_limits[0], map_y_limits[1]],
            [map_x_limits[0], map_x_limits[1]],
        ],
    )
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros_like(hist)
    return hist / total


def compute_team_ergodic_error(
    robot_paths: np.ndarray,
    target_density_grid: np.ndarray,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
) -> float:
    """
    Lower is better.
    Computes MSE between team occupancy and target density on a common grid.
    """
    target = _normalize_density(target_density_grid)
    grid_bins = bins if bins is not None else (target.shape[1], target.shape[0])
    occupancy = _team_occupancy(robot_paths, map_x_limits, map_y_limits, grid_bins)
    if occupancy.shape != target.shape:
        raise ValueError(
            f"occupancy grid shape {occupancy.shape} does not match target shape {target.shape}"
        )
    diff = occupancy - target
    return float(np.mean(diff * diff))


from __future__ import annotations

import itertools

import numpy as np


def _robot_occupancy(
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
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros_like(hist)
    return hist / total


def compute_pairwise_overlap(
    robot_paths: np.ndarray,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] = (40, 40),
) -> float:
    """
    Lower is better.
    Returns average pairwise occupancy overlap in [0, 1].
    """
    paths = np.asarray(robot_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("robot_paths must have shape (steps, robots, state_dim)")
    num_robots = paths.shape[1]
    if num_robots <= 1:
        return 0.0

    occupancies = [
        _robot_occupancy(paths, i, map_x_limits, map_y_limits, bins)
        for i in range(num_robots)
    ]
    overlaps: list[float] = []
    for i, j in itertools.combinations(range(num_robots), 2):
        # histogram intersection, naturally bounded in [0, 1]
        overlap = float(np.minimum(occupancies[i], occupancies[j]).sum())
        overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 0.0


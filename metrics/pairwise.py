from __future__ import annotations

import itertools

import numpy as np


def compute_pairwise_redundancy(
    robot_paths: np.ndarray,
    d_thresh: float = 1.0,
) -> float:
    """
    Mean count of close robot pairs over time.
    R_pair = mean_t sum_{i<j} 1(||x_i(t) - x_j(t)|| < d_thresh)
    """
    paths = np.asarray(robot_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("robot_paths must have shape (steps, robots, state_dim)")
    if d_thresh < 0.0:
        raise ValueError("d_thresh must be >= 0")

    xy = paths[..., :2]  # (steps, robots, 2)
    _, num_robots, _ = xy.shape
    if num_robots <= 1:
        return 0.0

    pair_counts = np.zeros(xy.shape[0], dtype=np.float64)
    for i, j in itertools.combinations(range(num_robots), 2):
        d = np.linalg.norm(xy[:, i, :] - xy[:, j, :], axis=1)
        pair_counts += (d < d_thresh).astype(np.float64)
    return float(np.mean(pair_counts))


def compute_pairwise_min_distance(robot_paths: np.ndarray) -> float:
    """
    Minimum pairwise distance across all time and robot pairs.
    D_min_pair = min_{t, i<j} ||x_i(t) - x_j(t)||
    """
    paths = np.asarray(robot_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("robot_paths must have shape (steps, robots, state_dim)")

    xy = paths[..., :2]  # (steps, robots, 2)
    _, num_robots, _ = xy.shape
    if num_robots <= 1:
        return float("inf")

    d_min = float("inf")
    for i, j in itertools.combinations(range(num_robots), 2):
        d = np.linalg.norm(xy[:, i, :] - xy[:, j, :], axis=1)
        d_min = min(d_min, float(np.min(d)))
    return d_min

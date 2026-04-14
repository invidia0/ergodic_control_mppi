from __future__ import annotations

import itertools

import numpy as np


def compute_safety_metric(
    robot_paths: np.ndarray,
    obstacle_map: np.ndarray,
    safety_radius: float,
) -> float:
    """
    Lower is better.
    Returns mean clearance violation against other robots and obstacles.
    """
    paths = np.asarray(robot_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("robot_paths must have shape (steps, robots, state_dim)")
    if safety_radius < 0.0:
        raise ValueError("safety_radius must be >= 0")

    xy = paths[..., :2]  # (steps, robots, 2)
    steps, num_robots, _ = xy.shape
    violations: list[float] = []

    if num_robots > 1:
        for i, j in itertools.combinations(range(num_robots), 2):
            d = np.linalg.norm(xy[:, i, :] - xy[:, j, :], axis=1)
            v = np.maximum(0.0, safety_radius - d)
            violations.append(float(np.mean(v)))

    obs = np.asarray(obstacle_map, dtype=np.float64)
    if obs.size > 0:
        for i in range(num_robots):
            pos = xy[:, i, :]  # (steps, 2)
            d_center = np.linalg.norm(
                pos[:, None, :] - obs[None, :, :2],
                axis=2,
            )
            d_surface = d_center - obs[None, :, 2]
            v = np.maximum(0.0, safety_radius - d_surface)
            violations.append(float(np.mean(v)))

    if not violations:
        return 0.0
    return float(np.mean(violations))


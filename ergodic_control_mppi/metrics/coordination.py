"""Robot overlap, spacing, redundancy, and obstacle-safety metrics."""

import itertools

import numpy as np


def _occupancy(paths, robot, x_limits, y_limits, bins, binary=False):
    histogram, _, _ = np.histogram2d(
        paths[:, robot, 1],
        paths[:, robot, 0],
        bins=bins,
        range=[y_limits, x_limits],
    )
    if binary:
        return (histogram > 0).astype(np.float64)
    total = histogram.sum()
    return histogram / total if total else np.zeros_like(histogram)


def _paths(robot_paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(robot_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("robot_paths must have shape (steps, robots, state_dim)")
    return paths


def compute_pairwise_overlap(
    robot_paths: np.ndarray,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] = (40, 40),
) -> float:
    """Return mean pairwise occupancy intersection in ``[0, 1]``."""
    paths = _paths(robot_paths)
    if paths.shape[1] <= 1:
        return 0.0
    occupancy = [_occupancy(paths, i, map_x_limits, map_y_limits, bins) for i in range(paths.shape[1])]
    return float(np.mean([
        np.minimum(occupancy[i], occupancy[j]).sum()
        for i, j in itertools.combinations(range(paths.shape[1]), 2)
    ]))


def compute_pairwise_redundancy(robot_paths: np.ndarray, d_thresh: float = 1.0) -> float:
    """Return the mean count of robot pairs closer than ``d_thresh``."""
    paths = _paths(robot_paths)
    if d_thresh < 0:
        raise ValueError("d_thresh must be >= 0")
    pairs = list(itertools.combinations(range(paths.shape[1]), 2))
    if not pairs:
        return 0.0
    return float(np.mean(np.sum([
        np.linalg.norm(paths[:, i, :2] - paths[:, j, :2], axis=1) < d_thresh
        for i, j in pairs
    ], axis=0)))


def compute_pairwise_min_distance(robot_paths: np.ndarray) -> float:
    """Return minimum robot-pair distance across all steps."""
    paths = _paths(robot_paths)
    pairs = list(itertools.combinations(range(paths.shape[1]), 2))
    return min(
        (float(np.min(np.linalg.norm(paths[:, i, :2] - paths[:, j, :2], axis=1))) for i, j in pairs),
        default=float("inf"),
    )


def compute_redundancy_metric(
    robot_paths: np.ndarray,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] = (40, 40),
) -> float:
    """Return cell-wise shared-visit redundancy in ``[0, 1]``."""
    paths = _paths(robot_paths)
    robots = paths.shape[1]
    if robots <= 1:
        return 0.0
    visited = np.stack([
        _occupancy(paths, i, map_x_limits, map_y_limits, bins, binary=True)
        for i in range(robots)
    ])
    counts = visited.sum(axis=0)
    covered = counts > 0
    return float(np.mean((counts[covered] - 1) / (robots - 1))) if np.any(covered) else 0.0


def compute_safety_metric(
    robot_paths: np.ndarray,
    obstacle_map: np.ndarray,
    safety_radius: float,
) -> float:
    """Return mean clearance violation against robots and circular obstacles."""
    paths = _paths(robot_paths)
    if safety_radius < 0:
        raise ValueError("safety_radius must be >= 0")
    xy = paths[..., :2]
    violations = [
        float(np.mean(np.maximum(0, safety_radius - np.linalg.norm(xy[:, i] - xy[:, j], axis=1))))
        for i, j in itertools.combinations(range(paths.shape[1]), 2)
    ]
    obstacles = np.asarray(obstacle_map, dtype=np.float64).reshape(-1, 3)
    for robot in range(paths.shape[1]):
        surface_distance = np.linalg.norm(
            xy[:, robot, None, :] - obstacles[None, :, :2], axis=2
        ) - obstacles[None, :, 2]
        if obstacles.size:
            violations.append(float(np.mean(np.maximum(0, safety_radius - surface_distance))))
    return float(np.mean(violations)) if violations else 0.0

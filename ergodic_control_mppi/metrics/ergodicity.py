from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray


def _normalize_density(grid: ArrayLike) -> ArrayLike:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"target density grid must be 2D, got shape {arr.shape}")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("target density grid must have positive total mass")
    return arr / total


def _validate_map_limits(
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
) -> None:
    x_min, x_max = map_x_limits
    y_min, y_max = map_y_limits
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("map limits must satisfy lower < upper on both axes")


def compute_reachable_mask(
    obstacles: ArrayLike,
    safe_distance: float,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> ArrayLike:
    """Boolean grid ``(bins_y, bins_x)``: True where a cell center lies outside
    every obstacle's ``radius + safe_distance``.

    Used to restrict coverage metrics to the space the robot can actually reach,
    so target mass sitting inside an obstacle is not counted as "missed".
    """
    _validate_map_limits(map_x_limits, map_y_limits)
    bins_x, bins_y = bins
    x_min, x_max = map_x_limits
    y_min, y_max = map_y_limits
    xs = x_min + (np.arange(bins_x) + 0.5) * (x_max - x_min) / bins_x
    ys = y_min + (np.arange(bins_y) + 0.5) * (y_max - y_min) / bins_y
    grid_x, grid_y = np.meshgrid(xs, ys)  # (bins_y, bins_x)
    mask = np.ones((bins_y, bins_x), dtype=bool)
    for ox, oy, radius in np.asarray(obstacles, dtype=np.float64).reshape(-1, 3):
        mask &= (grid_x - ox) ** 2 + (grid_y - oy) ** 2 > (radius + safe_distance) ** 2
    return mask


def _restrict_to_mask(grid: ArrayLike, mask: ArrayLike | None) -> ArrayLike:
    """Zero cells outside ``mask`` and renormalize to unit mass over the rest."""
    if mask is None:
        return grid
    restricted = grid * mask
    total = float(restricted.sum())
    return restricted / total if total > 0.0 else restricted


def _as_team_paths(robot_paths: ArrayLike) -> ArrayLike:
    """
    Convert robot paths to shape (steps, robots, state_dim).
    """
    arr = np.asarray(robot_paths, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(
        "robot_paths must have shape (steps, state_dim) or (steps, robots, state_dim)"
    )


def _bin_points_2d(
    xy: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Bin 2D points into integer cell indices.

    Returns:
      ix, iy, valid
    where:
      - ix has shape (n_points,) in [0, bins_x - 1]
      - iy has shape (n_points,) in [0, bins_y - 1]
      - valid marks points inside map bounds

    Points exactly on the upper boundary are assigned to the last bin.
    """
    _validate_map_limits(map_x_limits, map_y_limits)

    bins_x, bins_y = bins
    if bins_x <= 0 or bins_y <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"xy must have shape (n_points, >=2), got {pts.shape}")

    x_min, x_max = map_x_limits
    y_min, y_max = map_y_limits
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)

    xs = pts[:, 0]
    ys = pts[:, 1]

    valid = (
        (xs >= x_min) & (xs <= x_max) &
        (ys >= y_min) & (ys <= y_max)
    )

    ix = np.floor((xs - x_min) / x_span * bins_x).astype(np.int64)
    iy = np.floor((ys - y_min) / y_span * bins_y).astype(np.int64)
    ix = np.clip(ix, 0, bins_x - 1)
    iy = np.clip(iy, 0, bins_y - 1)
    return ix, iy, valid


def _team_occupancy_grid(
    robot_paths: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> ArrayLike:
    """
    Returns normalized team occupancy histogram with shape (bins_y, bins_x).
    bins is ordered as (bins_x, bins_y).
    """
    team_paths = _as_team_paths(robot_paths)
    xy = team_paths[..., :2].reshape(-1, 2)

    bins_x, bins_y = bins
    counts = np.zeros((bins_y, bins_x), dtype=np.float64)

    ix, iy, valid = _bin_points_2d(xy, map_x_limits, map_y_limits, bins)
    if np.any(valid):
        np.add.at(counts, (iy[valid], ix[valid]), 1.0)

    total = float(counts.sum())
    if total <= 0.0:
        return counts
    return counts / total


def compute_team_occupancy_grid(
    robot_paths: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> ArrayLike:
    """
    Public wrapper returning normalized occupancy grid of shape (bins_y, bins_x).
    """
    return _team_occupancy_grid(robot_paths, map_x_limits, map_y_limits, bins)


def compute_team_ergodic_error(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
    reachable_mask: ArrayLike | None = None,
) -> float:
    """
    Compute the canonical scalar occupancy-based ergodic error.
    Lower is better.

    If ``reachable_mask`` (a ``(bins_y, bins_x)`` boolean grid from
    :func:`compute_reachable_mask`) is given, the target and occupancy are
    restricted to reachable cells and renormalized, so target mass inside
    obstacles is not counted as uncovered.
    """
    target = _restrict_to_mask(_normalize_density(target_density_grid), reachable_mask)
    grid_bins = bins if bins is not None else (target.shape[1], target.shape[0])
    occupancy = _team_occupancy_grid(robot_paths, map_x_limits, map_y_limits, grid_bins)

    if occupancy.shape != target.shape:
        raise ValueError(
            f"occupancy grid shape {occupancy.shape} does not match target shape {target.shape}"
        )

    diff = _restrict_to_mask(occupancy, reachable_mask) - target
    return float(np.mean(diff * diff))


def compute_cumulative_team_ergodic_error(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
    reachable_mask: ArrayLike | None = None,
) -> ArrayLike:
    """
    Cumulative occupancy MSE over time.

    For each step t, the occupancy uses samples from steps [0, t]. If
    ``reachable_mask`` is given, target and occupancy are restricted to reachable
    cells and renormalized (see :func:`compute_team_ergodic_error`).
    """
    team_paths = _as_team_paths(robot_paths)
    target = _restrict_to_mask(_normalize_density(target_density_grid), reachable_mask)
    grid_bins = bins if bins is not None else (target.shape[1], target.shape[0])

    bins_x, bins_y = grid_bins
    if target.shape != (bins_y, bins_x):
        raise ValueError(
            f"target density shape {target.shape} does not match bins-derived shape {(bins_y, bins_x)}"
        )

    _validate_map_limits(map_x_limits, map_y_limits)

    steps, _, _ = team_paths.shape
    counts = np.zeros((bins_y, bins_x), dtype=np.float64)
    total = 0.0
    series = np.zeros((steps,), dtype=np.float64)

    for t in range(steps):
        xy_t = team_paths[t, :, :2]
        ix, iy, valid = _bin_points_2d(xy_t, map_x_limits, map_y_limits, grid_bins)

        if np.any(valid):
            np.add.at(counts, (iy[valid], ix[valid]), 1.0)
            total += float(np.sum(valid))

        occupancy = counts / total if total > 0.0 else np.zeros_like(counts)
        diff = _restrict_to_mask(occupancy, reachable_mask) - target
        series[t] = float(np.mean(diff * diff))

    return series


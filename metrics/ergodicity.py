from __future__ import annotations

from typing import Iterable

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


def compute_team_occupancy_mse(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
) -> float:
    """
    Canonical scalar occupancy-based ergodicity proxy for experiments.
    Lower is better.
    """
    target = _normalize_density(target_density_grid)
    grid_bins = bins if bins is not None else (target.shape[1], target.shape[0])
    occupancy = _team_occupancy_grid(robot_paths, map_x_limits, map_y_limits, grid_bins)

    if occupancy.shape != target.shape:
        raise ValueError(
            f"occupancy grid shape {occupancy.shape} does not match target shape {target.shape}"
        )

    diff = occupancy - target
    return float(np.mean(diff * diff))


def compute_team_ergodic_error(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
) -> float:
    """
    Backward-compatible alias for the canonical occupancy MSE metric.
    """
    return compute_team_occupancy_mse(
        robot_paths,
        target_density_grid,
        map_x_limits,
        map_y_limits,
        bins=bins,
    )


def compute_cumulative_team_occupancy_mse(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
) -> ArrayLike:
    """
    Cumulative occupancy MSE over time.

    For each step t, the occupancy uses samples from steps [0, t].
    """
    team_paths = _as_team_paths(robot_paths)
    target = _normalize_density(target_density_grid)
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
        diff = occupancy - target
        series[t] = float(np.mean(diff * diff))

    return series


def compute_cumulative_team_ergodic_error(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
) -> ArrayLike:
    """
    Backward-compatible alias for the cumulative occupancy MSE series.
    """
    return compute_cumulative_team_occupancy_mse(
        robot_paths,
        target_density_grid,
        map_x_limits,
        map_y_limits,
        bins=bins,
    )


def _box_kernel_2d(radius_cells: int) -> ArrayLike:
    """
    Simple square kernel for optional multiscale smoothing proxies.
    """
    if radius_cells < 0:
        raise ValueError("radius_cells must be >= 0")
    size = 2 * radius_cells + 1
    kernel = np.ones((size, size), dtype=np.float64)
    kernel /= float(kernel.sum())
    return kernel


def _convolve2d_same(image: ArrayLike, kernel: ArrayLike) -> ArrayLike:
    """
    Small NumPy-only 2D convolution with zero padding.
    """
    img = np.asarray(image, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    if img.ndim != 2 or ker.ndim != 2:
        raise ValueError("image and kernel must both be 2D")

    kh, kw = ker.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(window * ker)

    return out


def compute_team_multiscale_ergodic_proxy(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
    radii_cells: Iterable[int] = (0, 1, 2, 4),
) -> float:
    """
    Optional scalar multiscale smoothing proxy for visualization only.

    This is not the canonical experiment metric in this repository.
    """
    target = _normalize_density(target_density_grid)
    grid_bins = bins if bins is not None else (target.shape[1], target.shape[0])
    occupancy = _team_occupancy_grid(robot_paths, map_x_limits, map_y_limits, grid_bins)

    if occupancy.shape != target.shape:
        raise ValueError(
            f"occupancy grid shape {occupancy.shape} does not match target shape {target.shape}"
        )

    radii = list(radii_cells)
    if len(radii) == 0:
        raise ValueError("radii_cells must contain at least one radius")

    errors = []
    for radius in radii:
        kernel = _box_kernel_2d(int(radius))
        occ_s = _convolve2d_same(occupancy, kernel)
        tgt_s = _convolve2d_same(target, kernel)
        diff = occ_s - tgt_s
        errors.append(float(np.mean(diff * diff)))

    return float(np.mean(errors))


def compute_cumulative_team_multiscale_ergodic_proxy(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    bins: tuple[int, int] | None = None,
    radii_cells: Iterable[int] = (0, 1, 2, 4),
) -> ArrayLike:
    """
    Optional cumulative multiscale proxy for plotting convergence.
    """
    team_paths = _as_team_paths(robot_paths)
    steps = team_paths.shape[0]
    series = np.zeros((steps,), dtype=np.float64)

    for t in range(steps):
        series[t] = compute_team_multiscale_ergodic_proxy(
            team_paths[: t + 1],
            target_density_grid,
            map_x_limits,
            map_y_limits,
            bins=bins,
            radii_cells=radii_cells,
        )

    return series

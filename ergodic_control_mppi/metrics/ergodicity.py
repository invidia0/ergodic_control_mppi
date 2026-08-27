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
    stride: int = 1,
) -> ArrayLike:
    """
    Cumulative occupancy MSE over time.

    For each step t, the occupancy uses samples from steps [0, t]. If
    ``reachable_mask`` is given, target and occupancy are restricted to reachable
    cells and renormalized (see :func:`compute_team_ergodic_error`).

    ``stride`` subsamples the returned series without subsampling the histogram:
    every step still contributes to the occupancy, only the O(bins) error
    evaluation is skipped. The last entry therefore always equals
    :func:`compute_team_ergodic_error` on the same path, which subsampling the
    path beforehand would not.
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

    if stride < 1:
        raise ValueError("stride must be >= 1")

    steps, _, _ = team_paths.shape
    counts = np.zeros((bins_y, bins_x), dtype=np.float64)
    total = 0.0
    # Anchored on the final step so the series always ends at the scalar metric.
    report = set(range(steps - 1, -1, -stride))
    series = np.zeros((len(report),), dtype=np.float64)

    written = 0
    for t in range(steps):
        xy_t = team_paths[t, :, :2]
        ix, iy, valid = _bin_points_2d(xy_t, map_x_limits, map_y_limits, grid_bins)

        if np.any(valid):
            np.add.at(counts, (iy[valid], ix[valid]), 1.0)
            total += float(np.sum(valid))

        if t not in report:
            continue
        occupancy = counts / total if total > 0.0 else np.zeros_like(counts)
        diff = _restrict_to_mask(occupancy, reachable_mask) - target
        series[written] = float(np.mean(diff * diff))
        written += 1

    return series



# --- Fourier (spectral) ergodic metric -------------------------------------
#
# The metric of Mathew & Mezic / Ivic et al.: compare the trajectory's
# time-averaged Fourier coefficients against the target's spatial ones under a
# Sobolev-like weighting that discounts high wavenumbers,
#
#     eps = sum_k Lambda_k (c_k - phi_k)^2,   Lambda_k = (1 + ||k||^2)^-(d+1)/2,
#
# with the cosine basis on the box. This is the metric the Experiments section
# reports, so ablation numbers stay commensurable with the baselines. The basis
# helpers live here rather than in experiments/ because both consume them.


def fourier_wavenumbers(order: int) -> tuple[ArrayLike, ArrayLike]:
    """Return the ``(k, Lambda_k)`` pair for all modes up to ``order``.

    Excludes the ``(0, 0)`` mode, whose coefficient is identically one for any
    normalized measure and therefore carries no information.

    Raises:
        ValueError: If ``order`` is below one.
    """
    if order < 1:
        raise ValueError("fourier order must be >= 1")
    k_list = [
        (float(kx), float(ky))
        for kx in range(order + 1)
        for ky in range(order + 1)
        if not (kx == 0 and ky == 0)
    ]
    k_arr = np.asarray(k_list, dtype=np.float64)
    lambda_k = np.power(1.0 + np.sum(k_arr * k_arr, axis=1), -1.5)
    return k_arr, lambda_k


def fourier_basis_values(
    xy: ArrayLike,
    k_arr: ArrayLike,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> ArrayLike:
    """Evaluate the cosine basis at ``(n, >=2)`` points, returning ``(n, n_k)``.

    Unnormalized (the ``h_k`` factor is omitted): it is a fixed per-mode scale
    that cancels in the ``c_k - phi_k`` difference up to a constant absorbed by
    ``Lambda_k``, and both terms here use the same convention.
    """
    pts = np.asarray(xy, dtype=np.float64)
    kx = np.asarray(k_arr, dtype=np.float64)[:, 0][None, :]
    ky = np.asarray(k_arr, dtype=np.float64)[:, 1][None, :]
    x_span = max(float(x_max - x_min), 1e-12)
    y_span = max(float(y_max - y_min), 1e-12)
    xn = (pts[:, 0:1] - x_min) / x_span
    yn = (pts[:, 1:2] - y_min) / y_span
    return np.cos(np.pi * xn * kx) * np.cos(np.pi * yn * ky)


def fourier_target_coefficients(
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    k_arr: ArrayLike,
    reachable_mask: ArrayLike | None = None,
) -> ArrayLike:
    """Spatial coefficients ``phi_k`` of the target grid, shape ``(n_k,)``.

    If ``reachable_mask`` is given the target is restricted and renormalized
    first, exactly as :func:`compute_team_ergodic_error` does, so the two
    metrics score against the same reference measure.
    """
    _validate_map_limits(map_x_limits, map_y_limits)
    target = _restrict_to_mask(_normalize_density(target_density_grid), reachable_mask)
    ny, nx = target.shape
    xs = np.linspace(map_x_limits[0], map_x_limits[1], nx, dtype=np.float64)
    ys = np.linspace(map_y_limits[0], map_y_limits[1], ny, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
    values = fourier_basis_values(
        points, k_arr, map_x_limits[0], map_x_limits[1], map_y_limits[0], map_y_limits[1]
    )
    return values.T @ target.reshape(-1)


def compute_fourier_ergodic_metric(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    order: int = 10,
    reachable_mask: ArrayLike | None = None,
) -> float:
    """Compute the scalar spectral ergodic metric. Lower is better."""
    series = compute_cumulative_fourier_ergodic_metric(
        robot_paths,
        target_density_grid,
        map_x_limits,
        map_y_limits,
        order=order,
        reachable_mask=reachable_mask,
    )
    return float(series[-1])


def compute_cumulative_fourier_ergodic_metric(
    robot_paths: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    order: int = 10,
    reachable_mask: ArrayLike | None = None,
    stride: int = 1,
) -> ArrayLike:
    """Spectral ergodic metric after each step, shape ``(ceil(steps/stride),)``.

    Vectorized: the running coefficients are one ``cumsum`` over the per-step
    basis values, so unlike the occupancy series this costs no Python loop.
    ``stride`` subsamples the *output* only; every step still contributes to the
    time average.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    team_paths = _as_team_paths(robot_paths)
    xy = team_paths[..., :2]
    steps, robots, _ = xy.shape

    k_arr, lambda_k = fourier_wavenumbers(order)
    phi_k = fourier_target_coefficients(
        target_density_grid, map_x_limits, map_y_limits, k_arr, reachable_mask
    )
    values = fourier_basis_values(
        xy.reshape(-1, 2),
        k_arr,
        map_x_limits[0],
        map_x_limits[1],
        map_y_limits[0],
        map_y_limits[1],
    ).reshape(steps, robots, -1)

    # Time-averaged coefficients after each step, averaged over robots too.
    running = np.cumsum(values.sum(axis=1), axis=0)
    counts = (np.arange(steps, dtype=np.float64) + 1.0) * robots
    coefficients = running / counts[:, None]

    index = np.arange(steps - 1, -1, -stride, dtype=np.int64)[::-1]
    difference = coefficients[index] - phi_k[None, :]
    return np.sum(lambda_k[None, :] * difference * difference, axis=1)


# --- Ball (multiscale) ergodic metric --------------------------------------
#
# The metric the analysis of Sec. "guarantees" is actually stated in:
#
#     E_K = int_0^R int_Omega [ rho_K(B(x,r)) - p*(B(x,r)) ]^2 dx dr,
#
# the squared disagreement in the mass the two measures assign to every ball,
# integrated over centres and radii. The Fourier metric above is what the
# Experiments section reports and what the baselines are scored on; this one
# exists so the bounds of Prop. "ergodic_error_decomposition" can be evaluated
# on their own left-hand side rather than on a proxy. They are not
# interchangeable and are never pooled.
#
# rho_K(B(x,r)) is a convolution of the mass grid with a disc indicator, so the
# whole inner integral is one FFT per radius. Zero-padding is not optional: a
# circular convolution would wrap a ball off the west edge onto the east one and
# silently credit coverage across the workspace.


def _disc_mass_field(
    mass: ArrayLike, radius: float, cell_x: float, cell_y: float
) -> ArrayLike:
    """Mass inside a disc of world-unit ``radius`` centred on each cell, via zero-padded FFT.

    The disc is built in world units rather than in cells, so a grid whose cells are not
    exactly square -- the deployment's 267x134 raster over a 40x20 workspace is square only
    to 0.4% -- still gets a circle rather than an ellipse, at any aspect ratio.
    """
    height, width = mass.shape
    pad_y, pad_x = height * 2, width * 2
    offsets_y = np.fft.fftfreq(pad_y, d=1.0 / pad_y) * cell_y
    offsets_x = np.fft.fftfreq(pad_x, d=1.0 / pad_x) * cell_x
    disc = (
        offsets_y[:, None] ** 2 + offsets_x[None, :] ** 2 <= radius * radius
    ).astype(np.float64)
    spectrum = np.fft.rfft2(mass, s=(pad_y, pad_x)) * np.fft.rfft2(disc)
    return np.fft.irfft2(spectrum, s=(pad_y, pad_x))[:height, :width]


def compute_ball_ergodic_metric(
    occupancy: ArrayLike,
    target_density_grid: ArrayLike,
    map_x_limits: tuple[float, float],
    map_y_limits: tuple[float, float],
    max_radius: float | None = None,
    radii: int = 24,
    reachable_mask: ArrayLike | None = None,
) -> float:
    """Multiscale ball ergodic metric between an occupancy grid and a target. Lower is better.

    Both inputs are treated as *mass* grids (each renormalized to unit total), so the
    returned value carries units of area squared times length and is comparable only
    between calls sharing a workspace, ``max_radius`` and ``radii``.

    Args:
        occupancy: Visitation mass with shape ``(bins_y, bins_x)``, e.g. from
            :func:`compute_team_occupancy_grid`.
        target_density_grid: Target on the same grid; renormalized here.
        map_x_limits: Workspace ``x`` bounds.
        map_y_limits: Workspace ``y`` bounds.
        max_radius: Largest ball radius ``R``; defaults to half the shorter workspace side,
            beyond which every ball covers the whole workspace and contributes nothing.
        radii: Number of midpoint quadrature nodes on ``[0, R]``.
        reachable_mask: Optional mask restricting both measures to reachable cells.

    Returns:
        The scalar metric.
    """
    if radii < 1:
        raise ValueError("radii must be >= 1")
    _validate_map_limits(map_x_limits, map_y_limits)

    target = _restrict_to_mask(_normalize_density(target_density_grid), reachable_mask)
    visited = _restrict_to_mask(_normalize_density(occupancy), reachable_mask)
    if visited.shape != target.shape:
        raise ValueError(
            f"occupancy grid shape {visited.shape} does not match target shape {target.shape}"
        )

    height, width = target.shape
    cell_x = (map_x_limits[1] - map_x_limits[0]) / width
    cell_y = (map_y_limits[1] - map_y_limits[0]) / height
    span = min(map_x_limits[1] - map_x_limits[0], map_y_limits[1] - map_y_limits[0])
    radius = 0.5 * span if max_radius is None else float(max_radius)
    if radius <= 0.0:
        raise ValueError("max_radius must be positive")

    difference = visited - target
    cell_area = cell_x * cell_y
    step = radius / radii
    nodes = (np.arange(radii, dtype=np.float64) + 0.5) * step
    total = 0.0
    for node in nodes:
        residual = _disc_mass_field(difference, node, cell_x, cell_y)
        total += float(np.sum(residual * residual)) * cell_area * step
    return total

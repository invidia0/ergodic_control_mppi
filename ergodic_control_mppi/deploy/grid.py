"""
Occupancy-grid construction and queries for the UAV deployment.

Every query works on a planar grid ``(H, W)`` or a voxel grid ``(Z, H, W)``: world points are
``(x, y)`` or ``(x, y, z)``, and cells are indexed with the axes reversed.
"""

import itertools
from collections import deque

import numpy as np


def inflation_radius(
    robot_radius: float,
    clearance: float,
    tracking_allowance: float,
    max_speed: float,
    brake_accel: float,
    reaction_time: float,
    resolution: float,
    dimension: int = 2,
) -> float:
    """
    Return the radius obstacles must be grown by before planning against them.

    Args:
            robot_radius: Cylindrical footprint radius in metres.
            clearance: Additional discretionary margin in metres.
            tracking_allowance: Expected setpoint-tracking error in metres. Calibrate this
                from the measured position error rather than trusting the default.
            max_speed: Speed cap enforced by the shield, in metres per second.
            brake_accel: Deceleration the shield commands, in metres per second squared.
            reaction_time: Delay before braking takes effect, in seconds.
            resolution: Grid cell size in metres.
            dimension: Grid dimension; the cell half-diagonal grows with it.
    Returns:
            The inflation radius in metres.
    """
    if brake_accel <= 0.0:
        raise ValueError("brake_accel must be positive")
    discretization = 0.5 * np.sqrt(dimension) * resolution
    stopping = max_speed * reaction_time + max_speed * max_speed / (2.0 * brake_accel)
    return float(robot_radius + clearance + discretization + tracking_allowance + stopping)


def inflate(occupancy: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    """
    Grow occupied cells by a disk (or ball) of the given radius.

    Args:
            occupancy: Boolean grid with shape ``(H, W)`` or ``(Z, H, W)``.
            radius: Inflation radius in metres.
            resolution: Grid cell size in metres.
    Returns:
            A new boolean grid, never smaller than the input.
    """
    cells = int(np.ceil(radius / resolution))
    if cells <= 0:
        return occupancy.copy()
    inflated = occupancy.copy()
    # ponytail: shift-and-OR dilation, ~250 whole-array ORs at the shipped planar radius and a
    # few hundred in 3D at coarse voxels. Swap for a distance transform if the grid outgrows it.
    for offset in itertools.product(range(-cells, cells + 1), repeat=occupancy.ndim):
        if sum(step * step for step in offset) > cells * cells:
            continue
        destination = tuple(
            slice(max(0, step), size + min(0, step)) for step, size in zip(offset, occupancy.shape)
        )
        source = tuple(
            slice(max(0, -step), size - max(0, step)) for step, size in zip(offset, occupancy.shape)
        )
        inflated[destination] |= occupancy[source]
    return inflated


def world_to_cell(
    positions: np.ndarray, origin: tuple[float, ...], resolution: float
) -> np.ndarray:
    """Return cell indices of positions ``(..., d)``, axes reversed: ``(row, column)`` or
    ``(layer, row, column)``."""
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.floor((positions - np.asarray(origin, dtype=np.float64)) / resolution)
    return cell.astype(np.int64)[..., ::-1]


def _inside(grid: np.ndarray, cell: tuple[int, ...]) -> bool:
    return all(0 <= index < size for index, size in zip(cell, grid.shape))


def _cell(position, origin: tuple[float, ...], resolution: float) -> tuple[int, ...]:
    return tuple(int(index) for index in world_to_cell(np.asarray(position), origin, resolution))


def is_free(grid: np.ndarray, origin: tuple[float, ...], resolution: float, position) -> bool:
    """Whether ``position`` falls on a free cell of ``grid``; outside the grid is not free."""
    cell = _cell(position, origin, resolution)
    return _inside(grid, cell) and not grid[cell]


def entry_cell(
    grid: np.ndarray, origin: tuple[float, ...], resolution: float, start: tuple[float, ...]
) -> tuple[int, ...] | None:
    """Return the cell the vehicle first occupies, or ``None`` if it never can."""
    cell = _cell(start, origin, resolution)
    if _inside(grid, cell):
        return None if grid[cell] else cell
    nearest = nearest_free(grid, origin, resolution, start)
    if nearest is None:
        return None
    cell = _cell(nearest, origin, resolution)
    return cell if _inside(grid, cell) else None


def reachable_from(
    grid: np.ndarray, origin: tuple[float, ...], resolution: float, start: tuple[float, ...]
) -> np.ndarray:
    """
    Flood-fill the free cells face-connected to ``start``.

    Returns:
            Boolean mask with the shape of ``grid``.
    """
    visited = np.zeros(grid.shape, dtype=bool)
    cell = entry_cell(grid, origin, resolution, start)
    if cell is None:
        return visited
    visited[cell] = True
    queue = deque([cell])
    steps = [
        tuple(sign if axis == moved else 0 for axis in range(grid.ndim))
        for moved in range(grid.ndim)
        for sign in (1, -1)
    ]
    while queue:
        cell = queue.popleft()
        for step in steps:
            neighbour = tuple(index + delta for index, delta in zip(cell, step))
            if _inside(grid, neighbour) and not grid[neighbour] and not visited[neighbour]:
                visited[neighbour] = True
                queue.append(neighbour)
    return visited


def all_reachable(
    grid: np.ndarray,
    origin: tuple[float, ...],
    resolution: float,
    start: tuple[float, ...],
    targets: np.ndarray,
) -> tuple[bool, np.ndarray, dict]:
    """
    Check that every target is free and connected to ``start``.

    Args:
            grid: Inflated boolean occupancy.
            origin: World coordinates of the lowest grid corner.
            resolution: Grid cell size in metres.
            start: Arming position.
            targets: Positions that must be reachable, with shape ``(M, d)``.
    Returns:
            Whether every target is reachable, the per-target boolean mask, and a diagnosis
            mapping naming *why* it failed: whether the start itself is blocked, how large the
            reachable component is against the total free space, which modes are blocked
            outright versus merely cut off, and the reachable component itself.
    """
    visited = reachable_from(grid, origin, resolution, start)
    cells = [
        tuple(int(index) for index in cell)
        for cell in world_to_cell(
            np.asarray(targets, dtype=np.float64).reshape(-1, grid.ndim), origin, resolution
        )
    ]
    flags = np.array([_inside(grid, cell) and bool(visited[cell]) for cell in cells], dtype=bool)
    blocked = np.array([not _inside(grid, cell) or bool(grid[cell]) for cell in cells], dtype=bool)
    diagnosis = {
        # Outside the grid is not blocked: the vehicle flies in. Only a start sitting on an
        # obstacle has nowhere to begin.
        "start_blocked": bool(entry_cell(grid, origin, resolution, start) is None),
        "start_outside": not _inside(grid, _cell(start, origin, resolution)),
        "component_cells": int(visited.sum()),
        "free_cells": int((~grid).sum()),
        # A mode that is itself inside an obstacle is a different problem from one that is
        # merely cut off, and needs a different fix, so name them apart.
        "blocked_modes": [index for index, flag in enumerate(blocked) if flag],
        "disconnected_modes": [
            index for index, (ok, hit) in enumerate(zip(flags, blocked)) if not ok and not hit
        ],
        "component": visited,
    }
    return bool(flags.all()) and bool(visited.any()), flags, diagnosis


def nearest_free(
    grid: np.ndarray,
    origin: tuple[float, ...],
    resolution: float,
    position: tuple[float, ...],
) -> tuple[float, ...] | None:
    """
    Return the centre of the free cell closest to ``position``, or ``None`` if none.
    """
    free = np.argwhere(~grid)
    if free.size == 0:
        return None
    centres = np.asarray(origin, dtype=np.float64) + (free[:, ::-1] + 0.5) * resolution
    best = int(np.argmin(np.linalg.norm(centres - np.asarray(position, dtype=np.float64), axis=1)))
    return tuple(float(value) for value in centres[best])


def segment_blocked(
    grid: np.ndarray,
    origin: tuple[float, ...],
    resolution: float,
    start: np.ndarray,
    end: np.ndarray,
) -> bool:
    """
    Whether the straight segment from ``start`` to ``end`` touches a blocked cell.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / (0.5 * resolution))) + 1)
    samples = start + np.linspace(0.0, 1.0, count)[:, None] * (end - start)
    cells = world_to_cell(samples, origin, resolution)
    if ((cells < 0) | (cells >= np.asarray(grid.shape))).any():
        return True
    return bool(grid[tuple(cells.T)].any())


def path_blocked(
    grid: np.ndarray, origin: tuple[float, ...], resolution: float, positions: np.ndarray
) -> bool:
    """Whether any consecutive segment of a polyline touches a blocked cell."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, grid.ndim)
    if positions.shape[0] == 0:
        return True
    if positions.shape[0] == 1:
        return segment_blocked(grid, origin, resolution, positions[0], positions[0])
    return any(
        segment_blocked(grid, origin, resolution, positions[index], positions[index + 1])
        for index in range(positions.shape[0] - 1)
    )

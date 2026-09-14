"""
Occupancy-grid construction and queries for the fixed-altitude UAV deployment.
"""

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
    Returns:
            The inflation radius in metres.
    """
    if brake_accel <= 0.0:
        raise ValueError("brake_accel must be positive")
    discretization = 0.5 * np.sqrt(2.0) * resolution
    stopping = max_speed * reaction_time + max_speed * max_speed / (2.0 * brake_accel)
    return float(robot_radius + clearance + discretization + tracking_allowance + stopping)


def inflate(occupancy: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    """
    Grow occupied cells by a disk of the given radius.
    
    Args:
            occupancy: Boolean grid with shape ``(H, W)``.
            radius: Inflation radius in metres.
            resolution: Grid cell size in metres.
    Returns:
            A new boolean grid, never smaller than the input.
    """
    cells = int(np.ceil(radius / resolution))
    if cells <= 0:
        return occupancy.copy()
    height, width = occupancy.shape
    inflated = occupancy.copy()
    # ponytail: shift-and-OR disk dilation, ~250 whole-array ORs at the shipped radius.
    # Swap for a chamfer distance transform only if the grid outgrows a few hundred cells.
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > cells * cells:
                continue
            destination = (
                slice(max(0, dy), height + min(0, dy)),
                slice(max(0, dx), width + min(0, dx)),
            )
            source = (
                slice(max(0, -dy), height - max(0, dy)),
                slice(max(0, -dx), width - max(0, dx)),
            )
            inflated[destination] |= occupancy[source]
    return inflated


def world_to_cell(
    positions: np.ndarray, origin: tuple[float, float], resolution: float
) -> np.ndarray:
    """Return ``(row, column)`` indices for world positions with shape ``(..., 2)``."""
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.floor((positions - np.asarray(origin, dtype=np.float64)) / resolution)
    return cell.astype(np.int64)[..., ::-1]


def _inside(grid: np.ndarray, row: int, column: int) -> bool:
    return 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]


def entry_cell(
    grid: np.ndarray, origin: tuple[float, float], resolution: float, start: tuple[float, float]
) -> tuple[int, int] | None:
    """Return the cell the vehicle first occupies, or ``None`` if it never can."""
    row, column = (int(index) for index in world_to_cell(np.asarray(start), origin, resolution))
    if _inside(grid, row, column):
        return None if grid[row, column] else (row, column)
    nearest = nearest_free(grid, origin, resolution, start)
    if nearest is None:
        return None
    row, column = (int(index) for index in world_to_cell(np.asarray(nearest), origin, resolution))
    return (row, column) if _inside(grid, row, column) else None


def reachable_from(
    grid: np.ndarray, origin: tuple[float, float], resolution: float, start: tuple[float, float]
) -> np.ndarray:
    """
    Flood-fill the free cells four-connected to ``start``.
    
    Returns:
            Boolean mask with the shape of ``grid``.
    """
    visited = np.zeros(grid.shape, dtype=bool)
    cell = entry_cell(grid, origin, resolution, start)
    if cell is None:
        return visited
    row, column = cell
    visited[row, column] = True
    queue = deque([(row, column)])
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in (
            (row + 1, column),
            (row - 1, column),
            (row, column + 1),
            (row, column - 1),
        ):
            if (
                _inside(grid, next_row, next_column)
                and not grid[next_row, next_column]
                and not visited[next_row, next_column]
            ):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
    return visited


def all_reachable(
    grid: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    start: tuple[float, float],
    targets: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """
    Check that every target is free and connected to ``start``.
    
    Args:
            grid: Inflated boolean occupancy.
            origin: World coordinates of the lower-left grid corner.
            resolution: Grid cell size in metres.
            start: Arming position.
            targets: Positions that must be reachable, with shape ``(M, 2)``.
    Returns:
            Whether every target is reachable, the per-target boolean mask, and a diagnosis
            mapping naming *why* it failed: whether the start itself is blocked, how large the
            reachable component is against the total free space, and which modes are blocked
            outright versus merely cut off.
    """
    visited = reachable_from(grid, origin, resolution, start)
    cells = world_to_cell(np.asarray(targets, dtype=np.float64).reshape(-1, 2), origin, resolution)
    flags = np.array(
        [
            bool(_inside(grid, int(row), int(column)) and visited[int(row), int(column)])
            for row, column in cells
        ],
        dtype=bool,
    )
    blocked = np.array(
        [
            bool(not _inside(grid, int(row), int(column)) or grid[int(row), int(column)])
            for row, column in cells
        ],
        dtype=bool,
    )
    start_row, start_column = (
        int(index) for index in world_to_cell(np.asarray(start), origin, resolution)
    )
    outside = not _inside(grid, start_row, start_column)
    diagnosis = {
        # Outside the grid is not blocked: the vehicle flies in. Only a start sitting on an
        # obstacle has nowhere to begin.
        "start_blocked": bool(
            entry_cell(grid, origin, resolution, start) is None
        ),
        "start_outside": bool(outside),
        "component_cells": int(visited.sum()),
        "free_cells": int((~grid).sum()),
        # A mode that is itself inside an obstacle is a different problem from one that is
        # merely cut off, and needs a different fix, so name them apart.
        "blocked_modes": [index for index, flag in enumerate(blocked) if flag],
        "disconnected_modes": [
            index for index, (ok, hit) in enumerate(zip(flags, blocked)) if not ok and not hit
        ],
    }
    return bool(flags.all()) and bool(visited.any()), flags, diagnosis


def nearest_free(
    grid: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    position: tuple[float, float],
) -> tuple[float, float] | None:
    """
    Return the centre of the free cell closest to ``position``, or ``None`` if none.
    """
    free = np.argwhere(~grid)
    if free.size == 0:
        return None
    centres = np.column_stack(
        (
            origin[0] + (free[:, 1] + 0.5) * resolution,
            origin[1] + (free[:, 0] + 0.5) * resolution,
        )
    )
    best = int(np.argmin(np.linalg.norm(centres - np.asarray(position, dtype=np.float64), axis=1)))
    return (float(centres[best, 0]), float(centres[best, 1]))


def segment_blocked(
    grid: np.ndarray,
    origin: tuple[float, float],
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
    rows, columns = cells[:, 0], cells[:, 1]
    outside = (
        (rows < 0) | (rows >= grid.shape[0]) | (columns < 0) | (columns >= grid.shape[1])
    )
    if outside.any():
        return True
    return bool(grid[rows, columns].any())


def path_blocked(
    grid: np.ndarray, origin: tuple[float, float], resolution: float, positions: np.ndarray
) -> bool:
    """Whether any consecutive segment of a polyline touches a blocked cell."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if positions.shape[0] == 0:
        return True
    if positions.shape[0] == 1:
        return segment_blocked(grid, origin, resolution, positions[0], positions[0])
    return any(
        segment_blocked(grid, origin, resolution, positions[index], positions[index + 1])
        for index in range(positions.shape[0] - 1)
    )

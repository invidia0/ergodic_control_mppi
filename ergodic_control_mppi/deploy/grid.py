"""Occupancy-grid construction and queries for the fixed-altitude UAV deployment.

Pure NumPy so the map adapter, the safety shield, and the paired offline runner share one
implementation, and so the geometry is testable without a ROS environment. The grid is
row-major ``grid[row, column]`` with ``row`` indexing y and ``column`` indexing x, and
``origin`` naming the lower-left corner of cell ``(0, 0)`` -- the same convention as
``nav_msgs/OccupancyGrid`` and as the runtime grid in ``WorkspaceParams``.
"""

from collections import deque
from itertools import combinations

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
    """Return the radius obstacles must be grown by before planning against them.

    The budget is the robot footprint, a discretionary clearance for odometry and map
    error, the worst-case cell quantization, the setpoint-tracking allowance, and the
    distance needed to stop from ``max_speed`` after a ``reaction_time`` delay. The
    stopping term is included in the *planning* grid on purpose: it is what guarantees the
    controller never commands its way into a region the shield could not brake out of.

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


def slice_cloud(points: np.ndarray, altitude: float, half_extent: float) -> np.ndarray:
    """Keep the ``(x, y)`` of points inside the robot's vertical footprint.

    Args:
        points: Cloud with shape ``(N, 3)``.
        altitude: Flight altitude in metres.
        half_extent: Half the vertical footprint height in metres.

    Returns:
        Horizontal positions with shape ``(M, 2)``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inside = np.abs(points[:, 2] - altitude) <= half_extent
    return points[inside, :2]


def rasterize(
    positions: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    """Bin horizontal positions into a boolean occupancy grid covering the workspace."""
    width = int(np.ceil((x_limits[1] - x_limits[0]) / resolution))
    height = int(np.ceil((y_limits[1] - y_limits[0]) / resolution))
    occupancy = np.zeros((height, width), dtype=bool)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if positions.size == 0:
        return occupancy
    column = np.floor((positions[:, 0] - x_limits[0]) / resolution).astype(np.int64)
    row = np.floor((positions[:, 1] - y_limits[0]) / resolution).astype(np.int64)
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    occupancy[row[inside], column[inside]] = True
    return occupancy


def inflate(occupancy: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    """Grow occupied cells by a disk of the given radius.

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
    """Return the cell the vehicle first occupies, or ``None`` if it never can.

    A start *outside* the grid is legitimate: the vehicle flies in from open space, and the
    boundary term in the stage cost pulls it back over the workspace. Connectivity is then
    judged from where it enters, not from a cell that does not exist. A start *inside* the
    grid but on an obstacle is a different matter and has no entry cell.
    """
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
    """Flood-fill the free cells four-connected to ``start``.

    A start outside the grid is seeded from the cell it would enter through, so flying in
    from open space is allowed; a start on an obstacle reaches nothing.

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
    """Check that every target is free and connected to ``start``.

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


def metric_reachable_mask(
    grid: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    start: tuple[float, float],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    bins: tuple[int, int],
) -> np.ndarray:
    """Sample grid reachability onto the coverage-metric grid.

    The circle-based ``compute_reachable_mask`` cannot be used for a deployment run,
    because the deployment carries no circles -- its obstacles are the grid. Reachability
    here is literal: flood-fill the inflated grid from the arming position, then ask
    whether each metric cell centre landed in that component. Cells the robot could never
    occupy are excluded from the coverage error on both the UAV and ideal sides.

    Args:
        grid: Inflated boolean occupancy.
        origin: World coordinates of the lower-left grid corner.
        resolution: Grid cell size in metres.
        start: Arming position.
        x_limits: Workspace x bounds of the metric grid.
        y_limits: Workspace y bounds of the metric grid.
        bins: Metric grid shape as ``(rows, columns)``.

    Returns:
        Boolean mask with shape ``bins``.
    """
    visited = reachable_from(grid, origin, resolution, start)
    rows, columns = bins
    x = np.linspace(x_limits[0], x_limits[1], columns)
    y = np.linspace(y_limits[0], y_limits[1], rows)
    grid_x, grid_y = np.meshgrid(x, y)
    cells = world_to_cell(np.stack((grid_x, grid_y), axis=-1), origin, resolution)
    row = np.clip(cells[..., 0], 0, grid.shape[0] - 1)
    column = np.clip(cells[..., 1], 0, grid.shape[1] - 1)
    return visited[row, column]


def nearest_free(
    grid: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    position: tuple[float, float],
) -> tuple[float, float] | None:
    """Return the centre of the free cell closest to ``position``, or ``None`` if none.

    Serves two purposes: the entry cell for a start outside the grid, and a concrete
    suggestion when a start inside the grid is refused. It never relocates the vehicle --
    the vehicle spawns wherever the launch says.
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
    """Whether the straight segment from ``start`` to ``end`` touches a blocked cell.

    Sampled at half the cell size, so no cell along the segment can be stepped over.
    Positions outside the grid count as blocked: the grid covers the whole workspace, so
    leaving it is already a safety failure.
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


def blocked_mode_segments(
    grid: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    modes: np.ndarray,
) -> int:
    """Count blocked straight segments between pairs of target modes."""
    modes = np.asarray(modes, dtype=np.float64).reshape(-1, 2)
    return sum(
        segment_blocked(grid, origin, resolution, modes[first], modes[second])
        for first, second in combinations(range(modes.shape[0]), 2)
    )


def clearance_along(
    occupancy: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Return each position's distance to the nearest occupied cell centre.

    Uses the *raw* (uninflated) occupancy, so the result is physical map clearance rather
    than a margin-inclusive one, which is what a collision claim has to be based on.
    Positions are all infinitely clear when the map is empty.

    Args:
        occupancy: Boolean grid with shape ``(H, W)``.
        origin: World coordinates of the lower-left grid corner.
        resolution: Grid cell size in metres.
        positions: Query positions with shape ``(N, 2)``.

    Returns:
        Distances with shape ``(N,)``.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    occupied = np.argwhere(occupancy)
    if occupied.size == 0 or positions.size == 0:
        return np.full(positions.shape[0], np.inf)
    centres = np.column_stack(
        (
            origin[0] + (occupied[:, 1] + 0.5) * resolution,
            origin[1] + (occupied[:, 0] + 0.5) * resolution,
        )
    )
    # ponytail: dense (N x M) distance block, chunked over the path. A 20k-step path
    # against a few thousand occupied cells is the realistic worst case; tile over the
    # cells too only if the map gets much denser.
    distances = np.empty(positions.shape[0], dtype=np.float64)
    for begin in range(0, positions.shape[0], 2048):
        block = positions[begin : begin + 2048]
        block_distances = np.linalg.norm(block[:, None, :] - centres[None, :, :], axis=-1)
        distances[begin : begin + 2048] = block_distances.min(axis=1)
    return distances

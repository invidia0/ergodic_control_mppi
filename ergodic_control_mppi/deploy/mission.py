"""
Compile an operator mission spec into controller parameters in PX4's local NED frame.

The paddock draws in WGS84. The drone projects every point around its own EKF origin with
PX4's azimuthal-equidistant map projection, so the controller runs directly in NED x (north)
and y (east). A bearing measured clockwise from north is then the x-to-y angle: there is no
handedness flip, and headings taken from the plan are PX4 yaw as they are.

The command mode picks the dimension. The planar modes fly ``[x, y]`` at the altitude
mullet_core holds, with every obstacle blocking all altitudes. The ``_3d`` modes fly
``[x, y, z]`` with z NED (down), over a voxel grid where an obstacle blocks only up to its
height, so the controller may fly over it below the mission ceiling.
"""

import functools
import math
from dataclasses import replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import gmm_params
from ergodic_control_mppi.deploy.grid import (
    all_reachable,
    inflate,
    inflation_radius,
    path_blocked,
    world_to_cell,
)
from ergodic_control_mppi.mppi.single import SingleControllerState, measured_step, run_single
from ergodic_control_mppi.parameters import ControllerParams

SCHEMA = "ergodic/3"
COMMAND_MODES = (
    "trajectory", "velocity", "acceleration", "trajectory_3d", "velocity_3d", "acceleration_3d",
)
EARTH_RADIUS_M = 6371000.0  # PX4 CONSTANTS_RADIUS_OF_EARTH
# ponytail: fixed grid resolution caps the area near 200 m x 200 m at 0.15 m cells; coarsen
# the resolution for larger fields.
MAX_GRID_CELLS = 2_000_000
# ponytail: coarse cubic voxels keep a 3D mission under MAX_GRID_CELLS (about 100 m x 100 m x
# 30 m); refine them once the Jetson has the memory and the step time to spare.
VOXEL_RESOLUTION_M = 0.5
# ponytail: allowance for the model overshooting the scheduled speed peak; the node clamps
# every command to max_speed regardless, so this only catches a mis-scaled schedule.
SPEED_TOLERANCE = 1.25
# Oldest local position a mission may start or step from; mullet_core holds 0.25 s after the
# last command. Between measurements the controller steps on its own prediction.
STALE_POSITION_S = 0.2
# ponytail: proportional speed cap on the acceleration along the measured velocity, in 1/s per
# m/s of speed margin. Tune on the airframe if it brakes too softly or too hard near the limit.
SPEED_CAP_GAIN = 2.0

_SPEC_KEYS = {
    "schema", "mission_id", "command_mode", "duration_s", "max_altitude_m", "vehicle", "area",
    "obstacles", "density",
}
_VEHICLE_KEYS = {
    "radius_m", "clearance_m", "tracking_allowance_m", "max_speed_mps", "max_accel_mps2",
    "brake_accel_mps2", "reaction_time_s", "max_vertical_speed_mps",
}
_DENSITY_KEYS = {
    "mean", "sigma_major_m", "sigma_minor_m", "sigma_vertical_m", "bearing_deg", "pitch_deg",
    "roll_deg", "weight",
}


class Mission(NamedTuple):
    """A validated mission, ready to fly.

    Attributes:
        mission_id: Operator identifier.
        params: Controller parameters in NED with the safety grid folded in.
        guard_grid: The in-flight plan check's occupancy: obstacles grown by the hard core of the
            margin only (robot radius, clearance and cell size), shape of ``grid``.
        grid: Inflated boolean occupancy, ``(H, W)`` or ``(Z, H, W)``; outside the area, below
            the ground and above the ceiling are occupied.
        origin: NED corner of the grid's lowest cell, ``(x, y)`` or ``(x, y, z)``, float32-exact.
        resolution: Grid cell size in metres, float32-exact.
        duration_s: Mission length in seconds.
        command_mode: One of ``COMMAND_MODES``.
        max_speed: Speed limit in metres per second.
        max_vertical_speed: Vertical speed limit in metres per second (3D modes).
        max_altitude: Mission ceiling above the EKF origin, in metres.
        margin: Safety margin every obstacle and boundary is grown by, in metres.
        tracking_allowance: The part of ``margin`` budgeted for the drone's tracking error, in
            metres; a trajectory mission re-anchors its plan when the drone strays further.
        reachable: Free cells connected to the density modes, shape of ``grid``; the drone
            must start in one of them.
        dry_run_start: Position the preflight model flight starts from: the heaviest mode.
    """

    mission_id: str
    params: ControllerParams
    grid: np.ndarray
    guard_grid: np.ndarray
    origin: tuple[float, ...]
    resolution: float
    duration_s: float
    command_mode: str
    max_speed: float
    max_vertical_speed: float
    max_altitude: float
    margin: float
    tracking_allowance: float
    reachable: np.ndarray
    dry_run_start: tuple[float, ...]


def dimension(command_mode: str) -> int:
    """Position dimension a command mode flies: 3 for the ``_3d`` modes, otherwise 2."""
    return 3 if command_mode.endswith("_3d") else 2


def ned_from_wgs84(points: np.ndarray, reference: tuple[float, float]) -> np.ndarray:
    """
    Project ``[lat, lon]`` degrees onto the local NED plane around ``reference``.

    Mirrors PX4's ``MapProjection::project``, which is how PX4 places its own local frame, so
    a projected point lands where the EKF believes it is.

    Args:
            points: Geodetic points with shape ``(N, 2)``, in degrees.
            reference: EKF origin ``(lat, lon)`` in degrees.
    Returns:
            NED ``(north, east)`` in metres, shape ``(N, 2)``.
    """
    lat, lon = np.radians(np.asarray(points, dtype=np.float64).reshape(-1, 2)).T
    ref_lat, ref_lon = np.radians(np.asarray(reference, dtype=np.float64))
    cos_d_lon = np.cos(lon - ref_lon)
    arg = np.clip(
        np.sin(ref_lat) * np.sin(lat) + np.cos(ref_lat) * np.cos(lat) * cos_d_lon, -1.0, 1.0
    )
    c = np.arccos(arg)
    safe = np.where(c > 0.0, c, 1.0)
    k = np.where(c > 0.0, safe / np.sin(safe), 1.0)
    north = k * (np.cos(ref_lat) * np.sin(lat) - np.sin(ref_lat) * np.cos(lat) * cos_d_lon)
    east = k * np.cos(lat) * np.sin(lon - ref_lon)
    return EARTH_RADIUS_M * np.stack((north, east), axis=-1)


def _keys(mapping: Any, expected: set[str], path: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} must be an object")
    missing, unknown = sorted(expected - mapping.keys()), sorted(mapping.keys() - expected)
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if unknown:
        raise ValueError(f"{path} has unknown keys {unknown}")


def _number(value: Any, path: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{path} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{path} must not be negative")
    return float(value)


def _points(value: Any, path: str, minimum: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{path} must be a list of [lat, lon] pairs") from None
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or array.shape[0] < minimum
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{path} must hold at least {minimum} [lat, lon] pair(s)")
    if np.any(np.abs(array[:, 0]) > 90.0) or np.any(np.abs(array[:, 1]) > 180.0):
        raise ValueError(f"{path} has a latitude or longitude out of range")
    return array


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd test of points ``(N, 2)`` against a closed polygon ``(M, 2)``."""
    x, y = points[:, :1], points[:, 1:]
    xa, ya = polygon[:, 0], polygon[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    straddles = (ya > y) != (yb > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = xa + (y - ya) * (xb - xa) / (yb - ya)
    return np.count_nonzero(straddles & (x < crossing), axis=1) % 2 == 1


def _simple(polygon: np.ndarray) -> bool:
    """Whether no two non-adjacent edges of a closed polygon ``(M, 2)`` cross each other."""

    def turn(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    count = len(polygon)
    edges = [(polygon[index], polygon[(index + 1) % count]) for index in range(count)]
    for first in range(count):
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue  # adjacent through the closing edge
            (p, q), (r, t) = edges[first], edges[second]
            # ponytail: proper crossings only; touching or collinear edges pass.
            if turn(p, q, r) * turn(p, q, t) < 0 and turn(r, t, p) * turn(r, t, q) < 0:
                return False
    return True


def _heights(
    area: np.ndarray,
    circles: list[tuple[np.ndarray, float, float]],
    polygons: list[tuple[np.ndarray, float]],
    origin: tuple[float, float],
    shape: tuple[int, int],
    resolution: float,
) -> np.ndarray:
    """Tallest thing over each cell centre in metres: ``inf`` outside the area, 0 where open."""
    height, width = shape
    x = origin[0] + (np.arange(width) + 0.5) * resolution
    tallest = np.empty(shape)
    # ponytail: one grid row at a time keeps the (cells x edges) block small on the Jetson.
    for row in range(height):
        centres = np.column_stack((x, np.full(width, origin[1] + (row + 0.5) * resolution)))
        top = np.where(_inside(centres, area), 0.0, np.inf)
        for centre, radius, obstacle_height in circles:
            covered = np.sum((centres - centre) ** 2, axis=1) <= radius * radius
            top = np.where(covered, np.maximum(top, obstacle_height), top)
        for polygon, obstacle_height in polygons:
            top = np.where(_inside(centres, polygon), np.maximum(top, obstacle_height), top)
        tallest[row] = top
    return tallest


def _voxels(heights: np.ndarray, max_altitude: float, resolution: float) -> tuple[np.ndarray, float]:
    """
    Stack a height map into NED voxels between the ground and the ceiling.

    Layer 0 lies wholly above the ceiling and the last layer wholly below the ground, so both
    bounds are occupied. A voxel is occupied when it overlaps an obstacle's height, the
    ground, or the space above the ceiling.

    Returns:
            The occupancy ``(Z, H, W)`` and the NED z of layer 0's upper (most negative) face.
    """
    above = math.ceil(max_altitude / resolution) + 1
    layers = above + 1
    top_z = -above * resolution
    occupied = np.empty((layers, *heights.shape), dtype=bool)
    for layer in range(layers):
        top = -(top_z + layer * resolution)  # altitude of the voxel's upper face
        bottom = top - resolution
        if top > max_altitude or bottom < 0.0:
            occupied[layer] = True
        else:
            occupied[layer] = heights > bottom
    return occupied, float(np.float32(top_z))


def _covariance(sigmas: tuple[float, float, float], bearing: float, pitch: float, roll: float) -> np.ndarray:
    """
    NED covariance of a blob with body axes (major, minor, up) turned by yaw, pitch and roll.

    ``R = R_yaw(bearing) R_pitch R_roll`` maps the body axes into north-east-up, exactly as the
    Paddock's ``ellipsoid.ts`` draws it: positive pitch lifts the major axis, positive roll the
    minor axis. Flipping the third axis turns north-east-up into NED.
    """
    cy, sy, cp, sp, cr, sr = (f(a) for a in (bearing, pitch, roll) for f in (math.cos, math.sin))
    rotation = (
        np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
        @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    )
    flip = np.diag([1.0, 1.0, -1.0])
    return flip @ rotation @ np.diag(np.square(sigmas)) @ rotation.T @ flip


def compile_mission(spec: dict, base: ControllerParams, reference: tuple[float, float]) -> Mission:
    """
    Validate an ``ergodic/3`` spec and fold it into controller parameters.

    The drone may be anywhere when the mission loads; :func:`start_failure` decides at start
    whether it is somewhere the mission can fly from.

    Args:
            spec: Parsed spec document.
            base: Tuned planar parameters the mission specialises (``configs/uav_profile.yaml``).
            reference: EKF origin ``(lat, lon)`` in degrees, from ``VehicleLocalPosition``.
    Returns:
            The compiled mission.
    Raises:
            ValueError: With an operator-readable reason when the spec is malformed, the area
                is too large or self-intersecting, or a density mode is blocked or cut off
                from the heaviest mode.
    """
    _keys(spec, _SPEC_KEYS, "spec")
    if spec["schema"] != SCHEMA:
        raise ValueError(f"spec.schema must be {SCHEMA!r}")
    mission_id = spec["mission_id"]
    if not isinstance(mission_id, str) or not mission_id:
        raise ValueError("spec.mission_id must be a non-empty string")
    if spec["command_mode"] not in COMMAND_MODES:
        raise ValueError(f"spec.command_mode must be one of {list(COMMAND_MODES)}")
    d = dimension(spec["command_mode"])
    duration_s = _number(spec["duration_s"], "spec.duration_s", positive=True)
    max_altitude = _number(spec["max_altitude_m"], "spec.max_altitude_m", positive=True)

    vehicle = spec["vehicle"]
    _keys(vehicle, _VEHICLE_KEYS, "spec.vehicle")
    margins = ("clearance_m", "tracking_allowance_m")
    limits = {
        key: _number(
            vehicle[key], f"spec.vehicle.{key}", positive=key not in margins,
            non_negative=key in margins,
        )
        for key in _VEHICLE_KEYS
    }

    def project(value: Any, path: str, minimum: int) -> np.ndarray:
        return ned_from_wgs84(_points(value, path, minimum), reference)

    area = project(spec["area"], "spec.area", 3)
    if not _simple(area):
        raise ValueError("spec.area crosses itself; draw the boundary without intersections")
    if not isinstance(spec["obstacles"], list):
        raise ValueError("spec.obstacles must be a list")
    circles, polygons = [], []
    for index, obstacle in enumerate(spec["obstacles"]):
        path = f"spec.obstacles[{index}]"
        kind = obstacle.get("type") if isinstance(obstacle, dict) else None
        if kind == "circle":
            _keys(obstacle, {"type", "center", "radius_m", "height_m"}, path)
            circles.append((
                project([obstacle["center"]], f"{path}.center", 1)[0],
                _number(obstacle["radius_m"], f"{path}.radius_m", positive=True),
                _number(obstacle["height_m"], f"{path}.height_m", positive=True),
            ))
        elif kind == "polygon":
            _keys(obstacle, {"type", "points", "height_m"}, path)
            polygons.append((
                project(obstacle["points"], f"{path}.points", 3),
                _number(obstacle["height_m"], f"{path}.height_m", positive=True),
            ))
        else:
            raise ValueError(f"{path}.type must be 'circle' or 'polygon'")

    density = spec["density"]
    if not isinstance(density, list) or not density:
        raise ValueError("spec.density must list at least one component")
    means, covariances, weights = [], [], []
    for index, component in enumerate(density):
        path = f"spec.density[{index}]"
        _keys(component, _DENSITY_KEYS, path)
        mean = component["mean"]
        if not isinstance(mean, list) or len(mean) != 3:
            raise ValueError(f"{path}.mean must be [lat, lon, altitude_m]")
        horizontal = project([mean[:2]], f"{path}.mean", 1)[0]
        altitude = _number(mean[2], f"{path}.mean altitude", positive=True)
        if altitude > max_altitude:
            raise ValueError(
                f"{path}.mean altitude {altitude:g} m is above the {max_altitude:g} m ceiling"
            )
        major = _number(component["sigma_major_m"], f"{path}.sigma_major_m", positive=True)
        minor = _number(component["sigma_minor_m"], f"{path}.sigma_minor_m", positive=True)
        vertical = _number(component["sigma_vertical_m"], f"{path}.sigma_vertical_m", positive=True)
        angles = (
            math.radians(_number(component[key], f"{path}.{key}"))
            for key in ("bearing_deg", "pitch_deg", "roll_deg")
        )
        covariance = _covariance((major, minor, vertical), *angles)
        if d == 3:
            # NED z points down: the mode sits at -altitude.
            horizontal = np.array([*horizontal, -altitude])
        else:
            # The planar mission flies the blob's footprint: its horizontal marginal.
            covariance = covariance[:2, :2]
        means.append(horizontal)
        covariances.append(covariance)
        weights.append(_number(component["weight"], f"{path}.weight", positive=True))
    means = np.asarray(means)
    weights = np.asarray(weights) / np.sum(weights)

    # Float32-exact resolution and origin: the controller indexes the grid in float32, and
    # every check here must pick the same cells it does.
    resolution = float(np.float32(base.workspace.grid_resolution if d == 2 else VOXEL_RESOLUTION_M))
    lower, upper = area.min(axis=0), area.max(axis=0)
    shape = (
        int(np.ceil((upper[1] - lower[1]) / resolution)),
        int(np.ceil((upper[0] - lower[0]) / resolution)),
    )
    layers = 1 if d == 2 else math.ceil(max_altitude / resolution) + 2
    if shape[0] * shape[1] * layers > MAX_GRID_CELLS:
        extent = upper - lower
        raise ValueError(
            f"spec.area spans {extent[0]:.0f} m x {extent[1]:.0f} m"
            + ("" if d == 2 else f" x {max_altitude:.0f} m")
            + f", more than the {MAX_GRID_CELLS} cells a {resolution:.2f} m grid may hold"
        )
    margin = inflation_radius(
        robot_radius=limits["radius_m"],
        clearance=limits["clearance_m"],
        tracking_allowance=limits["tracking_allowance_m"],
        max_speed=limits["max_speed_mps"],
        brake_accel=limits["brake_accel_mps2"],
        reaction_time=limits["reaction_time_s"],
        resolution=resolution,
        dimension=d,
    )
    if d == 3 and max_altitude <= 2.0 * margin + resolution:
        raise ValueError(
            f"spec.max_altitude_m must exceed {2.0 * margin + resolution:.2f} m: the ground and "
            f"the ceiling each take a {margin:.2f} m safety margin"
        )
    heights = _heights(area, circles, polygons, (lower[0], lower[1]), shape, resolution)
    if d == 2:
        origin = (float(np.float32(lower[0])), float(np.float32(lower[1])))
        occupied = heights > 0.0
    else:
        occupied, top_z = _voxels(heights, max_altitude, resolution)
        origin = (float(np.float32(lower[0])), float(np.float32(lower[1])), top_z)
    grid = inflate(occupied, margin, resolution)
    # The margin is a hard core (the drone's radius, clearance and the cell size) plus a buffer
    # (tracking allowance and stopping distance) that the controller's soft obstacle cost may
    # briefly dip into. The controller plans against the whole margin; the in-flight plan check
    # faults only on the core, where a plan really is about to hit something. The core is
    # rounded to whole cells, not up: the cell-size term already covers where in a cell a
    # point lies, and rounding up would make the 0.5 m voxels swallow the whole buffer.
    core = inflation_radius(
        robot_radius=limits["radius_m"], clearance=limits["clearance_m"], tracking_allowance=0.0,
        max_speed=0.0, brake_accel=1.0, reaction_time=0.0, resolution=resolution, dimension=d,
    )
    guard_grid = inflate(occupied, resolution * round(core / resolution) - 1e-6, resolution)

    heaviest = tuple(float(value) for value in means[int(np.argmax(weights))])
    reachable, _, diagnosis = all_reachable(grid, origin, resolution, heaviest, means)
    if diagnosis["blocked_modes"]:
        raise ValueError(
            f"density modes {diagnosis['blocked_modes']} sit inside an obstacle, its "
            f"{margin:.2f} m safety margin, or outside the area"
            + ("" if d == 2 else " or the altitude band")
        )
    if not reachable:
        raise ValueError(
            f"density modes {diagnosis['disconnected_modes']} are cut off from the heaviest mode"
        )

    workspace = replace(
        base.workspace,
        x_limits=jnp.asarray([lower[0], upper[0]], dtype=jnp.float32),
        y_limits=jnp.asarray([lower[1], upper[1]], dtype=jnp.float32),
        extra_limits=(
            jnp.zeros((0, 2), dtype=jnp.float32) if d == 2
            else jnp.asarray([[-max_altitude, 0.0]], dtype=jnp.float32)
        ),
        obstacles=jnp.zeros((0, 3), dtype=jnp.float32),
        grid=jnp.asarray(grid, dtype=jnp.float32),
        grid_origin=jnp.asarray(origin, dtype=jnp.float32),
        grid_resolution=resolution,
    )
    mppi = base.mppi
    if d == 3:
        # The tuned planar sampling noise, with z sampled like x and y (as main's lift does).
        variance = np.diag(np.asarray(base.mppi.covariance))
        noise = np.diag(np.r_[np.full(d, variance[0]), variance[-1]]).astype(np.float32)
        mppi = replace(
            mppi,
            covariance=jnp.asarray(noise),
            covariance_inverse=jnp.asarray(np.linalg.inv(noise).astype(np.float32)),
        )
    params = replace(
        base,
        mppi=mppi,
        gmm=gmm_params(means, np.asarray(covariances), weights),
        workspace=workspace,
        model=replace(base.model, max_accel_lin_abs=limits["max_accel_mps2"]),
        # The speed schedule peaks at reference_speed * transit_speedup in the corridors.
        # Scale it so that peak is the vehicle limit, keeping the 1/p* shape of the schedule.
        field=replace(
            base.field, reference_speed=limits["max_speed_mps"] / base.field.transit_speedup
        ),
    )
    return Mission(
        mission_id=mission_id,
        params=params,
        grid=grid,
        guard_grid=guard_grid,
        origin=origin,
        resolution=resolution,
        duration_s=duration_s,
        command_mode=spec["command_mode"],
        max_speed=limits["max_speed_mps"],
        max_vertical_speed=limits["max_vertical_speed_mps"],
        max_altitude=max_altitude,
        margin=margin,
        tracking_allowance=limits["tracking_allowance_m"],
        reachable=diagnosis["component"],
        dry_run_start=heaviest,
    )


def altitude_failure(mission: Mission, altitude: float) -> str | None:
    """Say why ``altitude`` (metres above the EKF origin) breaks the ceiling, or ``None``."""
    if altitude > mission.max_altitude:
        return (
            f"the drone is {altitude:.1f} m up, above the {mission.max_altitude:.1f} m "
            "mission ceiling"
        )
    return None


def start_failure(
    mission: Mission,
    position: tuple[float, ...],
    altitude: float,
    position_age_s: float,
    origin_moved: bool,
) -> str | None:
    """
    Say why the mission must not start or resume from here, or ``None`` when it may.

    Args:
            mission: The loaded mission.
            position: Measured NED position in metres, ``(x, y)`` or ``(x, y, z)`` to match the
                mission's dimension.
            altitude: Measured altitude above the EKF origin in metres.
            position_age_s: Seconds since that position arrived; ``inf`` when it is invalid.
            origin_moved: Whether PX4 reset its EKF origin since the mission was loaded.
    Returns:
            The refusal reason, or ``None``.
    """
    if position_age_s > STALE_POSITION_S:
        return "no fresh, valid local position from PX4"
    if origin_moved:
        return "PX4 moved its EKF origin since the mission was loaded; load it again"
    failure = altitude_failure(mission, altitude)
    if failure is not None:
        return failure
    cell = tuple(int(index) for index in world_to_cell(np.asarray(position), mission.origin, mission.resolution))
    inside = all(0 <= index < size for index, size in zip(cell, mission.grid.shape))
    if not inside or not mission.reachable[cell]:
        return (
            "bring the drone inside the area, clear of the "
            f"{mission.margin:.2f} m safety margin and connected to the density modes"
        )
    return None


def command_vectors(
    mission: Mission, planned_state: np.ndarray, control: np.ndarray, measured_velocity
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn one controller step into the ``MissionCommand`` velocity and acceleration.

    Args:
            mission: The mission being flown.
            planned_state: State one step ahead on the optimal trajectory, shape ``(2d + 2,)``.
            control: First control of the step, shape ``(d + 1,)``.
            measured_velocity: Measured NED velocity, shape ``(d,)``.
    Returns:
            ``(velocity, acceleration)``, each shape ``(3,)`` float32; z is NaN in the planar
            modes. In the acceleration modes the velocity is all NaN. The acceleration (the
            feedforward in the velocity modes) is capped along the measured velocity so the
            speed settles at ``max_speed``: the cap shrinks to zero at the limit and brakes
            above it. In the 3D modes the vertical axis gets the same cap at
            ``max_vertical_speed``, and the planned vz is clipped to it.
    """
    d = dimension(mission.command_mode)

    def padded(vector) -> np.ndarray:
        return np.array([*vector, *([np.nan] * (3 - d))], dtype=np.float32)

    acceleration = np.asarray(control[:d], dtype=np.float64)
    measured = np.asarray(measured_velocity, dtype=np.float64)
    speed = float(np.linalg.norm(measured))
    if speed > 1e-3:
        heading = measured / speed
        along = float(acceleration @ heading)
        cap = SPEED_CAP_GAIN * (mission.max_speed - speed)
        if along > cap:
            acceleration = acceleration + (cap - along) * heading
    # ponytail: the planner's speed schedule is isotropic and knows nothing of the vertical
    # limit (the field scene plans climbs near 0.8 m/s), so the commands enforce it and the
    # replan from the measured state absorbs the slower climb. Scale the vertical part of the
    # reference flow in mppi/core.py if the drone lags its plans near short obstacles.
    if d == 3 and abs(measured[2]) > 1e-3:
        up_or_down = math.copysign(1.0, measured[2])
        cap = SPEED_CAP_GAIN * (mission.max_vertical_speed - abs(measured[2]))
        if acceleration[2] * up_or_down > cap:
            acceleration[2] = cap * up_or_down
    if not mission.command_mode.startswith("acceleration"):
        velocity = np.asarray(planned_state[d:2 * d], dtype=np.float64)
        planned_speed = float(np.linalg.norm(velocity))
        if planned_speed > mission.max_speed:
            velocity = velocity * (mission.max_speed / planned_speed)
        if d == 3:
            velocity[2] = np.clip(velocity[2], -mission.max_vertical_speed, mission.max_vertical_speed)
        return padded(velocity), padded(acceleration)
    return np.full(3, np.nan, dtype=np.float32), padded(acceleration)


def command_position(mission: Mission, planned_state: np.ndarray) -> np.ndarray:
    """
    The ``MissionCommand`` position: the planned position in the trajectory modes, else NaN.

    Args:
            mission: The mission being flown.
            planned_state: State one step ahead on the reference trajectory, shape ``(2d + 2,)``.
    Returns:
            Shape ``(3,)`` float32 NED metres; z is NaN in the planar modes.
    """
    if not mission.command_mode.startswith("trajectory"):
        return np.full(3, np.nan, dtype=np.float32)
    d = dimension(mission.command_mode)
    return np.array([*planned_state[:d], *([np.nan] * (3 - d))], dtype=np.float32)


def replans_from_measurement(mission: Mission, reference_state: np.ndarray, measured_state: np.ndarray) -> bool:
    """
    Whether the next controller step starts from a fresh measurement or from the reference.

    The velocity and acceleration modes replan from every PX4 measurement. The trajectory modes
    plan from their own reference, the state the previous step planned, so the position they
    send stays a smooth trajectory ahead of the drone and PX4's position loop pulls the drone
    onto it: replanning from the measurement would put the setpoint on the drone and leave that
    loop nothing to correct. When the drone strays from the reference by more than the mission's
    tracking allowance, the clearance the plan was built with no longer covers it, so the
    reference re-anchors on the measurement.

    Args:
            mission: The mission being flown.
            reference_state: The controller's current state, shape ``(2d + 2,)``.
            measured_state: A fresh measurement in the same layout.
    Returns:
            ``True`` to step from ``measured_state``.
    """
    if not mission.command_mode.startswith("trajectory"):
        return True
    d = dimension(mission.command_mode)
    error = np.linalg.norm(np.asarray(measured_state[:d], np.float64) - np.asarray(reference_state[:d], np.float64))
    return bool(error > mission.tracking_allowance)


def flight_step(
    params: ControllerParams,
    carry: SingleControllerState,
    observation: jax.Array,
    plan_steps: int,
) -> tuple[SingleControllerState, jax.Array]:
    """
    One controller step, packed so the host needs a single device-to-host copy.

    Each copy costs milliseconds on the Jetson, so everything a flight tick reads comes back
    in one flat array; split it with :func:`unpack_flight`.

    Args:
            params: Controller parameters.
            carry: Current closed-loop carry.
            observation: Measured (or predicted) state with shape ``(2d + 2,)``.
            plan_steps: Planned positions to return; static under JIT.
    Returns:
            The next carry and a float32 array of length ``d * plan_steps + 3d + 3``.
    """
    d = params.gmm.means.shape[-1]
    carry, result = measured_step(params, carry, observation)
    packed = jnp.concatenate(
        (result.optimal_trajectory[:plan_steps, :d].ravel(), carry.state, result.control)
    )
    return carry, packed


def compile_flight(
    params: ControllerParams,
    carry: SingleControllerState,
    observation: np.ndarray,
    plan_steps: int,
):
    """
    JIT :func:`flight_step` and compile every argument kind the flight loop passes.

    The loop feeds a fresh measurement as a host array and, between measurements, the
    prediction already on the device, first with the initial carry and then with carries the
    step returned. Each kind is its own compilation, and compiling one mid-flight stalls the
    loop for seconds, so all of them are compiled here.

    Args:
            params: Controller parameters, already on the flight device.
            carry: Initial carry, already on the flight device.
            observation: A measured state as a host float32 array, shape ``(6,)``.
            plan_steps: Planned positions each step returns.
    Returns:
            The compiled step, called as ``step(params, carry, observation)``.
    """
    step = jax.jit(functools.partial(flight_step, plan_steps=plan_steps))
    observation = np.asarray(observation, dtype=np.float32)
    next_carry, packed = step(params, carry, observation)
    jax.device_get(packed)
    for state in (carry.state, next_carry.state):
        jax.device_get(step(params, next_carry, state)[1])
    jax.device_get(step(params, next_carry, observation)[1])
    return step


def unpack_flight(
    packed: np.ndarray, plan_steps: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split :func:`flight_step`'s array into planned positions, next state and control."""
    split = d * plan_steps
    state_end = split + 2 * d + 2
    return packed[:split].reshape(plan_steps, d), packed[split:state_end], packed[state_end:state_end + d + 1]


def plan_blocked(mission: Mission, positions: np.ndarray) -> bool:
    """
    Whether a planned polyline enters the hard core of a margin (``guard_grid``) or leaves it.

    Planned points are one controller step apart, far closer than half a cell at the speed
    cap, so checking the points alone cannot step over a blocked cell. Sparser plans fall back
    to the segment-exact :func:`path_blocked`.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, mission.grid.ndim)
    if positions.shape[0] > 1 and np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) > 0.5 * mission.resolution:
        return path_blocked(mission.guard_grid, mission.origin, mission.resolution, positions)
    return bool(_hits(mission, positions, mission.guard_grid).any())


def _hits(mission: Mission, positions: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Per position ``(N, d)``: whether it leaves the grid or lands on a blocked cell of ``grid``
    (the mission's full-margin grid by default)."""
    grid = mission.grid if grid is None else grid
    cells = world_to_cell(positions, mission.origin, mission.resolution)
    shape = np.asarray(grid.shape)
    inside = np.all((cells >= 0) & (cells < shape), axis=1)
    return ~inside | grid[tuple(np.clip(cells, 0, shape - 1).T)]


def dry_run_failure(mission: Mission, state: jax.Array, key: jax.Array, seconds: float) -> str | None:
    """
    Fly the compiled mission in the controller's own model and report the first problem.

    Args:
            mission: Compiled mission, its parameters already on the target device.
            state: Start state with shape ``(2d + 2,)``, on the same device.
            key: PRNG key.
            seconds: Simulated flight time.
    Returns:
            Why the mission must not fly, or ``None`` when the model flight is clean.
    """
    # Parameters placed on a device hold delta_t as an array; the scan length must be static.
    delta_t = float(mission.params.model.delta_t)
    steps = max(1, round(seconds / delta_t))
    d = mission.params.gmm.means.shape[-1]
    controls = jnp.zeros((mission.params.mppi.horizon, d + 1), dtype=jnp.float32)
    result = jax.jit(run_single, static_argnames="steps")(
        mission.params, state, controls, key, steps=steps
    )
    path = np.asarray(result.path)
    if not np.all(np.isfinite(path)):
        return "the model flight produced non-finite states"
    hit = _hits(mission, path[:, :d])
    if hit.any():
        return (
            "the model flight enters a safety margin or leaves the area after "
            f"{(int(np.argmax(hit)) + 1) * delta_t:.1f} s"
        )
    speed = float(np.max(np.linalg.norm(path[:, d:2 * d], axis=1)))
    if speed > SPEED_TOLERANCE * mission.max_speed:
        return f"the model flight reaches {speed:.2f} m/s against a {mission.max_speed:.2f} m/s limit"
    return None

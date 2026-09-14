"""
Compile an operator mission spec into controller parameters in PX4's local NED frame.

The paddock draws in WGS84. The drone projects every point around its own EKF origin with
PX4's azimuthal-equidistant map projection, so the controller runs directly in NED x (north)
and y (east). A bearing measured clockwise from north is then the x-to-y angle: there is no
handedness flip, and headings taken from the plan are PX4 yaw as they are.
"""

import math
from dataclasses import replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import gmm_params
from ergodic_control_mppi.deploy.grid import all_reachable, inflate, inflation_radius, world_to_cell
from ergodic_control_mppi.mppi.single import run_single
from ergodic_control_mppi.parameters import ControllerParams

SCHEMA = "ergodic/1"
COMMAND_MODES = ("velocity", "acceleration")
EARTH_RADIUS_M = 6371000.0  # PX4 CONSTANTS_RADIUS_OF_EARTH
# ponytail: fixed grid resolution caps the area near 200 m x 200 m at 0.15 m cells; coarsen
# the resolution for larger fields.
MAX_GRID_CELLS = 2_000_000
# ponytail: allowance for the model overshooting the scheduled speed peak; the node clamps
# every command to max_speed regardless, so this only catches a mis-scaled schedule.
SPEED_TOLERANCE = 1.25
# Oldest local position a mission may start or step from; mullet_core holds after 0.25 s.
STALE_POSITION_S = 0.1

_SPEC_KEYS = {
    "schema", "mission_id", "command_mode", "duration_s", "vehicle", "area", "obstacles", "density",
}
_VEHICLE_KEYS = {
    "radius_m", "clearance_m", "tracking_allowance_m", "max_speed_mps", "max_accel_mps2",
    "brake_accel_mps2", "reaction_time_s",
}
_DENSITY_KEYS = {"mean", "sigma_major_m", "sigma_minor_m", "bearing_deg", "weight"}


class Mission(NamedTuple):
    """A validated mission, ready to fly.

    Attributes:
        mission_id: Operator identifier.
        params: Controller parameters in NED with the safety grid folded in.
        grid: Inflated boolean occupancy, shape ``(H, W)``; outside the area is occupied.
        origin: NED ``(x, y)`` of the grid's lower corner, float32-exact.
        resolution: Grid cell size in metres, float32-exact.
        duration_s: Mission length in seconds.
        command_mode: ``"velocity"`` or ``"acceleration"``.
        max_speed: Horizontal speed limit in metres per second.
    """

    mission_id: str
    params: ControllerParams
    grid: np.ndarray
    origin: tuple[float, float]
    resolution: float
    duration_s: float
    command_mode: str
    max_speed: float


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


def _free(grid: np.ndarray, origin: tuple[float, float], resolution: float, xy) -> bool:
    """Whether ``xy`` falls on a free cell of ``grid``."""
    row, column = (int(index) for index in world_to_cell(np.asarray(xy), origin, resolution))
    return 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1] and not grid[row, column]


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd test of points ``(N, 2)`` against a closed polygon ``(M, 2)``."""
    x, y = points[:, :1], points[:, 1:]
    xa, ya = polygon[:, 0], polygon[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    straddles = (ya > y) != (yb > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = xa + (y - ya) * (xb - xa) / (yb - ya)
    return np.count_nonzero(straddles & (x < crossing), axis=1) % 2 == 1


def _occupancy(
    area: np.ndarray,
    circles: list[tuple[np.ndarray, float]],
    polygons: list[np.ndarray],
    origin: tuple[float, float],
    shape: tuple[int, int],
    resolution: float,
) -> np.ndarray:
    """Rasterize the outside of the area and every obstacle at the cell centres."""
    height, width = shape
    x = origin[0] + (np.arange(width) + 0.5) * resolution
    occupied = np.empty(shape, dtype=bool)
    # ponytail: one grid row at a time keeps the (cells x edges) block small on the Jetson.
    for row in range(height):
        centres = np.column_stack((x, np.full(width, origin[1] + (row + 0.5) * resolution)))
        blocked = ~_inside(centres, area)
        for centre, radius in circles:
            blocked |= np.sum((centres - centre) ** 2, axis=1) <= radius * radius
        for polygon in polygons:
            blocked |= _inside(centres, polygon)
        occupied[row] = blocked
    return occupied


def compile_mission(
    spec: dict,
    base: ControllerParams,
    reference: tuple[float, float],
    start_xy: tuple[float, float],
) -> Mission:
    """
    Validate an ``ergodic/1`` spec and fold it into controller parameters.

    Args:
            spec: Parsed spec document.
            base: Tuned parameters the mission specialises (``configs/uav_profile.yaml``).
            reference: EKF origin ``(lat, lon)`` in degrees, from ``VehicleLocalPosition``.
            start_xy: Current NED position of the drone in metres.
    Returns:
            The compiled mission.
    Raises:
            ValueError: With an operator-readable reason when the spec is malformed, the area
                is too large, the drone is not in free space, or a density mode is blocked or
                cut off from the drone.
    """
    _keys(spec, _SPEC_KEYS, "spec")
    if spec["schema"] != SCHEMA:
        raise ValueError(f"spec.schema must be {SCHEMA!r}")
    mission_id = spec["mission_id"]
    if not isinstance(mission_id, str) or not mission_id:
        raise ValueError("spec.mission_id must be a non-empty string")
    if spec["command_mode"] not in COMMAND_MODES:
        raise ValueError(f"spec.command_mode must be one of {list(COMMAND_MODES)}")
    duration_s = _number(spec["duration_s"], "spec.duration_s", positive=True)

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
    if not isinstance(spec["obstacles"], list):
        raise ValueError("spec.obstacles must be a list")
    circles, polygons = [], []
    for index, obstacle in enumerate(spec["obstacles"]):
        path = f"spec.obstacles[{index}]"
        kind = obstacle.get("type") if isinstance(obstacle, dict) else None
        if kind == "circle":
            _keys(obstacle, {"type", "center", "radius_m"}, path)
            centre = project([obstacle["center"]], f"{path}.center", 1)[0]
            circles.append((centre, _number(obstacle["radius_m"], f"{path}.radius_m", positive=True)))
        elif kind == "polygon":
            _keys(obstacle, {"type", "points"}, path)
            polygons.append(project(obstacle["points"], f"{path}.points", 3))
        else:
            raise ValueError(f"{path}.type must be 'circle' or 'polygon'")

    density = spec["density"]
    if not isinstance(density, list) or not density:
        raise ValueError("spec.density must list at least one component")
    means, covariances, weights = [], [], []
    for index, component in enumerate(density):
        path = f"spec.density[{index}]"
        _keys(component, _DENSITY_KEYS, path)
        means.append(project([component["mean"]], f"{path}.mean", 1)[0])
        major = _number(component["sigma_major_m"], f"{path}.sigma_major_m", positive=True)
        minor = _number(component["sigma_minor_m"], f"{path}.sigma_minor_m", positive=True)
        bearing = math.radians(_number(component["bearing_deg"], f"{path}.bearing_deg"))
        axis = np.array([math.cos(bearing), math.sin(bearing)])
        normal = np.array([-axis[1], axis[0]])
        covariances.append(major**2 * np.outer(axis, axis) + minor**2 * np.outer(normal, normal))
        weights.append(_number(component["weight"], f"{path}.weight", positive=True))
    means = np.asarray(means)
    weights = np.asarray(weights) / np.sum(weights)

    # Float32-exact resolution and origin: the controller indexes the grid in float32, and
    # every check here must pick the same cells it does.
    resolution = float(np.float32(base.workspace.grid_resolution))
    lower, upper = area.min(axis=0), area.max(axis=0)
    shape = (
        int(np.ceil((upper[1] - lower[1]) / resolution)),
        int(np.ceil((upper[0] - lower[0]) / resolution)),
    )
    if shape[0] * shape[1] > MAX_GRID_CELLS:
        extent = upper - lower
        raise ValueError(
            f"spec.area spans {extent[0]:.0f} m x {extent[1]:.0f} m, more than the "
            f"{MAX_GRID_CELLS} cells a {resolution:.2f} m grid may hold"
        )
    origin = (float(np.float32(lower[0])), float(np.float32(lower[1])))
    margin = inflation_radius(
        robot_radius=limits["radius_m"],
        clearance=limits["clearance_m"],
        tracking_allowance=limits["tracking_allowance_m"],
        max_speed=limits["max_speed_mps"],
        brake_accel=limits["brake_accel_mps2"],
        reaction_time=limits["reaction_time_s"],
        resolution=resolution,
    )
    grid = inflate(_occupancy(area, circles, polygons, origin, shape, resolution), margin, resolution)

    if not _free(grid, origin, resolution, start_xy):
        raise ValueError(
            "the drone is not in the free part of the area: bring it inside, clear of every "
            f"obstacle and of the {margin:.2f} m safety margin, before loading"
        )
    reachable, _, diagnosis = all_reachable(grid, origin, resolution, tuple(start_xy), means)
    if not reachable:
        if diagnosis["blocked_modes"]:
            raise ValueError(
                f"density modes {diagnosis['blocked_modes']} sit inside an obstacle, its "
                f"{margin:.2f} m safety margin, or outside the area"
            )
        raise ValueError(
            f"density modes {diagnosis['disconnected_modes']} are cut off from the drone"
        )

    workspace = replace(
        base.workspace,
        x_limits=jnp.asarray([lower[0], upper[0]], dtype=jnp.float32),
        y_limits=jnp.asarray([lower[1], upper[1]], dtype=jnp.float32),
        obstacles=jnp.zeros((0, 3), dtype=jnp.float32),
        grid=jnp.asarray(grid, dtype=jnp.float32),
        grid_origin=jnp.asarray(origin, dtype=jnp.float32),
        grid_resolution=resolution,
    )
    params = replace(
        base,
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
        origin=origin,
        resolution=resolution,
        duration_s=duration_s,
        command_mode=spec["command_mode"],
        max_speed=limits["max_speed_mps"],
    )


def start_failure(
    mission: Mission, position_xy: tuple[float, float], position_age_s: float, origin_moved: bool
) -> str | None:
    """
    Say why the mission must not start or resume from here, or ``None`` when it may.

    Args:
            mission: The loaded mission.
            position_xy: Measured NED position in metres.
            position_age_s: Seconds since that position arrived; ``inf`` when it is invalid.
            origin_moved: Whether PX4 reset its EKF origin since the mission was loaded.
    Returns:
            The refusal reason, or ``None``.
    """
    if position_age_s > STALE_POSITION_S:
        return "no fresh, valid local position from PX4"
    if origin_moved:
        return "PX4 moved its EKF origin since the mission was loaded; load it again"
    if not _free(mission.grid, mission.origin, mission.resolution, position_xy):
        return "the drone is inside a safety margin or outside the area"
    return None


def command_vectors(
    mission: Mission, planned_state: np.ndarray, control: np.ndarray, measured_velocity
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn one controller step into the ``MissionCommand`` velocity and acceleration.

    Args:
            mission: The mission being flown.
            planned_state: State one step ahead on the optimal trajectory, shape ``(6,)``.
            control: First control of the step, shape ``(3,)``.
            measured_velocity: Measured horizontal NED velocity, shape ``(2,)``.
    Returns:
            ``(velocity, acceleration)``, each shape ``(3,)`` float32 with a NaN z. In
            acceleration mode the velocity is all NaN.
    """
    acceleration = np.asarray(control[:2], dtype=np.float64)
    if mission.command_mode == "velocity":
        velocity = np.asarray(planned_state[2:4], dtype=np.float64)
        speed = float(np.hypot(*velocity))
        if speed > mission.max_speed:
            velocity = velocity * (mission.max_speed / speed)
        return (
            np.array([*velocity, np.nan], dtype=np.float32),
            np.array([*acceleration, np.nan], dtype=np.float32),
        )
    # No velocity loop to clamp in acceleration mode: at the limit, drop the part of the
    # acceleration that would speed the vehicle up further.
    measured = np.asarray(measured_velocity, dtype=np.float64)
    speed = float(np.hypot(*measured))
    if speed >= mission.max_speed:
        heading = measured / speed
        along = float(acceleration @ heading)
        if along > 0.0:
            acceleration = acceleration - along * heading
    return np.full(3, np.nan, dtype=np.float32), np.array([*acceleration, np.nan], dtype=np.float32)


def dry_run_failure(mission: Mission, state: jax.Array, key: jax.Array, seconds: float) -> str | None:
    """
    Fly the compiled mission in the controller's own model and report the first problem.

    Args:
            mission: Compiled mission, its parameters already on the target device.
            state: Measured start state with shape ``(6,)``, on the same device.
            key: PRNG key.
            seconds: Simulated flight time.
    Returns:
            Why the mission must not fly, or ``None`` when the model flight is clean.
    """
    # Parameters placed on a device hold delta_t as an array; the scan length must be static.
    delta_t = float(mission.params.model.delta_t)
    steps = max(1, round(seconds / delta_t))
    controls = jnp.zeros((mission.params.mppi.horizon, 3), dtype=jnp.float32)
    result = jax.jit(run_single, static_argnames="steps")(
        mission.params, state, controls, key, steps=steps
    )
    path = np.asarray(result.path)
    if not np.all(np.isfinite(path)):
        return "the model flight produced non-finite states"
    height, width = mission.grid.shape
    rows, columns = world_to_cell(path[:, :2], mission.origin, mission.resolution).T
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    hit = ~inside | mission.grid[np.clip(rows, 0, height - 1), np.clip(columns, 0, width - 1)]
    if hit.any():
        return (
            "the model flight enters a safety margin or leaves the area after "
            f"{(int(np.argmax(hit)) + 1) * delta_t:.1f} s"
        )
    speed = float(np.max(np.hypot(path[:, 2], path[:, 3])))
    if speed > SPEED_TOLERANCE * mission.max_speed:
        return f"the model flight reaches {speed:.2f} m/s against a {mission.max_speed:.2f} m/s limit"
    return None

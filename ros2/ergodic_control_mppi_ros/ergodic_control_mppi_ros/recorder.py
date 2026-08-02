"""Buffer one deployment trial, score it, and end the launch.

The recorder is the only node that decides when a run is over: it counts control steps and
shuts itself down, which the launch turns into a full teardown through an ``OnProcessExit``
handler. Scoring goes through ``deploy.summary`` so a UAV trial and its paired offline twin
are measured by identical code.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import OccupancyGrid, Odometry
from quadrotor_msgs.msg import PositionCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.grid import inflation_radius, metric_reachable_mask
from ergodic_control_mppi.deploy.summary import append_summary, compute_row
from ergodic_control_mppi.experiments.common import build_target_grid

LATEST = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
TRANSIENT_LOCAL = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
METRIC_GRID = (80, 80)


def git_sha() -> str:
    """Return the current commit, or ``unknown`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_hash(path: str | Path) -> str:
    """Return a short content hash of a file, or ``unknown`` if unreadable."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


class Recorder(Node):
    """Record a fixed number of control steps, then flush and stop the launch."""

    def __init__(self) -> None:
        super().__init__("recorder")
        self.config_path = self.declare_parameter(
            "config", "/workspace/configs/mppi_params.yaml"
        ).value
        self.run_id = self.declare_parameter("run_id", "run").value
        self.output_root = Path(self.declare_parameter("output_root", "results/uav").value)
        self.steps = int(self.declare_parameter("steps", 200).value)
        self.overwrite = bool(self.declare_parameter("overwrite", False).value)
        self.bag = bool(self.declare_parameter("bag", False).value)
        self.profile = self.declare_parameter("profile", "baseline").value
        self.seed = int(self.declare_parameter("seed", -1).value)
        self.map_seed = int(self.declare_parameter("map_seed", 511).value)
        self.map_fill = float(self.declare_parameter("map_fill", 0.1).value)
        self.map_source = self.declare_parameter("map_source", "perlin3d").value
        self.obs_num = int(self.declare_parameter("obs_num", 45).value)
        self.pillar_min_radius = float(
            self.declare_parameter("pillar_min_radius", 0.3).value
        )
        self.pillar_max_radius = float(
            self.declare_parameter("pillar_max_radius", 0.6).value
        )
        self.pillar_min_height = float(
            self.declare_parameter("pillar_min_height", 2.0).value
        )
        self.pillar_max_height = float(
            self.declare_parameter("pillar_max_height", 3.0).value
        )
        self.pillar_min_distance = float(
            self.declare_parameter("pillar_min_distance", 1.2).value
        )
        self.robot_radius = float(self.declare_parameter("robot_radius", 0.30).value)
        self.deadline_ms = float(self.declare_parameter("deadline_ms", 16.0).value)
        self.preflight_steps = int(self.declare_parameter("preflight_steps", 0).value)
        self.clearance = float(self.declare_parameter("clearance", 0.15).value)
        self.tracking_allowance = float(
            self.declare_parameter("tracking_allowance", 0.05).value
        )
        self.max_speed = float(self.declare_parameter("max_speed", 2.0).value)
        self.brake_accel = float(self.declare_parameter("brake_accel", 6.0).value)
        self.reaction_time = float(self.declare_parameter("reaction_time", 0.10).value)
        self.start_xy = (
            float(self.declare_parameter("start_x", -16.0).value),
            float(self.declare_parameter("start_y", 0.0).value),
        )

        self.directory = self.output_root / self.run_id
        bag_setup = (
            self.bag
            and self.directory.exists()
            and {path.name for path in self.directory.iterdir()} <= {"bag"}
        )
        if self.directory.exists() and not self.overwrite and not bag_setup:
            raise FileExistsError(
                f"run already exists: {self.directory}; pass overwrite:=true to replace it"
            )

        self.config = load_config(self.config_path)
        self.delta_t = float(self.config.controller.model.delta_t)
        if self.seed < 0:
            self.seed = int(self.config.run.seed)

        self.commands: list[list[float]] = []
        self.safe_commands: list[list[float]] = []
        self.odometry: list[list[float]] = []
        self.step_ms: list[float] = []
        self.ess_fractions: list[float] = []
        self.temperatures: list[float] = []
        self.temperature_at_cap: list[bool] = []
        self.guard_states: list[str] = []
        self.guard_period = 0.01
        self.compile_seconds = float("nan")
        self.device = "unknown"
        self.grid = None  # inflated: what was planned against
        self.occupancy = None  # raw physical map: what a collision is measured against
        self.grid_origin = (0.0, 0.0)
        self.grid_resolution = 0.15
        self.started = time.perf_counter()
        self.finished = False

        self.create_subscription(PositionCommand, "/ergodic/cmd_raw", self.on_command, LATEST)
        self.create_subscription(PositionCommand, "/position_cmd", self.on_safe_command, 10)
        self.create_subscription(Odometry, "/sim/odom", self.on_odometry, LATEST)
        self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diagnostics, 10)
        self.create_subscription(
            OccupancyGrid, "/ergodic/safety_grid", self.on_grid, TRANSIENT_LOCAL
        )
        self.create_subscription(
            OccupancyGrid, "/ergodic/map_grid", self.on_map_grid, TRANSIENT_LOCAL
        )
        self.get_logger().info(f"recording {self.steps} steps into {self.directory}")

    def on_grid(self, message: OccupancyGrid) -> None:
        self.grid = _unpack(message)
        self.grid_origin = (message.info.origin.position.x, message.info.origin.position.y)
        self.grid_resolution = float(message.info.resolution)

    def on_map_grid(self, message: OccupancyGrid) -> None:
        self.occupancy = _unpack(message)

    def on_command(self, message: PositionCommand) -> None:
        """Count control steps and stop the run once the budget is spent."""
        if self.finished:
            return
        self.commands.append(_command_row(message))
        if len(self.commands) >= self.steps:
            self.finish()

    def on_safe_command(self, message: PositionCommand) -> None:
        self.safe_commands.append(_command_row(message))

    def on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        linear = message.twist.twist.linear
        self.odometry.append(
            [
                _seconds(message.header.stamp),
                position.x,
                position.y,
                position.z,
                linear.x,
                linear.y,
                linear.z,
                message.twist.twist.angular.z,
            ]
        )

    def on_diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            if status.name == "ergodic_controller":
                if "step_ms" in values:
                    self.step_ms.append(float(values["step_ms"]))
                if "compile_seconds" in values:
                    self.compile_seconds = float(values["compile_seconds"])
                if "ess_fraction" in values:
                    self.ess_fractions.append(float(values["ess_fraction"]))
                if "temperature" in values:
                    self.temperatures.append(float(values["temperature"]))
                if "temperature_at_cap" in values:
                    self.temperature_at_cap.append(
                        values["temperature_at_cap"].lower() == "true"
                    )
                self.device = status.hardware_id or self.device
            elif status.name == "safety_shield" and "guard_state" in values:
                # Only from the first control step onward. The guard necessarily holds
                # hover while the controller compiles and warms up, and counting those
                # seconds would swamp the intervention rate with startup rather than
                # flight -- the acceptance threshold is about flight.
                if self.commands:
                    self.guard_states.append(values["guard_state"])
                if "rate" in values and float(values["rate"]) > 0:
                    self.guard_period = 1.0 / float(values["rate"])

    def finish(self) -> None:
        """Write the arrays, manifest, and summary row, then end the launch."""
        self.finished = True
        wall_seconds = time.perf_counter() - self.started
        self.directory.mkdir(parents=True, exist_ok=True)

        commands = np.asarray(self.commands, dtype=np.float64).reshape(-1, 8)
        safe = np.asarray(self.safe_commands, dtype=np.float64).reshape(-1, 8)
        odometry = np.asarray(self.odometry, dtype=np.float64).reshape(-1, 8)
        positions = odometry[:, 1:3] if odometry.size else np.zeros((0, 2))
        speeds = np.linalg.norm(odometry[:, 4:6], axis=1) if odometry.size else np.zeros(0)
        odometry_seconds = (
            float(odometry[-1, 0] - odometry[0, 0]) if odometry.shape[0] > 1 else 0.0
        )
        # Span the control steps were issued over, excluding compilation before the first.
        control_seconds = (
            float(commands[-1, 0] - commands[0, 0]) if commands.shape[0] > 1 else 0.0
        )
        params = self.config.controller
        x_limits = tuple(float(value) for value in np.asarray(params.workspace.x_limits))
        y_limits = tuple(float(value) for value in np.asarray(params.workspace.y_limits))
        inflated = self.grid if self.grid is not None else np.zeros((1, 1), dtype=bool)
        # Physical obstacles, falling back to the inflated grid only if the raw map never
        # arrived -- a conservative fallback, since it can only over-report contact.
        occupancy = self.occupancy if self.occupancy is not None else inflated
        target_grid = build_target_grid(params, METRIC_GRID)
        # Reachability comes from the flown grid, not from the config's circles: a UAV
        # profile has no circles, so the circle-based mask would exclude nothing.
        reachable = metric_reachable_mask(
            np.asarray(inflated),
            self.grid_origin,
            self.grid_resolution,
            self.start_xy,
            x_limits,
            y_limits,
            METRIC_GRID,
        )

        row = compute_row(
            identity={
                "run_id": self.run_id,
                "profile": self.profile,
                "mode": "uav",
                "seed": self.seed,
                "map_seed": self.map_seed,
                "map_fill": self.map_fill,
                "steps": len(self.commands),
                "compile_s": self.compile_seconds,
                "run_hash": file_hash(self.directory / "manifest.json"),
                "config_hash": file_hash(self.config_path),
                "git_sha": git_sha(),
                "seed_controller": self.seed,
                "jax_version": jax.__version__,
                "ros_distro": _ros_distro(),
                "device": self.device,
                "steps_to_threshold": "",
            },
            positions=positions,
            target_grid=target_grid,
            x_limits=x_limits,
            y_limits=y_limits,
            reachable_mask=reachable,
            gmm_means=np.asarray(params.gmm.means),
            gmm_inverses=np.asarray(params.gmm.covariance_inverse),
            delta_t=self.delta_t,
            occupancy=np.asarray(occupancy),
            grid_origin=self.grid_origin,
            grid_resolution=self.grid_resolution,
            robot_radius=self.robot_radius,
            guard_states=np.asarray(self.guard_states),
            guard_period=self.guard_period,
            speeds=speeds,
            actual_times=odometry[:, 0] if odometry.size else np.zeros(0),
            commanded_times=safe[:, 0] if safe.size else np.zeros(0),
            commanded=safe[:, 1:3] if safe.size else np.zeros((0, 2)),
            step_ms=np.asarray(self.step_ms),
            deadline_ms=self.deadline_ms,
            wall_seconds=wall_seconds,
            odometry_seconds=odometry_seconds,
            control_seconds=control_seconds,
        )

        np.savez_compressed(
            self.directory / "arrays.npz",
            odometry=odometry,
            cmd_raw=commands,
            cmd_safe=safe,
            grid=np.asarray(inflated),
            occupancy=np.asarray(occupancy),
            grid_origin=np.asarray(self.grid_origin),
            grid_resolution=self.grid_resolution,
            target_grid=target_grid,
            reachable_mask=reachable,
            initial_state=_initial_state(odometry),
            step_ms=np.asarray(self.step_ms),
            ess_fraction=np.asarray(self.ess_fractions),
            temperature=np.asarray(self.temperatures),
            temperature_at_cap=np.asarray(self.temperature_at_cap),
            guard_state=np.asarray(self.guard_states),
        )
        radius = inflation_radius(
            self.robot_radius,
            self.clearance,
            self.tracking_allowance,
            self.max_speed,
            self.brake_accel,
            self.reaction_time,
            self.grid_resolution,
        )
        manifest = {
            "run_id": self.run_id,
            "profile": self.profile,
            "config": str(self.config_path),
            # Also relative to the workspace root: the absolute path above is the one
            # inside the container, and the pairing tool usually runs on the host.
            "config_relative": _relative_config(self.config_path),
            "steps": len(self.commands),
            "seed": self.seed,
            "map_seed": self.map_seed,
            "map_fill": self.map_fill,
            "map_source": self.map_source,
            "map_parameters": (
                {"fill": self.map_fill}
                if self.map_source == "perlin3d"
                else {
                    "pillar_count": self.obs_num,
                    "pillar_radius_m": [self.pillar_min_radius, self.pillar_max_radius],
                    "pillar_height_m": [self.pillar_min_height, self.pillar_max_height],
                    "pillar_min_distance_m": self.pillar_min_distance,
                    "ring_count": 0,
                }
            ),
            "delta_t": self.delta_t,
            "deadline_ms": self.deadline_ms,
            "preflight_steps": self.preflight_steps,
            "rng_scheme": "split_controller_v1",
            "robot_radius": self.robot_radius,
            "safety_budget": {
                "clearance_m": self.clearance,
                "tracking_allowance_m": self.tracking_allowance,
                "max_speed_mps": self.max_speed,
                "brake_accel_mps2": self.brake_accel,
                "reaction_time_s": self.reaction_time,
                "inflation_radius_m": radius,
                "inflation_cells": int(np.ceil(radius / self.grid_resolution)),
            },
            "grid_origin": list(self.grid_origin),
            "grid_resolution": self.grid_resolution,
            "start_xy": list(self.start_xy),
            "device": self.device,
            "jax_version": jax.__version__,
            "git_sha": row["git_sha"],
            "config_hash": row["config_hash"],
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        row["run_hash"] = file_hash(self.directory / "manifest.json")
        append_summary(self.output_root / "summary.csv", row)
        self.get_logger().info(
            f"recorded {len(self.commands)} steps: collisions={row['collisions']}, "
            f"guard_fraction={row['guard_fraction']:.4f}, "
            f"occupancy_mse={row['occupancy_mse']:.3e}, p99={row['step_p99_ms']:.2f} ms"
        )
        rclpy.shutdown()


def _command_row(message: PositionCommand) -> list[float]:
    return [
        _seconds(message.header.stamp),
        message.position.x,
        message.position.y,
        message.position.z,
        message.velocity.x,
        message.velocity.y,
        message.acceleration.x,
        message.acceleration.y,
    ]


def _initial_state(odometry: np.ndarray) -> np.ndarray:
    """Rebuild the six-state observation of the first odometry sample."""
    if not odometry.size:
        return np.zeros(6)
    first = odometry[0]
    return np.array([first[1], first[2], first[4], first[5], 0.0, first[7]])


def _relative_config(path: str | Path) -> str:
    """Return the config path relative to the workspace root, or its name if unrelated."""
    candidate = Path(path)
    for root in (Path("/workspace"), Path.cwd()):
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            continue
    return candidate.name


def _unpack(message: OccupancyGrid) -> np.ndarray:
    """Return an occupancy grid message as a boolean array."""
    data = np.asarray(message.data, dtype=np.int8).reshape(
        message.info.height, message.info.width
    )
    return data > 0


def _seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _ros_distro() -> str:
    import os

    return os.environ.get("ROS_DISTRO", "unknown")


def main() -> None:
    """Run the experiment recorder node."""
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()

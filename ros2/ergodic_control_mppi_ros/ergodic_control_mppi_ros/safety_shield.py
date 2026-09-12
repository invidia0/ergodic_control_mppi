"""Gate every setpoint reaching the vehicle.

This node is the only publisher on ``/position_cmd``. It runs independently of the
controller so that a controller that crashes, stalls, or emits garbage cannot reach the
vehicle: anything stale, unmatched, malformed, too fast, or geometrically unsafe is
replaced by a braking command and then a latched hover.
"""

import math
from typing import NamedTuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ergodic_control_mppi.deploy.grid import path_blocked

LATEST = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
TRANSIENT_LOCAL = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
PASS, BRAKE, HOVER = "pass", "brake", "hover"
STOPPED_SPEED = 0.1


class Candidate(NamedTuple):
    """The numbers the guard decides on, extracted from the incoming messages."""

    command_stamp: float
    path_stamp: float
    odometry_stamp: float
    position: np.ndarray
    velocity: np.ndarray
    path: np.ndarray


def reject_reason(
    candidate: Candidate | None,
    now: float,
    timeout: float,
    max_speed: float,
    grid: np.ndarray | None,
    origin: tuple[float, float],
    resolution: float,
) -> str | None:
    """Return why a candidate command must be refused, or ``None`` to accept it.

    Checks run cheapest first so a stale or malformed command never reaches the geometric
    test. Every failure mode is named, because the name is what the recorder logs.
    """
    if candidate is None:
        return "no command"
    if grid is None:
        return "no safety grid"
    if now - candidate.command_stamp > timeout:
        return "stale command"
    if now - candidate.odometry_stamp > timeout:
        return "stale odometry"
    if candidate.command_stamp != candidate.path_stamp:
        return "command and path stamps differ"
    if not (
        np.all(np.isfinite(candidate.position))
        and np.all(np.isfinite(candidate.velocity))
        and candidate.path.size
        and np.all(np.isfinite(candidate.path))
    ):
        return "non-finite command"
    if float(np.linalg.norm(candidate.velocity)) > max_speed:
        return "commanded speed over limit"
    if path_blocked(grid, origin, resolution, candidate.path):
        return "safety path enters a blocked cell"
    return None


class SafetyShield(Node):
    """Forward safe setpoints, brake on anything else, and latch hover once stopped."""

    def __init__(self) -> None:
        super().__init__("safety_shield")
        self.rate = self.declare_parameter("rate", 100.0).value
        self.timeout = self.declare_parameter("command_timeout", 0.10).value
        self.max_speed = self.declare_parameter("max_speed", 2.0).value
        self.brake_accel = self.declare_parameter("brake_accel", 6.0).value
        self.altitude = self.declare_parameter("altitude", 0.75).value

        # Command and path arrive as two messages. The guard runs faster than the
        # controller, so it would otherwise routinely sample the gap between them and
        # reject a perfectly good setpoint. Pair them by stamp and act only on a complete
        # pair; the freshness check below still catches a pair that stops being renewed.
        self.pending: dict[float, dict] = {}
        self.command: PositionCommand | None = None
        self.path: Path | None = None
        self.odometry: Odometry | None = None
        self.grid: np.ndarray | None = None
        self.origin = (0.0, 0.0)
        self.resolution = 0.15
        self.state = HOVER
        self.hover_position: np.ndarray | None = None
        self.last_safe_position: np.ndarray | None = None
        self.last_reason = "startup"

        self.create_subscription(PositionCommand, "/ergodic/cmd_raw", self.on_command, LATEST)
        self.create_subscription(Path, "/ergodic/safety_path", self.on_path, LATEST)
        self.create_subscription(Odometry, "/sim/odom", self.on_odometry, LATEST)
        self.create_subscription(
            OccupancyGrid, "/ergodic/safety_grid", self.on_grid, TRANSIENT_LOCAL
        )
        self.publisher = self.create_publisher(PositionCommand, "/position_cmd", 10)
        self.diagnostics = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_timer(1.0 / self.rate, self.on_tick)

    def on_command(self, message: PositionCommand) -> None:
        self.offer(_seconds(message.header.stamp), "command", message)

    def on_path(self, message: Path) -> None:
        self.offer(_seconds(message.header.stamp), "path", message)

    def offer(self, stamp: float, kind: str, message) -> None:
        """Hold a half-pair until its partner arrives, then publish it as the current pair."""
        self.pending.setdefault(stamp, {})[kind] = message
        entry = self.pending[stamp]
        if "command" in entry and "path" in entry:
            self.command, self.path = entry["command"], entry["path"]
            # Anything older than a completed pair can never complete usefully.
            self.pending = {key: value for key, value in self.pending.items() if key > stamp}
        elif len(self.pending) > 8:
            self.pending = dict(sorted(self.pending.items())[-4:])

    def on_odometry(self, message: Odometry) -> None:
        self.odometry = message

    def on_grid(self, message: OccupancyGrid) -> None:
        data = np.asarray(message.data, dtype=np.int8).reshape(
            message.info.height, message.info.width
        )
        self.grid = data > 0
        self.origin = (message.info.origin.position.x, message.info.origin.position.y)
        self.resolution = float(message.info.resolution)

    def candidate(self) -> Candidate | None:
        """Bundle the current command, path, and odometry, or ``None`` if any is missing."""
        if self.command is None or self.path is None or self.odometry is None:
            return None
        return Candidate(
            command_stamp=_seconds(self.command.header.stamp),
            path_stamp=_seconds(self.path.header.stamp),
            odometry_stamp=_seconds(self.odometry.header.stamp),
            position=np.array(
                [
                    self.command.position.x,
                    self.command.position.y,
                    self.command.position.z,
                ]
            ),
            velocity=np.array(
                [
                    self.command.velocity.x,
                    self.command.velocity.y,
                    self.command.velocity.z,
                ]
            ),
            path=np.array(
                [[pose.pose.position.x, pose.pose.position.y] for pose in self.path.poses]
            ),
        )

    def on_tick(self) -> None:
        """Decide once, then publish exactly one command."""
        if self.odometry is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        candidate = self.candidate()
        reason = reject_reason(
            candidate, now, self.timeout, self.max_speed, self.grid, self.origin, self.resolution
        )
        measured = np.array(
            [
                self.odometry.pose.pose.position.x,
                self.odometry.pose.pose.position.y,
                self.odometry.pose.pose.position.z,
            ]
        )
        velocity = np.array(
            [
                self.odometry.twist.twist.linear.x,
                self.odometry.twist.twist.linear.y,
                self.odometry.twist.twist.linear.z,
            ]
        )

        if reason is None:
            if self.state != PASS:
                self.get_logger().info(f"guard clear after: {self.last_reason}")
            self.state = PASS
            self.last_safe_position = measured
            self.hover_position = None
            self.publisher.publish(self.command)
        else:
            if self.state == PASS:
                self.get_logger().warn(f"guard engaged: {reason}")
            self.last_reason = reason
            speed = float(np.linalg.norm(velocity))
            if self.hover_position is None and speed >= STOPPED_SPEED:
                self.state = BRAKE
                self.publisher.publish(self.brake_command(measured, velocity, speed))
            else:
                if self.hover_position is None:
                    # Latch once, at the last position known to be safe.
                    self.hover_position = (
                        self.last_safe_position
                        if self.last_safe_position is not None
                        else measured
                    )
                    self.state = HOVER
                self.publisher.publish(self.hover_command(self.hover_position))
        self.publish_diagnostics(reason)

    def publish_diagnostics(self, reason: str | None) -> None:
        """Report the guard state at the guard's own rate, which the recorder integrates."""
        status = DiagnosticStatus(
            level=DiagnosticStatus.OK if self.state == PASS else DiagnosticStatus.WARN,
            name="safety_shield",
            hardware_id="guard",
            values=[
                KeyValue(key="guard_state", value=self.state),
                KeyValue(key="reason", value=reason or ""),
                KeyValue(key="rate", value=f"{self.rate:.6f}"),
            ],
        )
        array = DiagnosticArray(status=[status])
        array.header.stamp = self.get_clock().now().to_msg()
        self.diagnostics.publish(array)

    def brake_command(
        self, measured: np.ndarray, velocity: np.ndarray, speed: float
    ) -> PositionCommand:
        """Command zero velocity with the configured deceleration opposing motion."""
        command = self.blank()
        command.position.x, command.position.y = float(measured[0]), float(measured[1])
        command.position.z = float(self.altitude)
        direction = velocity / speed
        command.acceleration.x = float(-self.brake_accel * direction[0])
        command.acceleration.y = float(-self.brake_accel * direction[1])
        command.yaw = self.current_yaw()
        return command

    def hover_command(self, position: np.ndarray) -> PositionCommand:
        """Hold the latched position with no feedforward."""
        command = self.blank()
        command.position.x, command.position.y = float(position[0]), float(position[1])
        command.position.z = float(self.altitude)
        command.yaw = self.current_yaw()
        return command

    def blank(self) -> PositionCommand:
        command = PositionCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "world"
        return command

    def current_yaw(self) -> float:
        orientation = self.odometry.pose.pose.orientation
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def main() -> None:
    """Run the safety shield node."""
    rclpy.init()
    node = SafetyShield()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # The recorder tears the context down under us; rclpy can be mid-take when it
        # goes. Only tolerate it once the context is actually gone -- a RuntimeError while
        # still running is a real fault and must propagate.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        # The recorder ends the run by shutting the context down, so by the time the other
        # nodes unwind it is already closed; shutting down twice raises.
        if rclpy.ok():
            rclpy.shutdown()

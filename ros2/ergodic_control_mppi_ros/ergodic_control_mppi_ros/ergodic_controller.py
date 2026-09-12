"""Run the ergodic MPPI controller online against the simulator odometry.

The node owns no control law of its own: it folds the received safety grid into the
workspace parameters and drives the same ``single_step`` the offline runs use. The one
deliberate difference from the offline loop is that the fading memory records the position
the vehicle actually reached, not the one the model predicted.
"""

import math
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.single import initialize_single, single_step
from ergodic_control_mppi.simulation import controller_key, select_device

LATEST = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
TRANSIENT_LOCAL = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
SAFETY_PATH_SECONDS = 1.0
PLAN_PERIOD = 0.2
WARMUP_STEPS = 200
# Stop timing early once the loop is hopeless rather than grinding through every warmup
# step to confirm it. A configuration this far over budget will never be flown.
WARMUP_MIN_STEPS = 20
WARMUP_ABORT_FACTOR = 5.0


def yaw_of(orientation) -> float:
    """Extract yaw from a quaternion message."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def observation_from(odometry: Odometry) -> np.ndarray:
    """Build the controller's six-state observation from odometry."""
    position = odometry.pose.pose.position
    linear = odometry.twist.twist.linear
    return np.array(
        [
            position.x,
            position.y,
            linear.x,
            linear.y,
            yaw_of(odometry.pose.pose.orientation),
            odometry.twist.twist.angular.z,
        ],
        dtype=np.float32,
    )


def grid_from(message: OccupancyGrid) -> tuple[np.ndarray, tuple[float, float], float]:
    """Unpack an ``OccupancyGrid`` into a float occupancy array, origin, and resolution."""
    data = np.asarray(message.data, dtype=np.int8).reshape(message.info.height, message.info.width)
    origin = (message.info.origin.position.x, message.info.origin.position.y)
    return (data > 0).astype(np.float32), origin, float(message.info.resolution)


def limit_yaw_rate(current: float, target: float, max_rate: float, delta_t: float) -> float:
    """Step ``current`` toward ``target`` by at most ``max_rate * delta_t``, wrapping."""
    error = math.atan2(math.sin(target - current), math.cos(target - current))
    step = max(-max_rate * delta_t, min(max_rate * delta_t, error))
    return math.atan2(math.sin(current + step), math.cos(current + step))


def state_trace_u32(step_index: int, predicted: np.ndarray, observed: np.ndarray) -> str:
    """Encode the predicted and consumed float32 states without losing ULPs."""
    bits = np.concatenate(
        (
            np.asarray([step_index], dtype=np.uint32),
            np.asarray(predicted, dtype=np.float32).view(np.uint32),
            np.asarray(observed, dtype=np.float32).view(np.uint32),
        )
    )
    return ",".join(str(int(value)) for value in bits)


class ErgodicController(Node):
    """Publish raw ergodic setpoints and the matching one-second safety path."""

    def __init__(self) -> None:
        super().__init__("ergodic_controller")
        self.config_path = self.declare_parameter(
            "config", "/workspace/configs/mppi_params.yaml"
        ).value
        self.device = self.declare_parameter("device", "auto").value
        self.altitude = self.declare_parameter("altitude", 0.75).value
        self.deadline_ms = self.declare_parameter("deadline_ms", 16.0).value
        self.max_yaw_rate = self.declare_parameter("max_yaw_rate", math.pi).value
        self.seed = int(self.declare_parameter("seed", -1).value)
        self.preflight_steps = int(self.declare_parameter("preflight_steps", 0).value)
        self.predicted_feedback = bool(
            self.declare_parameter("predicted_feedback", False).value
        )
        if self.preflight_steps < 0:
            raise ValueError("preflight_steps must be non-negative")

        self.config = load_config(self.config_path)
        self.delta_t = float(self.config.controller.model.delta_t)
        if self.seed < 0:
            self.seed = int(self.config.run.seed)

        self.odometry: Odometry | None = None
        self.pending_grid: OccupancyGrid | None = None
        self.carry = None
        self.compiled = False
        self.deadline_ok = False
        self.compile_seconds = float("nan")
        self.commanded_yaw = 0.0
        self.step_times: list[float] = []
        self.last_publish = None
        self.plan_due = 0.0

        subscriptions = ReentrantCallbackGroup()
        control = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Odometry, "/sim/odom", self.on_odometry, LATEST, callback_group=subscriptions
        )
        self.create_subscription(
            OccupancyGrid,
            "/ergodic/safety_grid",
            self.on_grid,
            TRANSIENT_LOCAL,
            callback_group=subscriptions,
        )
        self.command_publisher = self.create_publisher(PositionCommand, "/ergodic/cmd_raw", LATEST)
        self.path_publisher = self.create_publisher(Path, "/ergodic/safety_path", LATEST)
        self.plan_publisher = self.create_publisher(Path, "/ergodic/plan", LATEST)
        self.diagnostics = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_timer(self.delta_t, self.on_control, callback_group=control)

    def on_odometry(self, message: Odometry) -> None:
        self.odometry = message
        if self.pending_grid is not None and not self.compiled:
            pending, self.pending_grid = self.pending_grid, None
            self.on_grid(pending)

    def on_grid(self, message: OccupancyGrid) -> None:
        """Compile against the received grid and verify the loop can hold the deadline."""
        if self.compiled:
            return
        if self.odometry is None:
            self.pending_grid = message
            self.get_logger().info("safety grid received; waiting for measured odometry")
            return
        occupancy, origin, resolution = grid_from(message)
        self.get_logger().info(
            f"safety grid {occupancy.shape} at {resolution} m, {int(occupancy.sum())} blocked cells"
        )
        selected = select_device(self.device)
        self.selected_device = selected
        workspace = replace(
            self.config.controller.workspace,
            grid=jnp.asarray(occupancy),
            grid_origin=jnp.asarray(origin, dtype=jnp.float32),
            grid_resolution=resolution,
        )
        self.params = jax.device_put(
            replace(self.config.controller, workspace=workspace), selected
        )
        self.device_label = selected.platform

        started = time.perf_counter()

        def feedback_step(params, carry, observation):
            """Apply measured feedback inside the compiled online transition."""
            corrected = carry._replace(
                state=observation, memory=carry.memory.at[-1].set(observation[:2])
            )
            return single_step(params, corrected)

        self.step = jax.jit(feedback_step, static_argnums=())
        state = observation_from(self.odometry)
        device_state = jax.device_put(state, selected)
        stationary_memory = jnp.broadcast_to(
            device_state[:2], (self.params.mppi.memory_length, 2)
        )
        zero_step = jax.device_put(np.int32(0), selected)
        initial_carry = jax.device_put(
            initialize_single(
                self.params,
                device_state,
                jnp.zeros((self.params.mppi.horizon, 3), dtype=jnp.float32),
                controller_key(self.seed),
            ),
            selected,
        )
        # Compile without consuming the carry that defines the experiment.
        compiled_carry, _ = self.step(self.params, initial_carry, device_state)
        jax.block_until_ready(compiled_carry.state)
        self.compile_seconds = time.perf_counter() - started

        # Startup timing gate: a loop that cannot hold the deadline must not fly.
        durations = []
        carry = initial_carry
        for index in range(WARMUP_STEPS):
            begin = time.perf_counter()
            carry, _ = self.step(self.params, carry, device_state)
            carry = carry._replace(
                state=device_state,
                memory=stationary_memory,
                step_index=zero_step,
            )
            jax.block_until_ready(carry.state)
            durations.append((time.perf_counter() - begin) * 1e3)
            if (
                index + 1 >= WARMUP_MIN_STEPS
                and float(np.median(durations)) > WARMUP_ABORT_FACTOR * self.deadline_ms
            ):
                self.get_logger().warn(
                    f"abandoning warmup after {index + 1} steps: median "
                    f"{np.median(durations):.1f} ms is over {WARMUP_ABORT_FACTOR:g}x the "
                    f"{self.deadline_ms:.1f} ms deadline"
                )
                break
        percentile = float(np.percentile(durations, 99))
        self.deadline_ok = percentile <= self.deadline_ms
        message_text = (
            f"compiled in {self.compile_seconds:.1f} s on {self.device_label}; "
            f"warmup p99 {percentile:.2f} ms against a {self.deadline_ms:.1f} ms deadline"
        )
        if not self.deadline_ok:
            self.get_logger().fatal(
                f"{message_text} -- latching hover, no commands will be published. "
                "Reduce mppi.T / mppi.K or run on a faster device."
            )
            return
        self.get_logger().info(message_text)

        if self.preflight_steps == WARMUP_STEPS:
            self.carry = carry
        else:
            self.carry = initial_carry
            for _ in range(self.preflight_steps):
                self.carry, _ = self.step(self.params, self.carry, device_state)
                self.carry = self.carry._replace(
                    state=device_state,
                    memory=stationary_memory,
                    step_index=zero_step,
                )
                jax.block_until_ready(self.carry.state)
        self.commanded_yaw = float(state[4])
        self.compiled = True

    def on_control(self) -> None:
        """Solve one step and publish the raw command with its matching safety path."""
        if self.carry is None or not self.deadline_ok or self.odometry is None:
            return
        # Replace the model's prediction with what was measured, in the state and in the
        # newest fading-memory sample, so the buffer holds executed positions.
        predicted = np.asarray(jax.device_get(self.carry.state), dtype=np.float32)
        observed = (
            predicted if self.predicted_feedback else observation_from(self.odometry)
        )
        observation = jax.device_put(observed, self.selected_device)
        begin = time.perf_counter()
        self.carry, result = self.step(self.params, self.carry, observation)
        trajectory = np.asarray(jax.block_until_ready(result.optimal_trajectory))
        control = np.asarray(result.control)
        elapsed = (time.perf_counter() - begin) * 1e3
        self.step_times.append(elapsed)

        stamp = self.get_clock().now()
        self.publish_command(stamp, trajectory, control)
        self.publish_diagnostics(
            stamp,
            elapsed,
            result.weights,
            state_trace_u32(int(self.carry.step_index), predicted, observed),
        )
        now = stamp.nanoseconds * 1e-9
        if now >= self.plan_due:
            self.publish_path(self.plan_publisher, stamp, trajectory[:, :2])
            self.plan_due = now + PLAN_PERIOD

    def publish_command(self, stamp, trajectory: np.ndarray, control: np.ndarray) -> None:
        """Publish the setpoint and the one-second safety path under one shared stamp."""
        setpoint = trajectory[0]
        # Yaw tracks the path tangent rather than the model's yaw state, rate limited so
        # the attitude controller is never handed a step.
        tangent = trajectory[min(4, trajectory.shape[0] - 1), :2] - setpoint[:2]
        if float(np.linalg.norm(tangent)) > 1e-6:
            target = math.atan2(float(tangent[1]), float(tangent[0]))
            previous = self.commanded_yaw
            self.commanded_yaw = limit_yaw_rate(
                previous, target, self.max_yaw_rate, self.delta_t
            )
            yaw_rate = math.atan2(
                math.sin(self.commanded_yaw - previous), math.cos(self.commanded_yaw - previous)
            ) / self.delta_t
        else:
            yaw_rate = 0.0

        command = PositionCommand()
        command.header.stamp = stamp.to_msg()
        command.header.frame_id = "world"
        command.position.x = float(setpoint[0])
        command.position.y = float(setpoint[1])
        command.position.z = float(self.altitude)
        command.velocity.x = float(setpoint[2])
        command.velocity.y = float(setpoint[3])
        command.velocity.z = 0.0
        command.acceleration.x = float(control[0])
        command.acceleration.y = float(control[1])
        command.acceleration.z = 0.0
        command.yaw = float(self.commanded_yaw)
        command.yaw_dot = float(yaw_rate)
        self.command_publisher.publish(command)

        horizon = min(int(round(SAFETY_PATH_SECONDS / self.delta_t)), trajectory.shape[0])
        self.publish_path(self.path_publisher, stamp, trajectory[:horizon, :2])

    def publish_path(self, publisher, stamp, positions: np.ndarray) -> None:
        """Publish planar positions as a world-frame path at the flight altitude."""
        path = Path()
        path.header.stamp = stamp.to_msg()
        path.header.frame_id = "world"
        for x, y in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(self.altitude)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        publisher.publish(path)

    def publish_diagnostics(self, stamp, elapsed: float, weights, state_trace: str) -> None:
        now = stamp.nanoseconds * 1e-9
        jitter = 0.0 if self.last_publish is None else abs(now - self.last_publish - self.delta_t)
        self.last_publish = now
        weights_array = np.asarray(jax.device_get(weights))
        ess_fraction = 1.0 / (
            float(np.sum(weights_array * weights_array)) * self.params.mppi.samples
        )
        temperature = float(self.carry.temperature)
        status = DiagnosticStatus(
            level=DiagnosticStatus.OK if elapsed <= self.deadline_ms else DiagnosticStatus.WARN,
            name="ergodic_controller",
            hardware_id=getattr(self, "device_label", "unknown"),
            values=[
                KeyValue(key="compile_seconds", value=f"{self.compile_seconds:.6f}"),
                KeyValue(key="step_ms", value=f"{elapsed:.6f}"),
                KeyValue(key="jitter_ms", value=f"{jitter * 1e3:.6f}"),
                KeyValue(key="deadline_ok", value=str(elapsed <= self.deadline_ms)),
                KeyValue(key="step_index", value=str(int(self.carry.step_index))),
                KeyValue(key="state_trace_u32", value=state_trace),
                KeyValue(
                    key="ess_fraction",
                    value=f"{ess_fraction:.8f}",
                ),
                KeyValue(key="temperature", value=f"{temperature:.8f}"),
                KeyValue(
                    key="temperature_at_cap",
                    value=str(
                        temperature
                        >= float(self.params.mppi.temperature_max) * (1.0 - 1e-6)
                    ),
                ),
            ],
        )
        array = DiagnosticArray(status=[status])
        array.header.stamp = stamp.to_msg()
        self.diagnostics.publish(array)


def main() -> None:
    """Run the ergodic controller node."""
    rclpy.init()
    node = ErgodicController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
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

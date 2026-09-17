"""
Ergodic MPPI mission framework node for the MULLET stack.

It serves ``LoadMission`` and ``ListMissions``, keeps every accepted spec in its mission
library, runs the preflight, and while mullet_core reports ``MISSION`` it streams one
``MissionCommand`` per controller step. Every topic and service path comes from
this drone's heartbeat, never from a guessed namespace. Nothing here overrides mullet_core:
leaving ``MISSION`` pauses the mission, and every refusal ends in mullet_core's own hold.
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from mullet_interfaces.msg import DroneHeartbeat, MissionCommand, MissionStatus
from mullet_interfaces.srv import ListMissions, LoadMission
from px4_msgs.msg import VehicleLocalPosition
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.mission import (
    SCHEMA,
    STALE_POSITION_S,
    altitude_failure,
    command_position,
    command_vectors,
    compile_flight,
    compile_mission,
    dimension,
    dry_run_failure,
    plan_blocked,
    replans_from_measurement,
    start_failure,
    unpack_flight,
)
from ergodic_control_mppi.deploy.store import save_spec, stored_specs
from ergodic_control_mppi.mppi.single import initialize_single
from ergodic_control_mppi.simulation import controller_key

FRAMEWORK = "ergodic"
# The boot self-test: the fast, safety-relevant modules (config validation, grid geometry,
# mission compilation and flight rules, a closed-loop smoke run). The heavy numerical suites
# already passed when the image was built.
SELF_TEST = (
    "tests.test_config", "tests.test_controllers", "tests.test_deploy_grid", "tests.test_mission",
    "tests.test_store",
)
WARMUP_STEPS = 200
# Stop timing early once the loop is hopeless rather than grinding through every step.
WARMUP_MIN_STEPS = 20
WARMUP_ABORT_FACTOR = 5.0
PLAN_CHECK_SECONDS = 1.0
STATUS_PERIOD_S = 0.5
MODES = {
    "trajectory": MissionCommand.TRAJECTORY,
    "trajectory_3d": MissionCommand.TRAJECTORY_3D,
    "velocity": MissionCommand.VELOCITY,
    "acceleration": MissionCommand.ACCELERATION,
    "velocity_3d": MissionCommand.VELOCITY_3D,
    "acceleration_3d": MissionCommand.ACCELERATION_3D,
}
PX4_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE
)
HEARTBEAT_QOS = QoSProfile(
    depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)
COMMAND_QOS = PX4_QOS


def join_path(namespace: str, name: str) -> str:
    """Join a ROS namespace and a relative name into one absolute name."""
    return "/" + "/".join(part for part in (*namespace.split("/"), *name.split("/")) if part)


def px4_topic(name: str, message_type) -> str:
    """PX4's DDS name for a message topic: a ``MESSAGE_VERSION`` above 0 adds ``_v<N>``.

    The version comes from the px4_msgs build, which is pinned to the firmware's line
    (VehicleLocalPosition is v1 from PX4 1.17, so its topic is ``vehicle_local_position_v1``).
    """
    version = getattr(message_type, "MESSAGE_VERSION", 0)
    return f"{name}_v{version}" if version else name


def failed_tests(output: str) -> list[str]:
    """Names of the failing tests in unittest's report."""
    return [
        line.split(":", 1)[1].strip()
        for line in output.splitlines()
        if line.startswith(("FAIL:", "ERROR:"))
    ]


def warmup_p99(step, params, carry, observation, deadline_ms: float) -> float:
    """99th-percentile time of a compiled step and its host copy, in ms, over the warmup."""
    durations = []
    for index in range(WARMUP_STEPS):
        begin = time.perf_counter()
        carry, result = step(params, carry, observation)
        jax.device_get(result)
        durations.append((time.perf_counter() - begin) * 1e3)
        if (
            index + 1 >= WARMUP_MIN_STEPS
            and float(np.median(durations)) > WARMUP_ABORT_FACTOR * deadline_ms
        ):
            break
    return float(np.percentile(durations, 99))


def observation_from(position: VehicleLocalPosition, d: int, yaw_rate: float = 0.0) -> np.ndarray:
    """Controller state ``[p(d), v(d), yaw, yaw_rate]`` from PX4's local position."""
    heading = position.heading if math.isfinite(position.heading) else 0.0
    return np.array(
        [
            *(position.x, position.y, position.z)[:d],
            *(position.vx, position.vy, position.vz)[:d],
            heading,
            yaw_rate,
        ],
        dtype=np.float32,
    )


def position_problem(position, received_at: float, now: float) -> str | None:
    """Why the latest local position cannot be flown on, or ``None`` when it can."""
    if position is None:
        return "no local position from PX4 yet"
    if not (position.xy_valid and position.v_xy_valid):
        return "PX4 reports its horizontal position or velocity invalid"
    if not (position.z_valid and position.v_z_valid):
        return "PX4 reports its altitude or vertical velocity invalid"
    if now - received_at > STALE_POSITION_S:
        return f"PX4 local position is {now - received_at:.2f} s old"
    return None


class ErgodicMission(Node):
    """Load, check and fly ergodic MPPI missions for one drone."""

    def __init__(self) -> None:
        super().__init__("ergodic_mission")
        self.drone_id = self.declare_parameter("drone_id", "").value
        if not self.drone_id:
            raise ValueError("drone_id must match this drone's heartbeat drone_id")
        self.workspace = Path(self.declare_parameter("workspace", "/workspace").value)
        base_config = self.declare_parameter("base_config", "configs/uav_profile.yaml").value
        self.base = load_config(self.workspace / base_config).controller
        self.deadline_ms = float(self.declare_parameter("deadline_ms", 16.0).value)
        self.preflight_seconds = float(self.declare_parameter("preflight_seconds", 20.0).value)
        self.missions_dir = Path(self.declare_parameter("missions_dir", "/workspace/missions").value)
        self.delta_t = float(self.base.model.delta_t)
        self.plan_steps = max(1, round(PLAN_CHECK_SECONDS / self.delta_t))
        self.key = controller_key(0)

        self.lock = threading.Lock()
        self.state, self.detail = "SELF_TEST", "running the unit suite"
        self.self_test_passed: bool | None = None
        self.preflight: list[str] = []
        self.device_label, self.warmup_ms = "", 0.0
        self.step_ms: deque[float] = deque(maxlen=500)
        self.progress = 0.0
        self.steps = self.re_anchors = 0
        self.predicted_state: np.ndarray | None = None
        self.mission = self.params = self.step = self.carry = self.device = None
        self.reference_stamp = None
        self.heartbeat: DroneHeartbeat | None = None
        self.position: VehicleLocalPosition | None = None
        self.position_received = -math.inf
        self.last_position_stamp = None
        self.load_path = self.list_path = ""
        self.command_publisher = self.hold_client = None

        self.callbacks = ReentrantCallbackGroup()
        self.create_subscription(
            DroneHeartbeat, "/mullet/heartbeat", self.on_heartbeat, HEARTBEAT_QOS,
            callback_group=self.callbacks,
        )
        self.status_publisher = self.create_publisher(MissionStatus, "/mullet/mission_status", 10)
        self.create_timer(STATUS_PERIOD_S, self.publish_status, callback_group=self.callbacks)
        self.create_timer(
            self.delta_t, self.on_tick, callback_group=MutuallyExclusiveCallbackGroup()
        )
        threading.Thread(target=self.self_test, daemon=True).start()

    # --- Lifecycle ------------------------------------------------------------------------

    def set_state(self, state: str, detail: str) -> None:
        """Move to ``state`` and log the change. Callers hold the lock."""
        changed = (state, detail) != (self.state, self.detail)
        self.state, self.detail = state, detail
        # One call site per severity: rclpy refuses a call site whose severity changes.
        if changed and state == "FAULT":
            self.get_logger().error(f"{state}: {detail}")
        elif changed:
            self.get_logger().info(f"{state}: {detail}")

    def self_test(self) -> None:
        """Run the self-test modules once, on the CPU; a failure refuses every load."""
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "unittest", *SELF_TEST],
            cwd=self.workspace,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            capture_output=True,
            text=True,
        )
        with self.lock:
            self.self_test_passed = result.returncode == 0
            if self.self_test_passed:
                self.set_state("EMPTY", f"self-test passed in {time.monotonic() - started:.0f} s")
            else:
                names = ", ".join(failed_tests(result.stderr)) or result.stderr[-300:]
                self.set_state("FAULT", f"self-test failed: {names}")

    def abort(self, state: str, detail: str) -> None:
        """Stop flying the mission and hand the drone to mullet_core's hold."""
        self.set_state(state, detail)
        if self.hold_client is not None:
            self.hold_client.call_async(Trigger.Request())

    # --- Drone interfaces -----------------------------------------------------------------

    def on_heartbeat(self, message: DroneHeartbeat) -> None:
        if message.drone_id != self.drone_id:
            return
        with self.lock:
            first = self.heartbeat is None
            self.heartbeat = message
            if first:
                self.wire(message)

    def wire(self, heartbeat: DroneHeartbeat) -> None:
        """Create every drone-specific interface from the paths the heartbeat carries."""
        self.create_subscription(
            VehicleLocalPosition,
            px4_topic(heartbeat.px4_fmu_prefix + "/fmu/out/vehicle_local_position", VehicleLocalPosition),
            self.on_position,
            PX4_QOS,
            callback_group=self.callbacks,
        )
        self.command_publisher = self.create_publisher(
            MissionCommand, join_path(heartbeat.ros_namespace, "command"), COMMAND_QOS
        )
        self.hold_client = self.create_client(
            Trigger, heartbeat.hold_service_path, callback_group=self.callbacks
        )
        self.load_path = join_path(heartbeat.ros_namespace, f"mission/{FRAMEWORK}/load")
        self.create_service(LoadMission, self.load_path, self.on_load, callback_group=self.callbacks)
        self.list_path = join_path(heartbeat.ros_namespace, f"mission/{FRAMEWORK}/list")
        self.create_service(
            ListMissions, self.list_path, self.on_list, callback_group=self.callbacks
        )
        self.get_logger().info(f"wired to '{self.drone_id}', serving {self.load_path}")

    def on_position(self, message: VehicleLocalPosition) -> None:
        self.position, self.position_received = message, time.monotonic()

    # --- Load and preflight ---------------------------------------------------------------

    def load_refusal(self, position: VehicleLocalPosition | None) -> str | None:
        if self.self_test_passed is None:
            return "the self-test is still running"
        if not self.self_test_passed:
            return self.detail
        if self.state == "PREPARING":
            return "a preflight is already running"
        if self.heartbeat.state_label == "MISSION":
            return "the drone is in MISSION; hold it before loading"
        problem = position_problem(position, self.position_received, time.monotonic())
        if problem is not None:
            return problem
        if not position.xy_global:
            return "PX4 has no global origin yet (xy_global is false)"
        return None

    def on_load(self, request: LoadMission.Request, response: LoadMission.Response):
        with self.lock:
            position = self.position
            reason = self.load_refusal(position)
            if reason is None:
                try:
                    spec = json.loads(request.spec_json)
                    mission = compile_mission(spec, self.base, (position.ref_lat, position.ref_lon))
                except (ValueError, TypeError) as error:
                    reason = str(error)
            if reason is not None:
                self.get_logger().warn(f"load refused: {reason}")
                response.accepted, response.reason = False, reason
                return response
            self.mission, self.params, self.carry = mission, None, None
            self.reference_stamp = position.ref_timestamp
            self.preflight, self.device_label, self.warmup_ms = ["spec: ok"], "", 0.0
            self.step_ms.clear()
            self.progress, self.steps, self.re_anchors = 0.0, 0, 0
            self.set_state("PREPARING", f"preflight of '{mission.mission_id}'")
            # The drone may be anywhere at load: the model flight starts at the heaviest mode.
            d = dimension(mission.command_mode)
            state = np.array([*mission.dry_run_start, *[0.0] * (d + 2)], dtype=np.float32)
            try:
                save_spec(self.missions_dir, spec, datetime.now(timezone.utc))
            except OSError as error:  # the library is a convenience; it never refuses a mission
                self.get_logger().warn(f"could not store '{mission.mission_id}': {error}")
        threading.Thread(target=self.run_preflight, args=(mission, state), daemon=True).start()
        response.accepted, response.mission_id = True, mission.mission_id
        return response

    def on_list(self, request: ListMissions.Request, response: ListMissions.Response):
        """Every stored spec, newest first."""
        try:
            stored = stored_specs(self.missions_dir)
        except OSError as error:
            self.get_logger().warn(f"could not read the mission library: {error}")
            stored = []
        response.names = [name for name, _ in stored]
        response.specs_json = [spec_json for _, spec_json in stored]
        return response

    def run_preflight(self, mission, state: np.ndarray) -> None:
        """Pick a device that holds the deadline, then fly the mission in the model."""
        checks, chosen = [], None
        try:
            d = dimension(mission.command_mode)
            zeros = jnp.zeros((mission.params.mppi.horizon, d + 1), dtype=jnp.float32)
            for device in self.devices():
                params = jax.device_put(mission.params, device)
                carry = jax.device_put(
                    initialize_single(params, jnp.asarray(state), zeros, self.key), device
                )
                # Compiles every argument kind flight passes; none may compile mid-flight.
                step = compile_flight(params, carry, state, self.plan_steps)
                p99 = warmup_p99(step, params, carry, state, self.deadline_ms)
                verdict = "ok" if p99 <= self.deadline_ms else "FAIL"
                checks.append(
                    f"device: {verdict} {device.platform} p99 {p99:.1f} ms "
                    f"(deadline {self.deadline_ms:.1f} ms)"
                )
                if verdict == "ok":
                    chosen = (device, params, step, p99)
                    break
            if chosen is None:
                self.finish_preflight(mission, checks, None, "no device holds the control deadline")
                return
            device, params, step, p99 = chosen
            failure = dry_run_failure(
                mission._replace(params=params), jnp.asarray(state), self.key, self.preflight_seconds
            )
            checks.append(
                f"dry_run: {'ok' if failure is None else 'FAIL ' + failure} "
                f"({self.preflight_seconds:.0f} s)"
            )
            self.finish_preflight(mission, checks, chosen, failure)
        except Exception as error:  # a crashed preflight must end in FAULT, never hang
            checks.append(f"preflight: FAIL {error!r}")
            self.finish_preflight(mission, checks, None, repr(error))

    def finish_preflight(self, mission, checks: list[str], chosen, failure: str | None) -> None:
        with self.lock:
            if self.mission is not mission:
                return
            self.preflight.extend(checks)
            if failure is not None:
                self.set_state("FAULT", f"preflight failed: {failure}")
                return
            self.device, self.params, self.step, self.warmup_ms = chosen
            self.device_label = self.device.platform
            self.set_state("READY", f"'{mission.mission_id}' ready; start_mission flies it")

    @staticmethod
    def devices() -> list:
        """The GPU first when there is one, then the CPU fallback."""
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = []
        return gpus[:1] + jax.devices("cpu")[:1]

    # --- Flight ---------------------------------------------------------------------------

    def on_tick(self) -> None:
        with self.lock:
            if self.heartbeat is None or self.mission is None:
                return
            in_mission = self.heartbeat.state_label == "MISSION"
            if self.state == "RUNNING" and not in_mission:
                self.set_state("PAUSED", "mullet_core left MISSION; start_mission resumes")
                return
            if self.state in ("READY", "PAUSED") and in_mission and not self.start():
                return
            if self.state == "RUNNING":
                self.fly()

    def start(self) -> bool:
        """Apply the start gate on every MISSION entry, first start or resume."""
        position = self.position
        d = dimension(self.mission.command_mode)
        failure = position_problem(position, self.position_received, time.monotonic())
        if failure is None:
            failure = start_failure(
                self.mission,
                (position.x, position.y, position.z)[:d],
                -position.z,
                time.monotonic() - self.position_received,
                position.ref_timestamp != self.reference_stamp,
            )
        if failure is not None:
            self.abort("FAULT", f"start refused: {failure}")
            return False
        if self.carry is None:  # first start: plan from where the drone is now
            zeros = jnp.zeros((self.params.mppi.horizon, d + 1), dtype=jnp.float32)
            self.predicted_state = observation_from(position, d)
            self.carry = jax.device_put(
                initialize_single(self.params, jnp.asarray(self.predicted_state), zeros, self.key),
                self.device,
            )
        self.set_state("RUNNING", f"flying '{self.mission.mission_id}'")
        return True

    def fly(self) -> None:
        """One controller step: guard, solve, check the plan, publish."""
        begin = time.perf_counter()
        mission, position = self.mission, self.position
        problem = position_problem(position, self.position_received, time.monotonic())
        if problem is not None:
            # No command: mullet_core holds once the stream goes stale.
            self.get_logger().warn(f"skipping a step: {problem}", throttle_duration_sec=1.0)
            return
        if position.ref_timestamp != self.reference_stamp:
            self.abort("FAULT", "PX4 moved its EKF origin mid-mission; load the mission again")
            return
        ceiling = altitude_failure(mission, -position.z)
        if ceiling is not None:
            self.abort("FAULT", f"{ceiling}; drone held")
            return
        d = dimension(mission.command_mode)
        # Feed each PX4 measurement once; between measurements, step from the model's own
        # prediction rather than re-applying a position the vehicle has already left. The
        # trajectory modes keep stepping from that prediction, their reference, until the drone
        # strays past the tracking allowance. Every host-device copy costs milliseconds on the
        # Jetson: at most one upload, one packed download.
        observation = self.carry.state
        if position.timestamp != self.last_position_stamp:
            self.last_position_stamp = position.timestamp
            measured = observation_from(position, d, float(self.predicted_state[2 * d + 1]))
            if replans_from_measurement(mission, self.predicted_state, measured):
                observation = measured
                if mission.command_mode.startswith("trajectory"):
                    self.re_anchors += 1
                    self.get_logger().warn(
                        f"drone strayed past the {mission.tracking_allowance:.2f} m tracking allowance: "
                        f"plan re-anchored ({self.re_anchors} so far)",
                        throttle_duration_sec=2.0,
                    )
        self.carry, packed = self.step(self.params, self.carry, observation)
        plan, self.predicted_state, control = unpack_flight(np.asarray(packed), self.plan_steps, d)
        if plan_blocked(mission, plan):
            self.abort("FAULT", "the plan comes within the drone's clearance of an obstacle; drone held")
            return
        velocity, acceleration = command_vectors(
            mission, self.predicted_state, control, (position.vx, position.vy, position.vz)[:d]
        )
        command = MissionCommand()
        command.stamp = self.get_clock().now().to_msg()
        command.mode = MODES[mission.command_mode]
        command.position_ned_m = command_position(mission, self.predicted_state).tolist()
        command.velocity_ned_mps = velocity.tolist()
        command.acceleration_ned_mps2 = acceleration.tolist()
        command.yaw_ned_rad = math.nan
        self.command_publisher.publish(command)
        tick_ms = (time.perf_counter() - begin) * 1e3
        self.step_ms.append(tick_ms)
        if tick_ms > self.delta_t * 1e3:
            self.get_logger().warn(
                f"tick took {tick_ms:.1f} ms, over the {self.delta_t * 1e3:.0f} ms period",
                throttle_duration_sec=5.0,
            )
        self.steps += 1
        elapsed = self.steps * self.delta_t
        self.progress = min(elapsed / mission.duration_s, 1.0)
        if elapsed >= mission.duration_s:
            self.abort("DONE", f"'{mission.mission_id}' completed {mission.duration_s:.0f} s")

    # --- Status ---------------------------------------------------------------------------

    def publish_status(self) -> None:
        with self.lock:
            status = MissionStatus()
            status.stamp = self.get_clock().now().to_msg()
            status.drone_id, status.framework, status.spec_schema = self.drone_id, FRAMEWORK, SCHEMA
            status.load_service_path = self.load_path
            status.list_service_path = self.list_path
            status.mission_id = self.mission.mission_id if self.mission is not None else ""
            status.state_label, status.detail = self.state, self.detail
            status.progress = float(self.progress)
            status.preflight = list(self.preflight)
            status.device = self.device_label
            status.step_ms_p99 = (
                float(np.percentile(self.step_ms, 99)) if self.step_ms else float(self.warmup_ms)
            )
        self.status_publisher.publish(status)


def main() -> None:
    """Run the ergodic mission node."""
    rclpy.init()
    node = ErgodicMission()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

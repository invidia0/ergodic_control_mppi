#!/usr/bin/env python3
"""End-to-end check of the ergodic mission framework against mullet_core, in SITL or HIL.

Needs one ROS domain with PX4 SITL + mullet_core (``mullet_sitl_sim/run.sh --headless --config
config/paddock.yaml --domain 42``) and this repo's mission container wired to the same drone
(``DRONE_ID``). Takes off; then, for each command mode, loads a mission around the drone, waits
for the preflight, starts, measures the command rate for 15 s, holds, resumes and runs to DONE;
then lands. Every path comes from the drone's heartbeat and the framework's MissionStatus. Not
part of the unit suite. Exits non-zero on any failed expectation.

    python3 tests/sitl/sitl_e2e.py --drone uav_1
"""

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from mullet_interfaces.action import Land, Takeoff
from mullet_interfaces.msg import DroneHeartbeat, MissionCommand, MissionStatus
from mullet_interfaces.srv import LoadMission
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

M_PER_DEG = 6371000.0 * math.pi / 180.0
CIRCLE = (6.0, -6.0, 1.5)  # north, east, radius (m) from the load position
ROBOT_RADIUS = 0.45
MAX_SPEED = 1.5
HALF_SIDE = 15.0
RELIABLE = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
BEST_EFFORT = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


class Driver(Node):
    def __init__(self, drone: str):
        super().__init__("ergodic_sitl_e2e")
        self.drone, self.hb, self.ms = drone, None, None
        self.track, self.commands = [], []
        self.takeoff = self.land = self.load = self.start = self.hold = None
        self.create_subscription(DroneHeartbeat, "/mullet/heartbeat", self.on_heartbeat, RELIABLE)
        self.create_subscription(MissionStatus, "/mullet/mission_status", self.on_status, 10)

    def on_heartbeat(self, message):
        if message.drone_id != self.drone:
            return
        if self.hb is None:
            self.takeoff = ActionClient(self, Takeoff, message.takeoff_action_path)
            self.land = ActionClient(self, Land, message.land_action_path)
            self.start = self.create_client(Trigger, message.start_mission_service_path)
            self.hold = self.create_client(Trigger, message.hold_service_path)
            command = "/" + "/".join(p for p in (*message.ros_namespace.split("/"), "command") if p)
            self.create_subscription(
                MissionCommand, command, lambda _: self.commands.append(time.monotonic()), BEST_EFFORT
            )
        self.hb = message
        self.track.append((time.monotonic(), message.state_label, message.x, message.y, message.z,
                           message.command_stream_valid))

    def on_status(self, message):
        if message.drone_id != self.drone or message.framework != "ergodic":
            return
        if self.load is None:
            self.load = self.create_client(LoadMission, message.load_service_path)
        self.ms = message


def spin_until(node, predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    hb = node.hb.state_label if node.hb else None
    ms = (node.ms.state_label, node.ms.detail, list(node.ms.preflight)) if node.ms else None
    sys.exit(f"TIMEOUT waiting for {what}; heartbeat={hb} mission={ms}")


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def call(node, client, request, timeout=30.0):
    if not client.wait_for_service(timeout_sec=10.0):
        sys.exit(f"{client.srv_name} unavailable")
    future = client.call_async(request)
    spin_until(node, future.done, timeout, client.srv_name)
    return future.result()


def send_goal(node, client, goal):
    if not client.wait_for_server(timeout_sec=10.0):
        sys.exit("action server unavailable")
    future = client.send_goal_async(goal)
    spin_until(node, future.done, 10.0, "goal response")
    if not future.result().accepted:
        sys.exit("goal rejected")


def spec(lat0, lon0, mode):
    k_lon = M_PER_DEG * math.cos(math.radians(lat0))

    def at(north, east):
        return [lat0 + north / M_PER_DEG, lon0 + east / k_lon]

    s = HALF_SIDE
    return {
        "schema": "ergodic/1",
        "mission_id": f"sitl-{mode}",
        "command_mode": mode,
        "duration_s": 40,
        "vehicle": {"radius_m": ROBOT_RADIUS, "clearance_m": 0.15, "tracking_allowance_m": 0.2,
                    "max_speed_mps": MAX_SPEED, "max_accel_mps2": 3.0, "brake_accel_mps2": 3.0,
                    "reaction_time_s": 0.1},
        "area": [at(-s, -s), at(s, -s), at(s, s), at(-s, s)],
        "obstacles": [
            {"type": "circle", "center": at(CIRCLE[0], CIRCLE[1]), "radius_m": CIRCLE[2]},
            {"type": "polygon", "points": [at(-4, 6), at(-2, 6), at(-2, 9), at(-4, 9)]},
        ],
        "density": [
            {"mean": at(8, 8), "sigma_major_m": 4, "sigma_minor_m": 2, "bearing_deg": 30, "weight": 2},
            {"mean": at(-8, -6), "sigma_major_m": 3, "sigma_minor_m": 3, "bearing_deg": 0, "weight": 1},
        ],
    }


def fly(node, mode, failures):
    lat0, lon0, x0, y0 = node.hb.lat, node.hb.lon, node.hb.x, node.hb.y
    response = call(node, node.load, LoadMission.Request(spec_json=json.dumps(spec(lat0, lon0, mode))))
    print(f"[{mode}] load accepted={response.accepted} reason='{response.reason}'", flush=True)
    if not response.accepted:
        failures.append(f"{mode}: load refused: {response.reason}")
        return
    begin = time.monotonic()
    spin_until(node, lambda: node.ms.mission_id == f"sitl-{mode}"
               and node.ms.state_label in ("READY", "FAULT"), 300, "preflight")
    print(f"[{mode}] preflight {node.ms.state_label} in {time.monotonic() - begin:.0f} s: "
          f"{list(node.ms.preflight)} device={node.ms.device} p99={node.ms.step_ms_p99:.1f} ms", flush=True)
    if node.ms.state_label != "READY":
        failures.append(f"{mode}: preflight {node.ms.detail}")
        return

    node.track.clear()
    print(f"[{mode}] start_mission: {call(node, node.start, Trigger.Request()).success}", flush=True)
    spin_until(node, lambda: node.ms.state_label == "RUNNING", 10, "RUNNING")
    node.commands.clear()
    spin_for(node, 15.0)
    gaps = np.diff(node.commands) if len(node.commands) > 1 else np.array([math.inf])
    print(f"[{mode}] command rate {len(node.commands) / 15.0:.1f} Hz over 15 s, "
          f"gap p50 {np.percentile(gaps, 50) * 1e3:.0f} ms, max {gaps.max() * 1e3:.0f} ms", flush=True)
    print(f"[{mode}] hold: {call(node, node.hold, Trigger.Request()).success}", flush=True)
    spin_until(node, lambda: node.ms.state_label == "PAUSED" and node.hb.state_label == "HOLD", 10, "PAUSED")
    spin_for(node, 3.0)
    print(f"[{mode}] resume: {call(node, node.start, Trigger.Request()).success}", flush=True)
    spin_until(node, lambda: node.ms.state_label == "RUNNING", 10, "RUNNING again")
    # Mission time counts controller steps; a slow host stretches it in wall time.
    spin_until(node, lambda: node.ms.state_label in ("DONE", "FAULT"), 240, "DONE")
    spin_until(node, lambda: node.hb.state_label == "HOLD", 10, "HOLD after DONE")
    print(f"[{mode}] finished {node.ms.state_label}: {node.ms.detail} p99={node.ms.step_ms_p99:.1f} ms", flush=True)
    if node.ms.state_label != "DONE":
        failures.append(f"{mode}: {node.ms.detail}")

    flown = [(t, x - x0, y - y0, z) for t, state, x, y, z, _ in node.track if state == "MISSION"]
    fresh = [valid for _, state, *_, valid in node.track if state == "MISSION"]
    # Over ~0.5 s windows: heartbeats arrive at 10 Hz with reception jitter.
    speeds = [math.hypot(b[1] - a[1], b[2] - a[2]) / max(b[0] - a[0], 1e-3) for a, b in zip(flown, flown[5:])]
    clearance = min(math.hypot(n - CIRCLE[0], e - CIRCLE[1]) - CIRCLE[2] - ROBOT_RADIUS for _, n, e, _ in flown)
    extent = max(max(abs(n), abs(e)) for _, n, e, _ in flown)
    travelled = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(flown, flown[1:]))
    altitude = [z for *_, z in flown]
    print(f"[{mode}] {len(flown)} samples, travelled {travelled:.1f} m, max speed {max(speeds):.2f} m/s, "
          f"min clearance to circle {clearance:.2f} m, max |n|,|e| {extent:.1f} m, "
          f"altitude {min(altitude):.2f}..{max(altitude):.2f} m NED, stream fresh {sum(fresh)}/{len(fresh)}", flush=True)
    if clearance <= 0.0:
        failures.append(f"{mode}: touched the obstacle (clearance {clearance:.2f} m)")
    if extent >= HALF_SIDE:
        failures.append(f"{mode}: left the area ({extent:.1f} m)")
    if max(speeds) > 1.3 * MAX_SPEED:
        failures.append(f"{mode}: speed {max(speeds):.2f} m/s")
    if travelled < 5.0:
        failures.append(f"{mode}: barely moved ({travelled:.1f} m)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drone", default="uav_1", help="heartbeat drone_id to fly (default uav_1)")
    parser.add_argument("--modes", nargs="+", default=["velocity", "acceleration"],
                        choices=["velocity", "acceleration"])
    args = parser.parse_args()

    rclpy.init()
    node = Driver(args.drone)
    failures = []
    spin_until(node, lambda: node.hb and node.hb.can_takeoff and node.ms and node.ms.state_label == "EMPTY",
               180, f"{args.drone} ready for takeoff and an EMPTY ergodic mission node")
    send_goal(node, node.takeoff, Takeoff.Goal(altitude_m=3.0, yaw_rad=0.0, use_current_yaw=True))
    spin_until(node, lambda: node.hb.state_label == "HOLD", 60, "HOLD after takeoff")
    print(f"takeoff done: z={node.hb.z:.2f} m NED at {node.hb.lat:.7f}, {node.hb.lon:.7f}", flush=True)
    for mode in args.modes:
        fly(node, mode, failures)
    send_goal(node, node.land, Land.Goal())
    spin_until(node, lambda: node.hb.state_label == "IDLE", 90, "IDLE after land")
    print("landed", flush=True)
    print("RESULT: PASS" if not failures else "RESULT: FAIL\n  " + "\n  ".join(failures), flush=True)
    rclpy.shutdown()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end check of the ergodic mission framework against mullet_core, in SITL or HIL.

Needs one ROS domain with PX4 SITL + mullet_core (``mullet_sitl_sim/run.sh --headless --config
config/paddock.yaml --domain 42``) and this repo's mission container wired to the same drone
(``DRONE_ID``). Takes off; checks that a mission whose area is away from the drone loads but
refuses to start, and that the framework lists what it loaded; then, for each command mode,
loads a mission around the drone, waits for the preflight, starts, measures the command rate for
15 s, holds, resumes and runs to DONE; then lands. The 3D modes must also change altitude, stay
under the ceiling, and hold altitude while paused. Every path comes from the drone's heartbeat and the framework's MissionStatus. Not
part of the unit suite. Exits non-zero on any failed expectation.

The scene is the Paddock's field fixture (``tests/fixtures/paddock_3d_spec.json``: 48 x 24 m,
8 m ceiling, six pillars of which three can be flown over, three blobs at 3.5-4.5 m), moved so
its centre is where the drone is at load.

    python3 tests/sitl/sitl_e2e.py --drone uav_1
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from mullet_interfaces.action import Land, Takeoff
from mullet_interfaces.msg import DroneHeartbeat, MissionCommand, MissionStatus
from mullet_interfaces.srv import ListMissions, LoadMission
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

M_PER_DEG = 6371000.0 * math.pi / 180.0
SCENE = json.loads((Path(__file__).resolve().parent.parent / "fixtures" / "paddock_3d_spec.json").read_text())
SCENE_CENTRE = (44.63, 10.95)  # the fixture's area centre


def scene_ned(lat, lon):
    """Metres north, east of the fixture's centre (flat Earth: the scene is 48 m across)."""
    return ((lat - SCENE_CENTRE[0]) * M_PER_DEG,
            (lon - SCENE_CENTRE[1]) * M_PER_DEG * math.cos(math.radians(SCENE_CENTRE[0])))


# north, east, radius, height (m), relative to the load position.
PILLARS = [(*scene_ned(*o["center"]), o["radius_m"], o["height_m"]) for o in SCENE["obstacles"]]
ROBOT_RADIUS = SCENE["vehicle"]["radius_m"]
MAX_SPEED = SCENE["vehicle"]["max_speed_mps"]
HALF_EXTENT = tuple(max(abs(scene_ned(*p)[i]) for p in SCENE["area"]) for i in (0, 1))
CEILING = SCENE["max_altitude_m"]
RELIABLE = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
DURATION_S, SAVE_DIR = 60.0, None
BEST_EFFORT = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


class Driver(Node):
    def __init__(self, drone: str):
        super().__init__("ergodic_sitl_e2e")
        self.drone, self.hb, self.ms = drone, None, None
        self.track, self.commands = [], []
        self.takeoff = self.land = self.load = self.list = self.start = self.hold = None
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
            self.list = self.create_client(ListMissions, message.list_service_path)
        self.ms = message


class Abort(Exception):
    """A step that cannot go on; main() records it and still ends the flight on the ground."""


def spin_until(node, predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    hb = node.hb.state_label if node.hb else None
    ms = (node.ms.state_label, node.ms.detail, list(node.ms.preflight)) if node.ms else None
    raise Abort(f"TIMEOUT waiting for {what}; heartbeat={hb} mission={ms}")


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def call(node, client, request, timeout=30.0):
    if not client.wait_for_service(timeout_sec=10.0):
        raise Abort(f"{client.srv_name} unavailable")
    future = client.call_async(request)
    spin_until(node, future.done, timeout, client.srv_name)
    return future.result()


def send_goal(node, client, goal):
    if not client.wait_for_server(timeout_sec=10.0):
        raise Abort("action server unavailable")
    future = client.send_goal_async(goal)
    spin_until(node, future.done, 10.0, "goal response")
    if not future.result().accepted:
        raise Abort("goal rejected")


def spec(lat0, lon0, mode, north0=0.0):
    """The field scene centred ``north0`` metres north of ``(lat0, lon0)``, flying ``mode``."""
    k_lon = M_PER_DEG * math.cos(math.radians(lat0))

    def at(point):
        north, east = scene_ned(*point[:2])
        return [round(lat0 + (north + north0) / M_PER_DEG, 8), round(lon0 + east / k_lon, 8), *point[2:]]

    document = json.loads(json.dumps(SCENE))
    document.update(mission_id=f"sitl-{mode}", command_mode=mode, duration_s=DURATION_S,
                    area=[at(p) for p in SCENE["area"]])
    for obstacle in document["obstacles"]:
        obstacle["center"] = at(obstacle["center"])
    for component in document["density"]:
        component["mean"] = at(component["mean"])
    if SAVE_DIR is not None:
        (SAVE_DIR / f"{document['mission_id']}{'-away' if north0 else ''}.json").write_text(json.dumps(document, indent=2))
    return document


def check_start_gate_and_library(node, failures):
    """A mission 60 m away loads, refuses to start, and shows up in the library."""
    away = spec(node.hb.lat, node.hb.lon, "velocity", north0=60.0)
    away["mission_id"] = "sitl-away"
    response = call(node, node.load, LoadMission.Request(spec_json=json.dumps(away)))
    if not response.accepted:
        failures.append(f"away: load refused: {response.reason}")
        return
    spin_until(node, lambda: node.ms.mission_id == "sitl-away"
               and node.ms.state_label in ("READY", "FAULT"), 300, "away preflight")
    listed = call(node, node.list, ListMissions.Request())
    print(f"[away] library newest: {list(listed.names[:1])}", flush=True)
    if not listed.names or json.loads(listed.specs_json[0])["mission_id"] != "sitl-away":
        failures.append("library: the loaded spec is not listed first")
    if node.ms.state_label != "READY":
        failures.append(f"away: preflight {node.ms.detail}")
        return
    call(node, node.start, Trigger.Request())
    spin_until(node, lambda: node.ms.state_label == "FAULT", 10, "start refusal")
    print(f"[away] start refused: {node.ms.detail}", flush=True)
    if "inside the area" not in node.ms.detail:
        failures.append(f"away: unexpected refusal '{node.ms.detail}'")
    spin_until(node, lambda: node.hb.state_label == "HOLD", 10, "HOLD after the refusal")


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
    spin_for(node, 1.0)
    paused_z = node.hb.z
    spin_for(node, 2.0)
    if mode.endswith("_3d") and abs(node.hb.z - paused_z) > 0.3:
        failures.append(f"{mode}: altitude drifted {node.hb.z - paused_z:+.2f} m while paused")
    print(f"[{mode}] resume: {call(node, node.start, Trigger.Request()).success}", flush=True)
    spin_until(node, lambda: node.ms.state_label == "RUNNING", 10, "RUNNING again")
    # Mission time counts controller steps; a slow host stretches it in wall time.
    spin_until(node, lambda: node.ms.state_label in ("DONE", "FAULT"), max(240, 4 * DURATION_S), "DONE")
    spin_until(node, lambda: node.hb.state_label == "HOLD", 10, "HOLD after DONE")
    print(f"[{mode}] finished {node.ms.state_label}: {node.ms.detail} p99={node.ms.step_ms_p99:.1f} ms", flush=True)
    if node.ms.state_label != "DONE":
        failures.append(f"{mode}: {node.ms.detail}")

    flown = [(t, x - x0, y - y0, z) for t, state, x, y, z, _ in node.track if state == "MISSION"]
    fresh = [valid for _, state, *_, valid in node.track if state == "MISSION"]
    # Over ~0.5 s windows: heartbeats arrive at 10 Hz with reception jitter.
    speeds = [math.hypot(b[1] - a[1], b[2] - a[2]) / max(b[0] - a[0], 1e-3) for a, b in zip(flown, flown[5:])]
    # A pillar is touched inside its radius plus the robot's; in 3D only below its top too.
    three_d = mode.endswith("_3d")
    clearance = min(
        math.hypot(n - pn, e - pe) - pr - ROBOT_RADIUS
        for _, n, e, z in flown
        for pn, pe, pr, ph in PILLARS
        if not three_d or -z < ph + ROBOT_RADIUS
    ) if flown else math.inf
    extent = max(max(abs(n) / HALF_EXTENT[0], abs(e) / HALF_EXTENT[1]) for _, n, e, _ in flown)
    travelled = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(flown, flown[1:]))
    altitude = [z for *_, z in flown]
    print(f"[{mode}] {len(flown)} samples, travelled {travelled:.1f} m, max speed {max(speeds):.2f} m/s, "
          f"min pillar clearance {clearance:.2f} m, max extent {extent:.2f} of the half area, "
          f"altitude {min(altitude):.2f}..{max(altitude):.2f} m NED, stream fresh {sum(fresh)}/{len(fresh)}", flush=True)
    if clearance <= 0.0:
        failures.append(f"{mode}: touched a pillar (clearance {clearance:.2f} m)")
    if extent >= 1.0:
        failures.append(f"{mode}: left the area ({extent:.2f} of the half extent)")
    if max(speeds) > 1.3 * MAX_SPEED:
        failures.append(f"{mode}: speed {max(speeds):.2f} m/s")
    if travelled < 5.0:
        failures.append(f"{mode}: barely moved ({travelled:.1f} m)")
    if -min(altitude) > CEILING:
        failures.append(f"{mode}: above the {CEILING:.0f} m ceiling ({-min(altitude):.2f} m)")
    if mode.endswith("_3d") and max(altitude) - min(altitude) < 0.5:
        failures.append(f"{mode}: never changed altitude")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drone", default="uav_1", help="heartbeat drone_id to fly (default uav_1)")
    parser.add_argument("--modes", nargs="+", default=["trajectory_3d", "trajectory", "velocity_3d"],
                        choices=["trajectory", "velocity", "acceleration", "trajectory_3d", "velocity_3d", "acceleration_3d"])
    parser.add_argument("--duration", type=float, default=60.0, help="mission length per mode, s")
    parser.add_argument("--save-spec", type=Path, help="write every loaded spec here (open it in the Paddock)")
    args = parser.parse_args()
    global DURATION_S, SAVE_DIR
    DURATION_S, SAVE_DIR = args.duration, args.save_spec
    if SAVE_DIR is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Driver(args.drone)
    failures = []
    try:
        spin_until(node, lambda: node.hb and node.hb.can_takeoff and node.ms and node.ms.state_label == "EMPTY",
                   180, f"{args.drone} ready for takeoff and an EMPTY ergodic mission node")
        send_goal(node, node.takeoff, Takeoff.Goal(altitude_m=3.0, yaw_rad=0.0, use_current_yaw=True))
        spin_until(node, lambda: node.hb.state_label == "HOLD", 60, "HOLD after takeoff")
        print(f"takeoff done: z={node.hb.z:.2f} m NED at {node.hb.lat:.7f}, {node.hb.lon:.7f}", flush=True)
        check_start_gate_and_library(node, failures)
        for mode in args.modes:
            try:
                fly(node, mode, failures)
            except Abort as error:
                # One mode failing (a FAULT mid-flight, say) must not strand the others.
                failures.append(f"{mode}: {error}")
                print(f"[{mode}] aborted: {error}", flush=True)
                if node.hb.state_label == "MISSION":
                    call(node, node.hold, Trigger.Request())
                spin_until(node, lambda: node.hb.state_label == "HOLD", 10, "HOLD after the abort")
    except Abort as error:
        failures.append(str(error))
        print(f"aborted: {error}", flush=True)
    finally:
        # Whatever failed, end the flight on the ground.
        if node.hb is not None and not node.hb.landed:
            try:
                send_goal(node, node.land, Land.Goal())
                spin_until(node, lambda: node.hb.state_label == "IDLE", 90, "IDLE after land")
                print("landed", flush=True)
            except Abort as error:
                failures.append(f"land: {error}")
                print(f"LAND FAILED: {error}", flush=True)
    print("RESULT: PASS" if not failures else "RESULT: FAIL\n  " + "\n  ".join(failures), flush=True)
    rclpy.shutdown()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

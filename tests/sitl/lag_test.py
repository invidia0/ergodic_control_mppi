#!/usr/bin/env python3
"""Measure how PX4 follows mullet_core MISSION commands: velocity steps and acceleration pulses.

Takes off, enters MISSION, streams MissionCommand at 50 Hz (VELOCITY_3D steps on each axis,
then ACCELERATION_3D pulses), holds, lands. Logs every tick: the command sent, the
TrajectorySetpoint mullet_core forwarded to PX4, and PX4's local position, velocity and
acceleration. Heading is held fixed so yaw never couples into the translation.

    python3 tests/sitl/lag_test.py --drone uav_0 --out /out/lag.npz
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from mullet_interfaces.action import Land, Takeoff
from mullet_interfaces.msg import DroneHeartbeat, MissionCommand
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

RATE_HZ = 50.0
NAN = float("nan")
RELIABLE = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
PX4 = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


def px4_topic(name, message_type):
    version = getattr(message_type, "MESSAGE_VERSION", 0)
    return f"{name}_v{version}" if version else name


def schedule(velocity_xy, velocity_z, accel_xy, accel_z):
    """(duration s, mode, velocity NED, acceleration NED) segments."""
    rest = 3.0
    segments = [(2.0, MissionCommand.VELOCITY_3D, (0, 0, 0), None)]
    for axis, amplitude in ((0, velocity_xy), (1, velocity_xy), (2, velocity_z)):
        for sign in (1, -1):
            v = [0.0, 0.0, 0.0]
            v[axis] = sign * amplitude
            segments += [(4.0, MissionCommand.VELOCITY_3D, tuple(v), None),
                         (rest, MissionCommand.VELOCITY_3D, (0, 0, 0), None)]
    for axis, amplitude in ((0, accel_xy), (1, accel_xy), (2, accel_z)):
        for sign in (1, -1):
            a = [0.0, 0.0, 0.0]
            a[axis] = sign * amplitude
            # Accelerate, then brake back to about zero velocity, then let it settle.
            segments += [(1.0, MissionCommand.ACCELERATION_3D, None, tuple(a)),
                         (1.0, MissionCommand.ACCELERATION_3D, None, tuple(-x for x in a)),
                         (rest, MissionCommand.VELOCITY_3D, (0, 0, 0), None)]
    return segments


class LagTest(Node):
    def __init__(self, drone):
        super().__init__("lag_test")
        self.drone, self.hb, self.position, self.setpoint = drone, None, None, None
        self.publisher = None
        self.create_subscription(DroneHeartbeat, "/mullet/heartbeat", self.on_heartbeat, RELIABLE)

    def on_heartbeat(self, message):
        if message.drone_id != self.drone:
            return
        if self.hb is None:
            prefix = message.px4_fmu_prefix
            self.create_subscription(VehicleLocalPosition, px4_topic(prefix + "/fmu/out/vehicle_local_position", VehicleLocalPosition),
                                     lambda m: setattr(self, "position", m), PX4)
            self.create_subscription(TrajectorySetpoint, px4_topic(prefix + "/fmu/in/trajectory_setpoint", TrajectorySetpoint),
                                     lambda m: setattr(self, "setpoint", m), PX4)
            command = "/" + "/".join(p for p in (*message.ros_namespace.split("/"), "command") if p)
            self.publisher = self.create_publisher(MissionCommand, command, PX4)
            self.takeoff = ActionClient(self, Takeoff, message.takeoff_action_path)
            self.land = ActionClient(self, Land, message.land_action_path)
            self.start = self.create_client(Trigger, message.start_mission_service_path)
            self.hold = self.create_client(Trigger, message.hold_service_path)
        self.hb = message


def spin_until(node, done, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if done():
            return True
    print(f"TIMEOUT waiting for {what}", flush=True)
    return False


def action(node, client, goal, what):
    if not client.wait_for_server(timeout_sec=10.0):
        return False
    future = client.send_goal_async(goal)
    if not spin_until(node, future.done, 10.0, what) or not future.result().accepted:
        return False
    result = future.result().get_result_async()
    return spin_until(node, result.done, 90.0, what)


def call(node, client):
    client.wait_for_service(timeout_sec=10.0)
    future = client.call_async(Trigger.Request())
    spin_until(node, future.done, 10.0, "service")
    return future.result() is not None and future.result().success


def fly(node, segments, yaw):
    """Stream the schedule at RATE_HZ; returns the log rows."""
    rows, period = [], 1.0 / RATE_HZ
    start = next_tick = time.monotonic()
    for duration, mode, velocity, acceleration in segments:
        segment_end = next_tick + duration
        while next_tick < segment_end:
            while time.monotonic() < next_tick:
                rclpy.spin_once(node, timeout_sec=max(0.0, next_tick - time.monotonic()))
            command = MissionCommand()
            command.stamp = node.get_clock().now().to_msg()
            command.mode = mode
            command.velocity_ned_mps = [float(x) for x in velocity] if velocity else [NAN] * 3
            command.acceleration_ned_mps2 = [float(x) for x in acceleration] if acceleration else [NAN] * 3
            command.yaw_ned_rad = yaw
            node.publisher.publish(command)
            p, s = node.position, node.setpoint
            rows.append([
                time.monotonic() - start, mode,
                *command.velocity_ned_mps, *command.acceleration_ned_mps2,
                *(list(s.velocity) + list(s.acceleration) if s else [NAN] * 6),
                *((p.x, p.y, p.z, p.vx, p.vy, p.vz, p.ax, p.ay, p.az, p.timestamp * 1e-6) if p else [NAN] * 10),
                node.hb.state_label == "MISSION",
            ])
            next_tick += period
    return rows


COLUMNS = ("t mode cmd_vx cmd_vy cmd_vz cmd_ax cmd_ay cmd_az sp_vx sp_vy sp_vz sp_ax sp_ay sp_az "
           "x y z vx vy vz ax ay az px4_t in_mission").split()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drone", default="uav_0")
    parser.add_argument("--out", default="/out/lag.npz")
    parser.add_argument("--velocity-xy", type=float, default=0.8, help="step amplitude, m/s (core clamps at max_vx/vy_ned_mps)")
    parser.add_argument("--velocity-z", type=float, default=0.4, help="step amplitude, m/s (core clamps at max_vz_ned_mps)")
    parser.add_argument("--accel-xy", type=float, default=1.0, help="pulse amplitude, m/s^2")
    parser.add_argument("--accel-z", type=float, default=0.5, help="pulse amplitude, m/s^2")
    parser.add_argument("--altitude", type=float, default=4.0)
    args = parser.parse_args()

    rclpy.init()
    node = LagTest(args.drone)
    code = 1
    try:
        if not spin_until(node, lambda: node.hb and node.hb.can_takeoff and node.position, 60, f"{args.drone} ready"):
            return
        if not action(node, node.takeoff, Takeoff.Goal(altitude_m=args.altitude, yaw_rad=0.0, use_current_yaw=True), "takeoff"):
            return
        spin_until(node, lambda: node.hb.state_label == "HOLD", 30, "HOLD")
        yaw = float(node.position.heading)
        if not call(node, node.start) or not spin_until(node, lambda: node.hb.state_label == "MISSION", 5, "MISSION"):
            print("start_mission refused", flush=True)
            return
        segments = schedule(args.velocity_xy, args.velocity_z, args.accel_xy, args.accel_z)
        print(f"streaming {sum(s[0] for s in segments):.0f} s of steps and pulses", flush=True)
        rows = fly(node, segments, yaw)
        np.savez(args.out, data=np.asarray(rows, dtype=np.float64), columns=np.array(COLUMNS))
        left = sum(1 for r in rows if not r[-1])
        print(f"logged {len(rows)} ticks to {args.out}; {left} ticks outside MISSION", flush=True)
        code = 0 if left == 0 else 1
    finally:
        # Always end on the ground, whatever failed above.
        if node.hb is not None:
            if node.hb.state_label == "MISSION":
                call(node, node.hold)
                spin_until(node, lambda: node.hb.state_label == "HOLD", 10, "HOLD")
            if not node.hb.landed:
                print("landing", flush=True)
                action(node, node.land, Land.Goal(), "land")
        rclpy.shutdown()
        sys.exit(code)


if __name__ == "__main__":
    main()

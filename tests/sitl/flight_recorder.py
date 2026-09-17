#!/usr/bin/env python3
"""Record how a drone tracks its mission commands, one row per MissionCommand.

Each row holds the command (position, velocity, acceleration), the TrajectorySetpoint
mullet_core forwarded to PX4 (after its clamps), PX4's local position, velocity and
acceleration, and the mullet_core and mission states. Runs next to a mission until
--seconds pass or it is stopped (SIGINT/SIGTERM), then writes the log. Analyse it with
tracking_report.py.

    python3 tests/sitl/flight_recorder.py --drone uav_0 --out /out/flight.npz
"""

import argparse
import math
import signal
import time

import numpy as np
import rclpy
from mullet_interfaces.msg import DroneHeartbeat, MissionCommand, MissionStatus
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

NAN = math.nan
RELIABLE = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
PX4 = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
COLUMNS = (
    "t mode cmd_x cmd_y cmd_z cmd_vx cmd_vy cmd_vz cmd_ax cmd_ay cmd_az "
    "sp_x sp_y sp_z sp_vx sp_vy sp_vz sp_ax sp_ay sp_az "
    "x y z vx vy vz ax ay az in_mission running mission"
).split()


def px4_topic(name, message_type):
    version = getattr(message_type, "MESSAGE_VERSION", 0)
    return f"{name}_v{version}" if version else name


class Recorder(Node):
    def __init__(self, drone):
        super().__init__("flight_recorder")
        self.drone, self.hb, self.status, self.position, self.setpoint = drone, None, None, None, None
        self.rows, self.missions, self.start = [], [], time.monotonic()
        self.create_subscription(DroneHeartbeat, "/mullet/heartbeat", self.on_heartbeat, RELIABLE)
        self.create_subscription(MissionStatus, "/mullet/mission_status", self.on_status, 10)

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
            self.create_subscription(MissionCommand, command, self.on_command, PX4)
            print(f"recording {command} for {self.drone}", flush=True)
        self.hb = message

    def on_status(self, message):
        if message.drone_id == self.drone and message.framework == "ergodic":
            self.status = message

    def on_command(self, command):
        p, s, status = self.position, self.setpoint, self.status
        mission = status.mission_id if status else ""
        if mission not in self.missions:
            self.missions.append(mission)
        self.rows.append([
            time.monotonic() - self.start, command.mode,
            *command.position_ned_m, *command.velocity_ned_mps, *command.acceleration_ned_mps2,
            *(list(s.position) + list(s.velocity) + list(s.acceleration) if s else [NAN] * 9),
            *((p.x, p.y, p.z, p.vx, p.vy, p.vz, p.ax, p.ay, p.az) if p else [NAN] * 9),
            self.hb is not None and self.hb.state_label == "MISSION",
            status is not None and status.state_label == "RUNNING",
            self.missions.index(mission),
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drone", default="uav_0")
    parser.add_argument("--out", default="/out/flight.npz")
    parser.add_argument("--seconds", type=float, default=3600.0)
    args = parser.parse_args()

    rclpy.init()
    node = Recorder(args.drone)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    end = time.monotonic() + args.seconds
    while not stop and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    np.savez(args.out, data=np.asarray(node.rows, dtype=np.float64), columns=np.array(COLUMNS),
             missions=np.array(node.missions))
    print(f"wrote {len(node.rows)} rows over {len(node.missions)} missions to {args.out}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

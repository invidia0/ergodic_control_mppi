"""A perfect tracker standing in for the quadrotor, to isolate what the airframe costs.

The flown system reaches its target modes later than the offline model and has never once
repeated a full tour, while the model does so routinely. Everything between the two that
could be measured from the archive has been ruled out -- the shield alters commands by
0.3 mm, the published setpoint is the same quantity the offline model executes, and the
controller replans every step. What is left is the vehicle, the odometry feedback, and the
loop timing, and those cannot be separated from a recording.

This node replaces ``so3_control`` and ``so3_quadrotor_simulator``: it consumes the shielded
``/position_cmd`` and republishes it as ``/sim/odom`` with zero dynamics and zero lag. The
rest of the deployment is untouched -- same controller, same shield, same DDS, same 50 Hz
real-time loop -- so a flight against this node differs from a normal flight *only* by the
airframe and its attitude loop. If the tours are still slow, the vehicle is exonerated and
the cause is in the feedback or the timing.

Semantics follow ``uav_simulator/fake_drone/poscmd_2_odom.cpp``, which does exactly this in
C++; it is reimplemented here only because that package is outside the container's colcon
build while this one is bind-mounted, so this needs no image rebuild.

Not a flight-worthy vehicle model and not a substitute for one: it cannot violate a thrust
limit, stall, or lag, so any real-time or tracking claim measured against it is vacuous. It
exists to answer one question.
"""

import math

import rclpy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

LATEST = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


class IdealVehicle(Node):
    """Echo the commanded state back as odometry, at the simulator's publish rate."""

    def __init__(self) -> None:
        super().__init__("ideal_vehicle")
        self.x = float(self.declare_parameter("init_x", -16.0).value)
        self.y = float(self.declare_parameter("init_y", 0.0).value)
        self.z = float(self.declare_parameter("init_z", 0.75).value)
        # Match so3_quadrotor_simulator's 1 kHz odometry so the controller's feedback is as
        # fresh as it is in a normal flight; a slower rate would confound the comparison
        # with an extra source of staleness.
        rate = float(self.declare_parameter("publish_rate", 1000.0).value)

        self.velocity = (0.0, 0.0)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.create_subscription(PositionCommand, "/position_cmd", self.on_command, LATEST)
        self.publisher = self.create_publisher(Odometry, "/sim/odom", LATEST)
        self.create_timer(1.0 / rate, self.publish)
        self.get_logger().warn(
            "ideal vehicle: commands are echoed as odometry with no dynamics. "
            "Timing and tracking numbers from this run mean nothing."
        )

    def on_command(self, command: PositionCommand) -> None:
        """Adopt the commanded state as the achieved state."""
        self.x = float(command.position.x)
        self.y = float(command.position.y)
        self.z = float(command.position.z)
        self.velocity = (float(command.velocity.x), float(command.velocity.y))
        self.yaw = float(command.yaw)
        self.yaw_rate = float(command.yaw_dot)

    def publish(self) -> None:
        """Publish the current state as odometry."""
        odometry = Odometry()
        odometry.header.stamp = self.get_clock().now().to_msg()
        odometry.header.frame_id = "world"
        odometry.pose.pose.position.x = self.x
        odometry.pose.pose.position.y = self.y
        odometry.pose.pose.position.z = self.z
        odometry.pose.pose.orientation.z = math.sin(0.5 * self.yaw)
        odometry.pose.pose.orientation.w = math.cos(0.5 * self.yaw)
        odometry.twist.twist.linear.x = self.velocity[0]
        odometry.twist.twist.linear.y = self.velocity[1]
        # The controller's observation reads yaw_rate off this field; leaving it at zero fed
        # back a yaw state the commanded motion contradicts.
        odometry.twist.twist.angular.z = self.yaw_rate
        self.publisher.publish(odometry)


def main() -> None:
    """Run the ideal vehicle node."""
    rclpy.init()
    node = IdealVehicle()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()

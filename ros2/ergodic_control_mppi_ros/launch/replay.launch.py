"""Replay a recorded trial for video and snapshots.

Deliberately starts nothing but playback and RViz: no simulator, no controller, no guard.
A figure produced here is a view of recorded data, never of a fresh run.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("ergodic_control_mppi_ros"))
    bag = LaunchConfiguration("bag")

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag", description="Path to the recorded bag directory"),
            DeclareLaunchArgument("rate", default_value="1.0"),
            ExecuteProcess(
                cmd=["ros2", "bag", "play", bag, "--rate", LaunchConfiguration("rate")],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["--display-config", str(share / "config" / "uav.rviz")],
            ),
        ]
    )

"""Launch the Perlin map, SO3 demo, target density, and optional RViz scene."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch(package: str, name: str) -> PythonLaunchDescriptionSource:
    return PythonLaunchDescriptionSource(
        str(Path(get_package_share_directory(package), "launch", name))
    )


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration("config")
    rviz = LaunchConfiguration("rviz")
    share = Path(get_package_share_directory("ergodic_control_mppi_ros"))

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config", default_value="/workspace/configs/mppi_params.yaml"
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            IncludeLaunchDescription(_launch("mockamap", "perlin3d.launch.py")),
            IncludeLaunchDescription(
                _launch("so3_quadrotor_simulator", "simulator_example.launch.py"),
                launch_arguments={"start_rviz": "false"}.items(),
            ),
            Node(
                package="so3_control",
                executable="control_example",
                name="control_example",
                output="screen",
            ),
            Node(
                package="ergodic_control_mppi_ros",
                executable="density_visualizer",
                name="density_visualizer",
                output="screen",
                parameters=[{"config": config}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["--display-config", str(share / "config" / "scene.rviz")],
                condition=IfCondition(rviz),
            ),
        ]
    )

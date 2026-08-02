"""Pillar-forest map, published on /mock_map so it drops in for perlin3d.

The obstacles are square columns extruded from below ground up to their sampled height, so
at any flight altitude inside that range they are full-height: a vehicle cannot climb over
one. That is what makes a planar planner lossless on this map, and it is why this is the
map the deployment is evaluated on.

`circle_num` is 0 by default. The upstream generator also scatters tilted rings, which are
overhead obstacles a fixed-altitude slice would either miss entirely or rasterize as a
phantom wall depending on where the slice cuts them.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    def number(name, cast=float):
        return ParameterValue(LaunchConfiguration(name), value_type=cast)

    return LaunchDescription(
        [
            # Named map_* to avoid colliding with the controller's own `seed` when this is
            # included from uav.launch.py -- an unscoped `seed` there silently overrode the
            # controller seed for a whole batch of runs before it was caught.
            DeclareLaunchArgument("map_seed", default_value="1"),
            DeclareLaunchArgument("obs_num", default_value="45"),
            DeclareLaunchArgument("circle_num", default_value="0"),
            DeclareLaunchArgument("lower_rad", default_value="0.3"),
            DeclareLaunchArgument("upper_rad", default_value="0.6"),
            # Above the 0.75 m flight altitude at every sample, so every pillar is
            # full-height from the planner's point of view.
            DeclareLaunchArgument("lower_hei", default_value="2.0"),
            DeclareLaunchArgument("upper_hei", default_value="3.0"),
            DeclareLaunchArgument("x_size", default_value="40.0"),
            DeclareLaunchArgument("y_size", default_value="20.0"),
            DeclareLaunchArgument("resolution", default_value="0.1"),
            DeclareLaunchArgument("min_distance", default_value="1.2"),
            Node(
                package="map_generator",
                executable="random_forest",
                name="random_forest",
                output="screen",
                # The adapter and the recorder both read /mock_map; remapping here means
                # neither has to know which generator produced the cloud.
                remappings=[("/map_generator/global_cloud", "/mock_map")],
                parameters=[
                    {"seed": number("map_seed", int)},
                    {"map/x_size": number("x_size")},
                    {"map/y_size": number("y_size")},
                    {"map/z_size": 4.0},
                    {"map/obs_num": number("obs_num", int)},
                    {"map/circle_num": number("circle_num", int)},
                    {"map/resolution": number("resolution")},
                    {"ObstacleShape/lower_rad": number("lower_rad")},
                    {"ObstacleShape/upper_rad": number("upper_rad")},
                    {"ObstacleShape/lower_hei": number("lower_hei")},
                    {"ObstacleShape/upper_hei": number("upper_hei")},
                    {"min_distance": number("min_distance")},
                    {"sensing/rate": 1.0},
                    {"sensing/radius": 10.0},
                ],
            ),
        ]
    )

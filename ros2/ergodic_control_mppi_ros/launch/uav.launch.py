"""Fly one ergodic coverage trial on the SO3 simulator and record it.

Headless by default: automated runs need no display. The recorder decides when the trial
is over and exits, which the ``OnProcessExit`` handler below turns into a full teardown, so
a launch started by a screening script always terminates on its own.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, LaunchConfigurationEquals, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Recorded for replay and for the paper figures; everything else is derivable.
BAG_TOPICS = [
    "/mock_map",
    "/ergodic/map_visual",
    "/target_density",
    "/sim/odom",
    "/path",
    "/robot",
    "/ergodic/safety_grid",
    "/ergodic/map_grid",
    "/ergodic/plan",
    "/ergodic/safety_path",
    "/ergodic/cmd_raw",
    "/position_cmd",
    "/diagnostics",
]


def _launch(package: str, name: str) -> PythonLaunchDescriptionSource:
    return PythonLaunchDescriptionSource(
        str(Path(get_package_share_directory(package), "launch", name))
    )


def _float(name: str):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _int(name: str):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("ergodic_control_mppi_ros"))
    config = LaunchConfiguration("config")
    altitude = LaunchConfiguration("altitude")
    run_id = LaunchConfiguration("run_id")
    output_root = LaunchConfiguration("output_root")

    arguments = [
        DeclareLaunchArgument("config", default_value="/workspace/configs/mppi_params.yaml"),
        DeclareLaunchArgument("run_id", default_value="run"),
        DeclareLaunchArgument("output_root", default_value="/workspace/results/uav"),
        DeclareLaunchArgument("profile", default_value="baseline"),
        DeclareLaunchArgument("steps", default_value="200"),
        DeclareLaunchArgument("seed", default_value="-1", description="-1 uses the config seed"),
        DeclareLaunchArgument("overwrite", default_value="false"),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("bag", default_value="false"),
        DeclareLaunchArgument("qualify_only", default_value="false"),
        # "so3" is the deployment. "ideal" swaps in a perfect tracker to measure what
        # the airframe costs; it is a diagnostic, never a source of a timing claim.
        DeclareLaunchArgument("vehicle", default_value="so3",
                              choices=["so3", "ideal"]),
        # Map. Frozen from the 511-520 arming sweep, re-derived after the float32 resolution
        # fix. Only 514 and 518 arm; the other eight bury a mode inside an inflated obstacle
        # (520 also blocks the start cell). 518 is preferred over 514 because it carries 153
        # occupied cells against 514's 97 -- same free fraction, but something to avoid.
        # See the README before changing either value.
        DeclareLaunchArgument("map_seed", default_value="518"),
        DeclareLaunchArgument("map_fill", default_value="0.002"),
        # perlin3d gives blobby noise whose obstacles do not span the flight altitude
        # cleanly; random_forest gives full-height pillars, which is what makes planning in
        # a plane lossless rather than merely convenient. `map_fill` applies only to
        # perlin3d, `obs_num` only to random_forest.
        DeclareLaunchArgument("map_source", default_value="perlin3d",
                              choices=["perlin3d", "random_forest"]),
        DeclareLaunchArgument("obs_num", default_value="45"),
        DeclareLaunchArgument("pillar_min_radius", default_value="0.3"),
        DeclareLaunchArgument("pillar_max_radius", default_value="0.6"),
        DeclareLaunchArgument("pillar_min_height", default_value="2.0"),
        DeclareLaunchArgument("pillar_max_height", default_value="3.0"),
        DeclareLaunchArgument("pillar_min_distance", default_value="1.2"),
        # Flight envelope and safety budget. brake_accel and max_speed feed BOTH the guard
        # and the map inflation, so they must stay single-sourced here.
        DeclareLaunchArgument("altitude", default_value="0.75"),
        # Free under seed 518 at fill 0.002; (-16, 0) is inside an inflated obstacle.
        DeclareLaunchArgument("start_x", default_value="-15.57"),
        DeclareLaunchArgument("start_y", default_value="0.42"),
        DeclareLaunchArgument("robot_radius", default_value="0.30"),
        DeclareLaunchArgument("clearance", default_value="0.15"),
        # Calibrated, not assumed: measured pos_p95_m is 0.015-0.024 m over the 8000-step
        # runs, so 0.05 is already 2x the observed error. The initial 0.20 guess was 8-13x
        # it and cost 0.15 m of inflation for nothing.
        DeclareLaunchArgument("tracking_allowance", default_value="0.05"),
        DeclareLaunchArgument("max_speed", default_value="2.0"),
        DeclareLaunchArgument("brake_accel", default_value="6.0"),
        DeclareLaunchArgument("reaction_time", default_value="0.10"),
        DeclareLaunchArgument("deadline_ms", default_value="16.0"),
        DeclareLaunchArgument("preflight_steps", default_value="0"),
        DeclareLaunchArgument("predicted_feedback", default_value="false"),
    ]

    safety_parameters = {
        "altitude": _float("altitude"),
        "robot_radius": _float("robot_radius"),
        "clearance": _float("clearance"),
        "tracking_allowance": _float("tracking_allowance"),
        "max_speed": _float("max_speed"),
        "brake_accel": _float("brake_accel"),
        "reaction_time": _float("reaction_time"),
    }

    recorder = Node(
        package="ergodic_control_mppi_ros",
        executable="recorder",
        name="recorder",
        output="screen",
        # Only the controller may touch the GPU. Every JAX process preallocates a fraction
        # of VRAM, so a second one starves the controller into an allocator death spiral.
        additional_env={"JAX_PLATFORMS": "cpu"},
        parameters=[
            {
                "config": config,
                "run_id": run_id,
                "output_root": output_root,
                "profile": LaunchConfiguration("profile"),
                "steps": _int("steps"),
                "seed": _int("seed"),
                "overwrite": ParameterValue(
                    LaunchConfiguration("overwrite"), value_type=bool
                ),
                "bag": ParameterValue(LaunchConfiguration("bag"), value_type=bool),
                "map_seed": _int("map_seed"),
                "map_fill": _float("map_fill"),
                "map_source": LaunchConfiguration("map_source"),
                "obs_num": _int("obs_num"),
                "pillar_min_radius": _float("pillar_min_radius"),
                "pillar_max_radius": _float("pillar_max_radius"),
                "pillar_min_height": _float("pillar_min_height"),
                "pillar_max_height": _float("pillar_max_height"),
                "pillar_min_distance": _float("pillar_min_distance"),
                "robot_radius": _float("robot_radius"),
                "deadline_ms": _float("deadline_ms"),
                "start_x": _float("start_x"),
                "start_y": _float("start_y"),
                "preflight_steps": _int("preflight_steps"),
                "clearance": _float("clearance"),
                "tracking_allowance": _float("tracking_allowance"),
                "max_speed": _float("max_speed"),
                "brake_accel": _float("brake_accel"),
                "reaction_time": _float("reaction_time"),
            }
        ],
        condition=UnlessCondition(LaunchConfiguration("qualify_only")),
    )

    return LaunchDescription(
        arguments
        + [
            # Scoped: perlin3d declares its own `seed`, which without a scope leaks back
            # out and silently overwrites the *controller* seed -- every run then shares the
            # map seed no matter what `seed:=` says. Forwarding stays on, or the parent's
            # map_seed/map_fill would not resolve inside the group.
            GroupAction(
                [
                    IncludeLaunchDescription(
                        _launch("mockamap", "perlin3d.launch.py"),
                        launch_arguments={
                            "seed": LaunchConfiguration("map_seed"),
                            "fill": LaunchConfiguration("map_fill"),
                        }.items(),
                    )
                ],
                scoped=True,
                condition=LaunchConfigurationEquals("map_source", "perlin3d"),
            ),
            # Pillar forest. Its own launch file already names the argument `map_seed`, so
            # there is no `seed` to leak and no scoping needed -- but the group is kept for
            # symmetry with the branch above.
            GroupAction(
                [
                    IncludeLaunchDescription(
                        _launch("map_generator", "random_forest.launch.py"),
                        launch_arguments={
                            "map_seed": LaunchConfiguration("map_seed"),
                            "obs_num": LaunchConfiguration("obs_num"),
                            "lower_rad": LaunchConfiguration("pillar_min_radius"),
                            "upper_rad": LaunchConfiguration("pillar_max_radius"),
                            "lower_hei": LaunchConfiguration("pillar_min_height"),
                            "upper_hei": LaunchConfiguration("pillar_max_height"),
                            "min_distance": LaunchConfiguration("pillar_min_distance"),
                        }.items(),
                    )
                ],
                scoped=True,
                condition=LaunchConfigurationEquals("map_source", "random_forest"),
            ),
            # The airframe. `vehicle:=ideal` swaps the attitude loop and quadrotor for a
            # perfect tracker, which isolates what the vehicle costs relative to the offline
            # model; see ideal_vehicle.py. Everything else in the deployment is unchanged.
            GroupAction(
                [
                    IncludeLaunchDescription(
                        _launch("so3_quadrotor_simulator", "simulator_example.launch.py"),
                        launch_arguments={
                            "start_rviz": "false",
                            "init_x": LaunchConfiguration("start_x"),
                            "init_y": LaunchConfiguration("start_y"),
                            "init_z": altitude,
                        }.items(),
                    )
                ],
                scoped=True,
                condition=IfCondition(
                    PythonExpression([
                        "'", LaunchConfiguration("qualify_only"), "' == 'false' and '",
                        LaunchConfiguration("vehicle"), "' == 'so3'",
                    ])
                ),
            ),
            Node(
                package="ergodic_control_mppi_ros",
                executable="ideal_vehicle",
                name="ideal_vehicle",
                output="screen",
                parameters=[{
                    "init_x": LaunchConfiguration("start_x"),
                    "init_y": LaunchConfiguration("start_y"),
                    "init_z": altitude,
                }],
                condition=IfCondition(
                    PythonExpression([
                        "'", LaunchConfiguration("qualify_only"), "' == 'false' and '",
                        LaunchConfiguration("vehicle"), "' == 'ideal'",
                    ])
                ),
            ),
            Node(
                package="ergodic_control_mppi_ros",
                executable="map_adapter",
                name="map_adapter",
                output="screen",
                additional_env={"JAX_PLATFORMS": "cpu"},
                parameters=[
                    {
                        "config": config,
                        "resolution": 0.15,
                        "vertical_half_extent": 0.20,
                        "start_x": _float("start_x"),
                        "start_y": _float("start_y"),
                        **safety_parameters,
                    }
                ],
            ),
            Node(
                package="ergodic_control_mppi_ros",
                executable="ergodic_controller",
                name="ergodic_controller",
                output="screen",
                parameters=[
                    {
                        "config": config,
                        "device": LaunchConfiguration("device"),
                        "altitude": _float("altitude"),
                        "deadline_ms": _float("deadline_ms"),
                        "seed": _int("seed"),
                        "preflight_steps": _int("preflight_steps"),
                        "predicted_feedback": ParameterValue(
                            LaunchConfiguration("predicted_feedback"), value_type=bool
                        ),
                    }
                ],
                condition=UnlessCondition(LaunchConfiguration("qualify_only")),
            ),
            Node(
                package="ergodic_control_mppi_ros",
                executable="safety_shield",
                name="safety_shield",
                output="screen",
                additional_env={"JAX_PLATFORMS": "cpu"},
                parameters=[
                    {
                        "rate": 100.0,
                        "command_timeout": 0.10,
                        "altitude": _float("altitude"),
                        "max_speed": _float("max_speed"),
                        "brake_accel": _float("brake_accel"),
                    }
                ],
                condition=UnlessCondition(LaunchConfiguration("qualify_only")),
            ),
            Node(
                package="ergodic_control_mppi_ros",
                executable="density_visualizer",
                name="density_visualizer",
                output="screen",
                additional_env={"JAX_PLATFORMS": "cpu"},
                parameters=[{"config": config}],
                condition=UnlessCondition(LaunchConfiguration("qualify_only")),
            ),
            recorder,
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-o",
                     PathJoinSubstitution([output_root, run_id, "bag"]), *BAG_TOPICS],
                output="screen",
                condition=IfCondition(LaunchConfiguration("bag")),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["--display-config", str(share / "config" / "uav.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            # The recorder owns the run length; its exit ends the launch.
            RegisterEventHandler(
                OnProcessExit(target_action=recorder, on_exit=[EmitEvent(event=Shutdown())]),
                condition=UnlessCondition(LaunchConfiguration("qualify_only")),
            ),
        ]
    )

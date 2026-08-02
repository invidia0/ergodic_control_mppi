from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # seed and fill are arguments because the flyable map is chosen by them: after the
    # safety inflation, a high fill leaves no connected free space for the planner.
    seed = DeclareLaunchArgument('seed', default_value='511')
    fill = DeclareLaunchArgument('fill', default_value='0.1')

    mockamap_node = launch_ros.actions.Node(
            package='mockamap', executable='mockamap_node',
            output='screen',
            parameters=[{'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int)},
                        # box edge length, unit meter
                        {'resolution': 0.15},
                        # map size unit meter
                        {'x_length': 40},
                        {'y_length': 20},
                        {'z_length': 4},
                        # 1 perlin noise 3D, 2 perlin box random map, 3 2d maze still developing
                        {'type': 1},
                        # 1 perlin noise parameters
                        # complexity:  base noise frequency,large value will be complex. typical 0.0 ~ 0.5
                        # fill:        infill persentage. typical: 0.4 ~ 0.0
                        # fractal:     large value will have more detail
                        # attenuation: for fractal attenuation. typical: 0.0 ~ 0.5
                        {'complexity': 0.07},
                        {'fill': ParameterValue(LaunchConfiguration('fill'), value_type=float)},
                        {'fractal': 1},
                        {'attenuation': 0.1}])

    ld = LaunchDescription()
    ld.add_action(seed)
    ld.add_action(fill)
    ld.add_action(mockamap_node)

    return ld

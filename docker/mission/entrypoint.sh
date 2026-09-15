#!/bin/bash
set -e
source /opt/ros2_base_ws/install/setup.bash
source /ros_ws/install/setup.bash
exec "$@"

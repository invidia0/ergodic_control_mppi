# Ergodic MPPI mission node for MULLET drones: arm64 on the Jetson, amd64 for SITL.
#
# ROS 2 Jazzy like mullet_core (JAX 0.11 needs its Python 3.12); the two containers talk Fast DDS
# over UDP on the host network. On the Jetson the image builds on MULLET's base, which carries
# px4_msgs (the PX4 release the Pixhawks run) and NVIDIA's CUDA 13 forward-compatibility
# libraries for the JetPack 6 driver; the host's NVIDIA container runtime supplies the driver.
FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM ghcr.io/skymule-ai/mullet-ros2-base:v0.5.0 AS base-arm64

# The base is published for arm64 only. On amd64 (the SITL harness on a laptop) rebuild the one
# part of it this image uses: px4_msgs, same branch (mullet_ros_base repos/ros2.repos) and path.
FROM ros:jazzy-ros-base AS base-amd64
RUN git clone --depth 1 --branch release/1.17 https://github.com/PX4/px4_msgs.git \
        /opt/ros2_base_ws/src/px4_msgs \
    && cd /opt/ros2_base_ws && . /opt/ros/jazzy/setup.sh && colcon build --merge-install

FROM base-${TARGETARCH}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=uv /uv /bin/
WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version ./
# Dependencies only: the package itself is imported from /workspace via PYTHONPATH. Only the
# lock files are copied first, so doc edits never invalidate the dependency and message layers.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --no-dev --no-install-project --extra cuda13

# Messages change rarely: build them in their own layer so code edits skip them.
COPY ros2/mullet-interfaces /ros_ws/src/mullet_interfaces
WORKDIR /ros_ws
RUN source /opt/ros2_base_ws/install/setup.bash \
    && colcon build --packages-select mullet_interfaces

COPY configs /workspace/configs
COPY ergodic_control_mppi /workspace/ergodic_control_mppi
COPY tests /workspace/tests
COPY ros2/ergodic_mission /ros_ws/src/ergodic_mission
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace
# Build-time tests: the full unit suite and the node's own, on the CPU. Red means no image.
RUN source /opt/ros2_base_ws/install/setup.bash \
    && colcon build --packages-select ergodic_mission \
    && source install/setup.bash \
    && cd /workspace && JAX_PLATFORMS=cpu python -m unittest discover -s tests \
    && cd /ros_ws/src/ergodic_mission && JAX_PLATFORMS=cpu python -m unittest discover -s test

COPY docker/mission/entrypoint.sh /entrypoint.sh
# Preallocate a fixed pool: on-demand allocation showed ~500 ms allocator stalls in the MPPI
# step. A quarter of the Orin's shared 8 GB leaves room for mullet_core and the OS.
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.25
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-c", "exec python -m ergodic_mission.mission_node --ros-args -p drone_id:=${DRONE_ID:?set DRONE_ID}"]

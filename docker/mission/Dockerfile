# Ergodic MPPI mission node for MULLET drones: arm64 on the Jetson, amd64 for SITL.
#
# JAX 0.11 needs Python 3.12, so this is a ROS 2 Jazzy image beside mullet_core's Humble one;
# the two talk Fast DDS over UDP on the host network. On the Jetson Orin Nano (JetPack 6.2.2,
# CUDA 12.6 driver) the GPU runs CUDA 13 through NVIDIA's user-space forward-compatibility
# libraries; the host's NVIDIA container runtime supplies the driver itself.
FROM ghcr.io/astral-sh/uv:0.11.26 AS uv
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TARGETARCH
# Must match the PX4 firmware on the drone; mullet_ros_base pins release/1.17.
ARG PX4_MSGS_REF=release/1.17
ARG CUDA_COMPAT_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-compat-orin-13-2_13.2.44290101-1_arm64.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl git python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        curl -fsSL -o /tmp/compat.deb "$CUDA_COMPAT_URL" \
        && dpkg-deb -x /tmp/compat.deb /opt/cuda-compat \
        && rm /tmp/compat.deb; \
    fi
ENV LD_LIBRARY_PATH=/opt/cuda-compat/usr/local/cuda-13.2/compat_orin

COPY --from=uv /uv /bin/
WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version ./
# Dependencies only: the package itself is imported from /workspace via PYTHONPATH. Only the
# lock files are copied first, so doc edits never invalidate the dependency and px4_msgs layers.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --no-dev --no-install-project --extra cuda13

# Messages change rarely: build them in their own layer so code edits skip px4_msgs.
RUN git clone --depth 1 --branch "$PX4_MSGS_REF" https://github.com/PX4/px4_msgs.git \
        /ros_ws/src/px4_msgs
COPY ros2/mullet-interfaces /ros_ws/src/mullet_interfaces
WORKDIR /ros_ws
RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --packages-select px4_msgs mullet_interfaces

COPY configs /workspace/configs
COPY ergodic_control_mppi /workspace/ergodic_control_mppi
COPY tests /workspace/tests
COPY ros2/ergodic_mission /ros_ws/src/ergodic_mission
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace
# Build-time tests: the full unit suite and the node's own, on the CPU. Red means no image.
RUN source /opt/ros/jazzy/setup.bash \
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

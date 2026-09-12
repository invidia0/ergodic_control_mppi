FROM ghcr.io/astral-sh/uv:0.11.26 AS uv
FROM osrf/ros:jazzy-desktop

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libarmadillo-dev \
        python3-colcon-common-extensions \
        ros-jazzy-pcl-ros \
        ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY ergodic_control_mppi ./ergodic_control_mppi
# The environment lives outside /workspace so bind-mounting a host checkout there cannot
# shadow it with the host's own .venv. Dependencies only: the project itself is imported
# from /workspace via PYTHONPATH below, so a bind-mounted checkout is live and there is
# never a second, stale copy in site-packages. The CUDA plugin falls back to CPU when no
# GPU is present, so one image serves both.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-dev --no-install-project --extra cuda13

COPY configs ./configs
COPY uav_simulator /ros_ws/src/uav_simulator
COPY ros2/ergodic_control_mppi_ros /ros_ws/src/ergodic_control_mppi_ros

WORKDIR /ros_ws
# --symlink-install so edits to bind-mounted Python nodes need no rebuild; tests stay on
# so `colcon test` can actually run the package's own regressions.
RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --packages-up-to ergodic_control_mppi_ros map_generator --symlink-install \
    && chmod -R a+w /ros_ws
# Writable by any uid so `colcon test` works when the container runs as the host user
# (see the `user:` mapping in compose.yaml), which colcon needs for build/ and log/.

COPY docker/ros2/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LIBGL_ALWAYS_SOFTWARE=1 \
    PYTHONPATH=/workspace:/opt/venv/lib/python3.12/site-packages \
    QT_X11_NO_MITSHM=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.75
# Preallocate a fixed fraction rather than allocating on demand: with on-demand allocation
# the MPPI step showed ~500 ms allocator stalls, which wrecked p99 while leaving p50 fine.

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "ergodic_control_mppi_ros", "scene.launch.py"]

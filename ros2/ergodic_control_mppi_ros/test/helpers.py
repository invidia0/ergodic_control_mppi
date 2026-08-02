"""Shared locations for the ROS package tests."""

import os
from pathlib import Path

# The container copies the repo to /workspace and the compose service bind-mounts it
# there, so that is the normal location. The override exists so the same tests can run
# from a checkout outside the container.
WORKSPACE = Path(os.environ.get("ERGODIC_WORKSPACE", "/workspace"))
CONFIG = str(WORKSPACE / "configs" / "mppi_params.yaml")
UAV_CONFIG = str(WORKSPACE / "configs" / "uav_profile.yaml")

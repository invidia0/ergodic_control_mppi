from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrialData:
    """
    Canonical input container for metric computation.

    robot_paths shape: (steps, robots, state_dim)
    """
    robot_paths: np.ndarray
    target_density_grid: np.ndarray
    map_x_limits: tuple[float, float]
    map_y_limits: tuple[float, float]
    obstacle_map: np.ndarray
    safety_radius: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    """
    Flat trial output row.
    """
    values: dict[str, float | int | str]


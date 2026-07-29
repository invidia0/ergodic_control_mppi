"""Experiment-independent metric inputs and aggregate evaluation."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ergodic_control_mppi.metrics.coordination import (
    compute_pairwise_min_distance,
    compute_pairwise_overlap,
    compute_pairwise_redundancy,
    compute_redundancy_metric,
    compute_safety_metric,
)
from ergodic_control_mppi.metrics.ergodicity import (
    compute_reachable_mask,
    compute_team_ergodic_error,
)


@dataclass(frozen=True)
class TrialData:
    """Canonical metric input; ``robot_paths`` has shape ``(N, R, 6)``."""

    robot_paths: np.ndarray
    target_density_grid: np.ndarray
    map_x_limits: tuple[float, float]
    map_y_limits: tuple[float, float]
    obstacle_map: np.ndarray
    safety_radius: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_trial_data(value: TrialData | Mapping[str, Any]) -> TrialData:
    if isinstance(value, TrialData):
        return value
    required = {
        "robot_paths", "target_density_grid", "map_x_limits", "map_y_limits",
        "obstacle_map", "safety_radius",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"missing trial_data keys: {sorted(missing)}")
    return TrialData(
        np.asarray(value["robot_paths"]),
        np.asarray(value["target_density_grid"]),
        tuple(value["map_x_limits"]),
        tuple(value["map_y_limits"]),
        np.asarray(value["obstacle_map"]),
        float(value["safety_radius"]),
        dict(value.get("metadata", {})),
    )


def compute_all_metrics(
    trial_data: TrialData | Mapping[str, Any], pairwise_d_thresh: float = 1.0
) -> dict[str, float]:
    """Compute every stable scalar experiment metric."""
    data = _as_trial_data(trial_data)
    target_shape = np.asarray(data.target_density_grid).shape
    bins = (target_shape[1], target_shape[0])
    reachable = compute_reachable_mask(
        data.obstacle_map, data.safety_radius, data.map_x_limits, data.map_y_limits, bins
    )
    return {
        "team_ergodic_error": compute_team_ergodic_error(
            data.robot_paths, data.target_density_grid, data.map_x_limits, data.map_y_limits,
            reachable_mask=reachable,
        ),
        "pairwise_overlap": compute_pairwise_overlap(
            data.robot_paths, data.map_x_limits, data.map_y_limits
        ),
        "safety_metric": compute_safety_metric(
            data.robot_paths, data.obstacle_map, data.safety_radius
        ),
        "redundancy_metric": compute_redundancy_metric(
            data.robot_paths, data.map_x_limits, data.map_y_limits
        ),
        "R_pair": compute_pairwise_redundancy(data.robot_paths, pairwise_d_thresh),
        "D_min_pair": compute_pairwise_min_distance(data.robot_paths),
    }

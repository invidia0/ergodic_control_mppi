from __future__ import annotations

from typing import Mapping, Any

import numpy as np

from experiments.trial_types import TrialData
from metrics.ergodicity import compute_team_ergodic_error
from metrics.overlap import compute_pairwise_overlap
from metrics.pairwise import compute_pairwise_min_distance, compute_pairwise_redundancy
from metrics.safety import compute_safety_metric
from metrics.redundancy import compute_redundancy_metric


def _to_trial_data(trial_data: TrialData | Mapping[str, Any]) -> TrialData:
    if isinstance(trial_data, TrialData):
        return trial_data
    required = {
        "robot_paths",
        "target_density_grid",
        "map_x_limits",
        "map_y_limits",
        "obstacle_map",
        "safety_radius",
    }
    missing = required - set(trial_data.keys())
    if missing:
        raise ValueError(f"missing trial_data keys: {sorted(missing)}")
    return TrialData(
        robot_paths=np.asarray(trial_data["robot_paths"]),
        target_density_grid=np.asarray(trial_data["target_density_grid"]),
        map_x_limits=tuple(trial_data["map_x_limits"]),
        map_y_limits=tuple(trial_data["map_y_limits"]),
        obstacle_map=np.asarray(trial_data["obstacle_map"]),
        safety_radius=float(trial_data["safety_radius"]),
        metadata=dict(trial_data.get("metadata", {})),
    )


def compute_all_metrics(
    trial_data: TrialData | Mapping[str, Any],
    pairwise_d_thresh: float = 1.0,
) -> dict[str, float]:
    """
    Aggregate all scalar metrics for one trial.
    """
    td = _to_trial_data(trial_data)
    return {
        "team_ergodic_error": compute_team_ergodic_error(
            td.robot_paths,
            td.target_density_grid,
            td.map_x_limits,
            td.map_y_limits,
        ),
        "pairwise_overlap": compute_pairwise_overlap(
            td.robot_paths,
            td.map_x_limits,
            td.map_y_limits,
        ),
        "safety_metric": compute_safety_metric(
            td.robot_paths,
            td.obstacle_map,
            td.safety_radius,
        ),
        "redundancy_metric": compute_redundancy_metric(
            td.robot_paths,
            td.map_x_limits,
            td.map_y_limits,
        ),
        "R_pair": compute_pairwise_redundancy(
            td.robot_paths,
            d_thresh=pairwise_d_thresh,
        ),
        "D_min_pair": compute_pairwise_min_distance(td.robot_paths),
    }

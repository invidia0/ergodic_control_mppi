"""The one summary row a deployment trial produces, and how it is computed.

Both the online recorder and the paired offline runner emit rows through here, so a UAV
trial and its ideal twin are always scored by identical code. ``SUMMARY_FIELDS`` is a
public contract: append to it, never reorder or rename.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ergodic_control_mppi.deploy.grid import clearance_along
from ergodic_control_mppi.experiments.common import append_csv
from ergodic_control_mppi.metrics.ergodicity import (
    compute_fourier_ergodic_metric,
    compute_team_ergodic_error,
)
from ergodic_control_mppi.metrics.modes import compute_mode_metrics

SUMMARY_FIELDS = [
    "run_id",
    "profile",
    "mode",
    "seed",
    "map_seed",
    "map_fill",
    "steps",
    "occupancy_mse",
    "fourier_ergodic",
    "steps_to_threshold",
    "mode_visits",
    "mode_switches",
    "mode_revisits",
    "mode_dwell_median_s",
    "mode_dwell_total_s",
    "mode_transitions",
    "mode_cycles",
    "first_all_modes_s",
    "in_mode_fraction",
    "collisions",
    "min_clearance_m",
    "guard_interventions",
    "guard_fraction",
    "guard_duration_s",
    "max_speed_mps",
    "pos_rmse_m",
    "pos_p95_m",
    "vel_rmse_mps",
    "vel_p95_mps",
    "compile_s",
    "step_p50_ms",
    "step_p95_ms",
    "step_p99_ms",
    "step_max_ms",
    "deadline_miss_fraction",
    "achieved_rate_hz",
    "wall_seconds",
    "real_time_factor",
    "run_hash",
    "config_hash",
    "git_sha",
    "seed_controller",
    "jax_version",
    "ros_distro",
    "device",
]


def _episodes(flags: np.ndarray) -> int:
    """Count maximal runs of ``True``, i.e. how many times a condition was entered."""
    flags = np.asarray(flags, dtype=bool).ravel()
    if not flags.size:
        return 0
    return int(flags[0]) + int(np.count_nonzero(np.diff(flags.astype(np.int8)) > 0))


def timing_stats(step_ms: np.ndarray, deadline_ms: float) -> dict[str, float]:
    """Summarize per-step solve times against the real-time deadline."""
    step_ms = np.asarray(step_ms, dtype=np.float64).ravel()
    if step_ms.size == 0:
        return {
            "step_p50_ms": float("nan"),
            "step_p95_ms": float("nan"),
            "step_p99_ms": float("nan"),
            "step_max_ms": float("nan"),
            "deadline_miss_fraction": float("nan"),
        }
    return {
        "step_p50_ms": float(np.percentile(step_ms, 50)),
        "step_p95_ms": float(np.percentile(step_ms, 95)),
        "step_p99_ms": float(np.percentile(step_ms, 99)),
        "step_max_ms": float(step_ms.max()),
        "deadline_miss_fraction": float((step_ms > deadline_ms).mean()),
    }


def tracking_stats(
    actual_times: np.ndarray,
    actual: np.ndarray,
    commanded_times: np.ndarray,
    commanded: np.ndarray,
) -> dict[str, float]:
    """Compare executed motion against the setpoints that were actually accepted.

    The two streams are sampled at different rates (odometry runs faster than the control
    loop) and start at different moments, so they are aligned by timestamp -- comparing
    them by index would difference two unrelated instants and report an error the size of
    the workspace. Velocity error is differenced from the same aligned pairs rather than
    taken from odometry twist, so both columns describe one consistent comparison.

    Args:
        actual_times: Odometry timestamps in seconds, shape ``(N,)``.
        actual: Executed positions, shape ``(N, 2)``.
        commanded_times: Setpoint timestamps in seconds, shape ``(M,)``.
        commanded: Accepted setpoint positions, shape ``(M, 2)``.
    """
    actual_times = np.asarray(actual_times, dtype=np.float64).ravel()
    commanded_times = np.asarray(commanded_times, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).reshape(-1, 2)
    commanded = np.asarray(commanded, dtype=np.float64).reshape(-1, 2)
    blank = dict.fromkeys(("pos_rmse_m", "pos_p95_m", "vel_rmse_mps", "vel_p95_mps"), float("nan"))
    if actual.shape[0] < 2 or commanded.shape[0] < 2:
        return blank
    # Restrict to the window both streams cover, so no sample is compared against an
    # extrapolated setpoint.
    lower = max(actual_times[0], commanded_times[0])
    upper = min(actual_times[-1], commanded_times[-1])
    window = (actual_times >= lower) & (actual_times <= upper)
    if window.sum() < 2:
        return blank
    times, measured = actual_times[window], actual[window]
    order = np.argsort(commanded_times)
    reference = np.column_stack(
        [np.interp(times, commanded_times[order], commanded[order, axis]) for axis in (0, 1)]
    )
    error = np.linalg.norm(measured - reference, axis=1)
    gaps = np.diff(times)
    valid = gaps > 0
    velocity_error = (
        np.linalg.norm(np.diff(measured, axis=0) - np.diff(reference, axis=0), axis=1)[valid]
        / gaps[valid]
    )
    if velocity_error.size == 0:
        velocity_error = np.zeros(1)
    return {
        "pos_rmse_m": float(np.sqrt((error**2).mean())),
        "pos_p95_m": float(np.percentile(error, 95)),
        "vel_rmse_mps": float(np.sqrt((velocity_error**2).mean())),
        "vel_p95_mps": float(np.percentile(velocity_error, 95)),
    }


def steps_to_threshold(convergence: np.ndarray, stride: int, factor: float = 1.5) -> float:
    """First step where the occupancy error stays within ``factor`` of its final value.

    Returns NaN when the run never settles, which is the censored case the campaign
    analysis already treats separately.
    """
    convergence = np.asarray(convergence, dtype=np.float64).ravel()
    if convergence.size == 0:
        return float("nan")
    threshold = convergence[-1] * factor
    settled = convergence <= threshold
    # Walk back from the end so a transient dip early on cannot claim convergence.
    index = settled.size
    while index > 0 and settled[index - 1]:
        index -= 1
    return float(index * stride) if index < settled.size else float("nan")


def compute_row(
    *,
    identity: dict[str, Any],
    positions: np.ndarray,
    target_grid: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    reachable_mask: np.ndarray,
    gmm_means: np.ndarray,
    gmm_inverses: np.ndarray,
    delta_t: float,
    occupancy: np.ndarray,
    grid_origin: tuple[float, float],
    grid_resolution: float,
    robot_radius: float,
    guard_states: np.ndarray,
    guard_period: float,
    speeds: np.ndarray,
    actual_times: np.ndarray,
    commanded_times: np.ndarray,
    commanded: np.ndarray,
    step_ms: np.ndarray,
    deadline_ms: float,
    wall_seconds: float,
    odometry_seconds: float,
    control_seconds: float,
) -> dict[str, Any]:
    """Build one complete summary row.

    Args:
        identity: Reproducibility and labelling columns (run id, seeds, hashes, versions).
        positions: Executed positions with shape ``(N, 2)``.
        target_grid: Normalized target density on the metric grid.
        x_limits: Workspace x bounds.
        y_limits: Workspace y bounds.
        reachable_mask: Boolean mask restricting the coverage metrics.
        gmm_means: Target mode centres with shape ``(M, 2)``.
        gmm_inverses: Inverse mode covariances with shape ``(M, 2, 2)``.
        delta_t: Control timestep in seconds.
        occupancy: Raw, uninflated map occupancy used for the clearance and collision test.
        grid_origin: Lower-left corner of the occupancy grid.
        grid_resolution: Occupancy cell size in metres.
        robot_radius: Physical footprint radius; clearance below it counts as a collision.
        guard_states: Guard state strings, sampled at the guard's own rate.
        guard_period: Seconds between guard samples; the guard runs faster than the
            controller, so its duration must not be integrated with ``delta_t``.
        speeds: Per-sample speed magnitudes.
        actual_times: Odometry timestamps, aligned with ``positions``.
        commanded_times: Setpoint timestamps, aligned with ``commanded``.
        commanded: Accepted setpoint positions with shape ``(M, 2)``.
        step_ms: Per-step solve times in milliseconds.
        deadline_ms: Real-time budget per step.
        wall_seconds: Wall-clock duration of the run, including compilation.
        odometry_seconds: Simulated duration covered by the odometry.
        control_seconds: Seconds between the first and last control step, which is what
            the achieved control rate is measured over.

    Returns:
        A mapping keyed by ``SUMMARY_FIELDS``.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    paths = positions[:, None, :]
    guard_states = np.asarray(guard_states)
    engaged = guard_states != "pass" if guard_states.size else np.zeros(0, dtype=bool)
    clearance = clearance_along(occupancy, grid_origin, grid_resolution, positions)
    # A collision is an episode, not a sample: count entries into the footprint so a long
    # scrape reads as one event, the same convention used for guard interventions.
    in_contact = clearance < robot_radius

    row: dict[str, Any] = dict.fromkeys(SUMMARY_FIELDS, "")
    row.update(identity)
    row.update(
        {
            "occupancy_mse": compute_team_ergodic_error(
                paths, target_grid, x_limits, y_limits, reachable_mask=reachable_mask
            ),
            "fourier_ergodic": compute_fourier_ergodic_metric(
                paths, target_grid, x_limits, y_limits, reachable_mask=reachable_mask
            ),
            "collisions": _episodes(in_contact),
            "min_clearance_m": float(clearance.min()) if clearance.size else float("nan"),
            "guard_interventions": _episodes(engaged),
            "guard_fraction": float(engaged.mean()) if engaged.size else 0.0,
            "guard_duration_s": float(engaged.sum() * guard_period),
            "max_speed_mps": float(np.max(speeds)) if np.size(speeds) else float("nan"),
            "wall_seconds": float(wall_seconds),
            # Control steps over the span they were actually issued in -- not odometry
            # samples (a different, much faster rate) and not wall time (which includes
            # compilation, before any step is taken).
            "achieved_rate_hz": float(np.size(step_ms) / control_seconds)
            if control_seconds > 0
            else float("nan"),
            "real_time_factor": float(odometry_seconds / wall_seconds)
            if wall_seconds > 0
            else float("nan"),
        }
    )
    # Mode metrics are timed by the spacing of *these* samples. Online, positions come
    # from odometry, which runs much faster than the control loop, so scaling them by the
    # control period would inflate every dwell and let a fleeting pass qualify as a visit.
    sample_period = (
        odometry_seconds / (positions.shape[0] - 1) if positions.shape[0] > 1 else delta_t
    )
    if not np.isfinite(sample_period) or sample_period <= 0:
        sample_period = delta_t
    row.update(compute_mode_metrics(positions, gmm_means, gmm_inverses, sample_period))
    row.update(tracking_stats(actual_times, positions, commanded_times, commanded))
    row.update(timing_stats(step_ms, deadline_ms))
    return row


def append_summary(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to the deployment summary CSV under the frozen schema."""
    unknown = set(row) - set(SUMMARY_FIELDS)
    if unknown:
        raise ValueError(f"unknown summary fields: {sorted(unknown)}")
    append_csv(path, {field: row.get(field, "") for field in SUMMARY_FIELDS}, SUMMARY_FIELDS)

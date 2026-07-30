"""Read the campaign archive and derive summaries. Never touches the GPU.

Everything downstream of the runs goes through here: the loaders that hide the
archive layout, the seed aggregates, the paired per-seed comparisons, and the
transient metric. Because the runner stores raw paths, any metric invented later
is a change to this file alone -- no re-run.

    python -m ergodic_control_mppi.experiments.analyze --stage components
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ergodic_control_mppi.experiments.common import append_csv

METRICS = ("occupancy_mse", "fourier_ergodic", "ms_per_step")

SUMMARY_FIELDS = [
    "stage", "arm", "density", "axes", "n_seeds",
    "occupancy_mse_median", "occupancy_mse_iqr",
    "occupancy_mse_delta_pct", "occupancy_mse_wins", "occupancy_mse_ci_lo",
    "occupancy_mse_ci_hi",
    "fourier_ergodic_median", "fourier_ergodic_iqr", "fourier_ergodic_delta_pct",
    "ms_per_step_median",
    "steps_to_threshold_median", "steps_to_threshold_censored", "threshold",
]

REFERENCE_ARMS = ("default", "full")


# --- archive access --------------------------------------------------------


def load_index(campaign_dir: str | Path, stage: str) -> list[dict[str, str]]:
    """Return the scalar CSV rows for one stage."""
    path = Path(campaign_dir) / f"{stage}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no index for stage '{stage}': {path}")
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_run(campaign_dir: str | Path, stage: str, cell_id: str) -> dict[str, np.ndarray]:
    """Load one run's archived arrays (paths, series, grids, obstacles)."""
    path = Path(campaign_dir) / "runs" / stage / f"{cell_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no archived run: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_series(
    campaign_dir: str | Path, stage: str, cell_id: str, metric: str = "occupancy"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(steps, values)`` for one run's stored convergence series."""
    run = load_run(campaign_dir, stage, cell_id)
    values = run[f"convergence_{metric}"]
    stride = int(run["series_stride"])
    total = int(run["paths"].shape[0])
    # The series is anchored on the final step (see the metrics module), so
    # reconstruct its step indices the same way.
    steps = np.arange(total - 1, -1, -stride, dtype=np.int64)[::-1]
    return steps[-len(values):], values


# --- statistics ------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray, statistic=np.median, resamples: int = 5000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI. Small n, no scipy -- so no p-values either."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    stats = statistic(draws, axis=1)
    tail = (1.0 - level) / 2.0
    return (float(np.quantile(stats, tail)), float(np.quantile(stats, 1.0 - tail)))


def steps_to_threshold(steps: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """First step after which the series stays at or below ``threshold``.

    Requires the condition to hold for the whole remainder of the series, so a
    transient dip does not count as convergence. ``nan`` if never reached.
    """
    below = np.asarray(values) <= threshold
    if not below.any() or not below[-1]:
        return float("nan")
    # Walk back from the end over the final contiguous run of True.
    first = len(below)
    while first > 0 and below[first - 1]:
        first -= 1
    return float(steps[first])


def _group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["stage"], row["arm"], row.get("density", ""), row.get("axes", "{}"))


def _median_iqr(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return (float(np.median(values)), float(q3 - q1))


def summarize_stage(
    campaign_dir: str | Path,
    stage: str,
    *,
    threshold_factor: float = 1.5,
    with_transient: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate one stage over seeds, paired against its reference arm.

    The threshold for ``steps_to_threshold`` is
    ``threshold_factor * median(final occupancy_mse of the reference arm)``,
    computed per density so arms on different targets are never compared against
    a shared, meaningless number.
    """
    rows = load_index(campaign_dir, stage)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    # Reference arm per density: the all-defaults cell.
    reference: dict[str, dict[str, np.ndarray]] = {}
    for key, members in grouped.items():
        _, arm, density, axes = key
        if arm in REFERENCE_ARMS and axes in ("{}", ""):
            reference[density] = {
                metric: np.asarray([float(r[metric]) for r in members], dtype=np.float64)
                for metric in METRICS
            }
            reference[density]["_seeds"] = np.asarray(
                [int(r["seed"]) for r in members], dtype=np.int64
            )

    thresholds = {
        density: threshold_factor * float(np.median(values["occupancy_mse"]))
        for density, values in reference.items()
    }

    summary: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        _, arm, density, axes = key
        seeds = np.asarray([int(r["seed"]) for r in members], dtype=np.int64)
        row: dict[str, Any] = {
            "stage": stage, "arm": arm, "density": density, "axes": axes,
            "n_seeds": len(members),
        }
        values = {
            metric: np.asarray([float(r[metric]) for r in members], dtype=np.float64)
            for metric in METRICS
        }
        for metric in METRICS:
            median, iqr = _median_iqr(values[metric])
            row[f"{metric}_median"] = median
            if f"{metric}_iqr" in SUMMARY_FIELDS:
                row[f"{metric}_iqr"] = iqr

        is_reference = arm in REFERENCE_ARMS and axes in ("{}", "")
        base = None if is_reference else reference.get(density)
        if base is not None:
            for metric in ("occupancy_mse", "fourier_ergodic"):
                base_median = float(np.median(base[metric]))
                row[f"{metric}_delta_pct"] = (
                    100.0 * (row[f"{metric}_median"] - base_median) / base_median
                    if base_median > 0 else float("nan")
                )
            # Paired per-seed comparison: only seeds present in both arms.
            shared = np.intersect1d(seeds, base["_seeds"])
            if shared.size:
                mine = np.asarray(
                    [values["occupancy_mse"][list(seeds).index(s)] for s in shared]
                )
                theirs = np.asarray(
                    [base["occupancy_mse"][list(base["_seeds"]).index(s)] for s in shared]
                )
                differences = mine - theirs
                row["occupancy_mse_wins"] = f"{int(np.sum(differences < 0))}/{shared.size}"
                low, high = bootstrap_ci(differences)
                row["occupancy_mse_ci_lo"] = low
                row["occupancy_mse_ci_hi"] = high

        threshold = thresholds.get(density)
        if with_transient and threshold is not None:
            reached = []
            censored = 0
            for member in members:
                steps, series = load_series(campaign_dir, stage, member["cell_id"])
                value = steps_to_threshold(steps, series, threshold)
                if np.isnan(value):
                    censored += 1
                else:
                    reached.append(value)
            row["steps_to_threshold_median"] = (
                float(np.median(reached)) if reached else float("nan")
            )
            row["steps_to_threshold_censored"] = censored
            row["threshold"] = threshold

        summary.append(row)
    return summary


def write_summary(
    campaign_dir: str | Path,
    stage: str,
    *,
    threshold_factor: float = 1.5,
    with_transient: bool = True,
) -> Path:
    """Write ``<stage>_summary.csv`` and return its path."""
    rows = summarize_stage(
        campaign_dir, stage,
        threshold_factor=threshold_factor, with_transient=with_transient,
    )
    path = Path(campaign_dir) / f"{stage}_summary.csv"
    if path.exists():
        path.unlink()
    for row in rows:
        append_csv(path, {key: row.get(key, "") for key in SUMMARY_FIELDS}, SUMMARY_FIELDS)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", default="results/campaign")
    parser.add_argument("--stage", action="append", dest="stages", required=True)
    parser.add_argument("--threshold-factor", type=float, default=1.5,
                        help="steps_to_threshold uses this x the reference arm's median error")
    parser.add_argument("--no-transient", action="store_true",
                        help="skip steps_to_threshold (avoids reading every .npz)")
    args = parser.parse_args()
    for stage in args.stages:
        path = write_summary(
            args.campaign_dir, stage,
            threshold_factor=args.threshold_factor,
            with_transient=not args.no_transient,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

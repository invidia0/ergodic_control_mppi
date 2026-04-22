from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments.literature_config import load_literature_comparison_config
from experiments.literature_methods import (
    make_no_obstacle_scenario,
    run_literature_method,
    sample_initial_states,
)
from experiments.scenarios import load_yaml_scenario
from experiments.trial_types import TrialData
from metrics.aggregate import compute_all_metrics
from metrics.ergodicity import compute_cumulative_team_ergodic_error


REQUIRED_METRICS = [
    "team_ergodic_error",
    "pairwise_overlap",
    "safety_metric",
    "redundancy_metric",
    "R_pair",
    "D_min_pair",
]



def _append_row(path: Path, row: dict[str, float | int | str], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)



def _summary_rows(
    rows: list[dict[str, float | int | str]],
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["scenario_name"]), str(row["method_name"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, float | int | str]] = []
    metrics = ["runtime_ms"] + REQUIRED_METRICS
    for (scenario_name, method_name), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        summary: dict[str, float | int | str] = {
            "scenario_name": scenario_name,
            "method_name": method_name,
            "num_seeds": int(len(group)),
            "seed_list": ",".join(str(s) for s in seeds),
            "team_size": int(group[0]["team_size"]),
            "steps": int(group[0]["steps"]),
        }
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals))
        out.append(summary)
    return out



def run_literature_comparison(
    config_path: str = "configs/sweeps/literature_comparison.yaml",
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    cfg = load_literature_comparison_config(config_path)
    base_scenario = load_yaml_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )

    scenarios = [
        make_no_obstacle_scenario(
            base_scenario,
            spec,
            team_size=cfg.team_size,
            steps=cfg.steps,
            grid_shape=tuple(base_scenario.target_density_grid.shape),
        )
        for spec in cfg.scenarios
    ]

    out_csv = Path(cfg.output_csv_path)
    summary_csv = Path(cfg.summary_csv_path)
    convergence_csv = Path(cfg.convergence_csv_path)
    for p in (out_csv, summary_csv, convergence_csv):
        if p.exists():
            p.unlink()

    row_fields = [
        "scenario_name",
        "method_name",
        "seed",
        "team_size",
        "steps",
        "runtime_ms",
    ] + REQUIRED_METRICS
    conv_fields = [
        "scenario_name",
        "method_name",
        "seed",
        "step",
        "team_ergodic_error",
    ]

    all_rows: list[dict[str, float | int | str]] = []
    all_conv_rows: list[dict[str, float | int | str]] = []

    for scenario in scenarios:
        for seed in cfg.seeds:
            x0_all = sample_initial_states(scenario.params, cfg.team_size, seed)
            for method_name in cfg.methods:
                print(f"Running scenario='{scenario.name}' method='{method_name}' seed={seed}...")
                t0 = time.perf_counter()
                paths = run_literature_method(
                    method_name,
                    scenario,
                    x0_all,
                    steps=cfg.steps,
                    seed=seed,
                    cfg=cfg,
                )
                runtime_ms = (time.perf_counter() - t0) * 1000.0

                trial_data = TrialData(
                    robot_paths=paths,
                    target_density_grid=scenario.target_density_grid,
                    map_x_limits=scenario.map_x_limits,
                    map_y_limits=scenario.map_y_limits,
                    obstacle_map=scenario.obstacle_map,
                    safety_radius=scenario.safety_radius,
                    metadata={
                        "scenario_name": scenario.name,
                        "method_name": method_name,
                        "seed": int(seed),
                        "team_size": int(cfg.team_size),
                        "steps": int(cfg.steps),
                    },
                )
                metrics = compute_all_metrics(trial_data, pairwise_d_thresh=cfg.d_thresh)
                row: dict[str, float | int | str] = {
                    "scenario_name": scenario.name,
                    "method_name": method_name,
                    "seed": int(seed),
                    "team_size": int(cfg.team_size),
                    "steps": int(cfg.steps),
                    "runtime_ms": float(runtime_ms),
                }
                row.update(metrics)
                all_rows.append(row)
                _append_row(out_csv, row, row_fields)

                cumulative = compute_cumulative_team_ergodic_error(
                    paths,
                    scenario.target_density_grid,
                    scenario.map_x_limits,
                    scenario.map_y_limits,
                )
                for t_idx, val in enumerate(np.asarray(cumulative, dtype=np.float64), start=1):
                    conv_row: dict[str, float | int | str] = {
                        "scenario_name": scenario.name,
                        "method_name": method_name,
                        "seed": int(seed),
                        "step": int(t_idx),
                        "team_ergodic_error": float(val),
                    }
                    all_conv_rows.append(conv_row)
                    _append_row(convergence_csv, conv_row, conv_fields)

    summary_rows = _summary_rows(all_rows, seeds=cfg.seeds)
    if len(summary_rows) > 0:
        summary_fields = list(summary_rows[0].keys())
        for row in summary_rows:
            _append_row(summary_csv, row, summary_fields)

    return all_rows, summary_rows, all_conv_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run no-obstacle literature method comparison sweeps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweeps/literature_comparison.yaml",
        help="Path to literature comparison YAML config.",
    )
    args = parser.parse_args()
    run_literature_comparison(args.config)

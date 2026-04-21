from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.run_single_trial import run_single_trial
from experiments.scenarios import load_yaml_scenario
from experiments.sensitivity_config import load_sensitivity_sweep_config


def _theta_from_rotation(A: jnp.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(A[1, 0]), float(A[0, 0]))))


def _baseline_from_scenario(scenario) -> dict[str, float | int]:
    return {
        "alpha_cross": float(scenario.params.stein.alpha_cross),
        "ell_x": float(scenario.params.stein.ell_x),
        "weight_stein": float(scenario.params.stein.weight_stein),
        "theta": _theta_from_rotation(scenario.params.stein.A),
        "history_window": int(scenario.params.history_len),
        "horizon": int(scenario.params.T),
    }


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected existing sensitivity CSV at '{csv_path}'")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"Sensitivity CSV '{csv_path}' exists but has no rows")
    return rows


def _existing_jobs(rows: list[dict[str, str]]) -> set[tuple[str, float, int]]:
    out: set[tuple[str, float, int]] = set()
    for row in rows:
        out.add((str(row["sweep_name"]), float(row["sweep_value"]), int(row["seed"])))
    return out


def _append_row(path: Path, row: dict[str, float | int | str], fieldnames: list[str]) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _rebuild_summary(
    all_rows: list[dict[str, str]],
    summary_csv_path: Path,
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in all_rows:
        key = (str(row["sweep_name"]), float(row["sweep_value"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    sorted_keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1]))
    for sweep_name, sweep_value in sorted_keys:
        group = grouped[(sweep_name, sweep_value)]
        vals = np.asarray([float(r["team_ergodic_error"]) for r in group], dtype=np.float64)
        seeds = sorted(int(r["seed"]) for r in group)
        summary_rows.append(
            {
                "sweep_name": sweep_name,
                "sweep_value": float(sweep_value),
                "num_seeds": int(len(group)),
                "seed_list": ",".join(str(s) for s in seeds),
                "team_ergodic_error_mean": float(np.mean(vals)),
                "team_ergodic_error_std": float(np.std(vals)),
            }
        )

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(summary_rows[0].keys()) if len(summary_rows) > 0 else [
            "sweep_name",
            "sweep_value",
            "num_seeds",
            "seed_list",
            "team_ergodic_error_mean",
            "team_ergodic_error_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_rows


def run_sensitivity_incremental(
    config_path: str = "configs/sweeps/open_multimodal_sensitivity.yaml",
    theta_value: float = 80.0,
    alpha_cross_value: float = 100.0,
    dry_run: bool = False,
) -> tuple[int, int]:
    cfg = load_sensitivity_sweep_config(config_path)
    scenario = load_yaml_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    baseline_controller = _baseline_from_scenario(scenario)
    print(
        "Using baseline controller from scenario_config_path "
        f"'{cfg.scenario_config_path}': {baseline_controller}"
    )

    output_csv = Path(cfg.output_csv_path)
    summary_csv = Path(cfg.summary_csv_path)
    rows = _load_rows(output_csv)
    fieldnames = list(rows[0].keys())
    existing = _existing_jobs(rows)

    additions = [
        ("theta", "theta", float(theta_value)),
        ("alpha_cross", "alpha_cross", float(alpha_cross_value)),
    ]

    jobs: list[tuple[str, str, float, int]] = []
    for sweep_name, param_key, sweep_value in additions:
        for seed in cfg.seeds:
            job_key = (sweep_name, float(sweep_value), int(seed))
            if job_key in existing:
                continue
            jobs.append((sweep_name, param_key, float(sweep_value), int(seed)))

    print(f"Found {len(jobs)} missing jobs to run.")
    if dry_run:
        for sweep_name, _, sweep_value, seed in jobs:
            print(f"[dry-run] sweep='{sweep_name}' value={sweep_value} seed={seed}")
        return 0, len(jobs)

    seed_list_str = ",".join(str(s) for s in cfg.seeds)
    appended = 0
    for sweep_name, param_key, sweep_value, seed in jobs:
        controller_config = dict(baseline_controller)
        controller_config[param_key] = float(sweep_value)
        print(f"Running incremental sweep='{sweep_name}' value={sweep_value} seed={seed}...")
        row = run_single_trial(
            scenario=scenario,
            controller_config=controller_config,
            seed=seed,
            team_size=cfg.team_size,
            steps=cfg.steps,
            pairwise_d_thresh=cfg.d_thresh,
        )
        row["sweep_name"] = sweep_name
        row["sweep_param"] = param_key
        row["sweep_value"] = float(sweep_value)
        row["num_seeds"] = int(len(cfg.seeds))
        row["seed_list"] = seed_list_str
        _append_row(output_csv, row, fieldnames)
        appended += 1

    updated_rows = _load_rows(output_csv)
    _rebuild_summary(updated_rows, summary_csv)
    print(f"Appended {appended} rows and rebuilt summary CSV.")
    return appended, len(jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Append targeted sensitivity sweep points. Baseline controller values are loaded "
            "from scenario_config_path in the sensitivity config (same logic as run_sensitivity_sweeps)."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweeps/open_multimodal_sensitivity.yaml",
        help="Path to sensitivity sweep YAML config.",
    )
    parser.add_argument(
        "--theta-value",
        type=float,
        default=80.0,
        help="Theta sweep value to append.",
    )
    parser.add_argument(
        "--alpha-cross-value",
        type=float,
        default=100.0,
        help="Alpha-cross sweep value to append.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview missing jobs without running trials.",
    )
    args = parser.parse_args()
    run_sensitivity_incremental(
        config_path=args.config,
        theta_value=args.theta_value,
        alpha_cross_value=args.alpha_cross_value,
        dry_run=args.dry_run,
    )

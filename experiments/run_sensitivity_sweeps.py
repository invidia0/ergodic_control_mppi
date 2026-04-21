from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.run_single_trial import run_single_trial
from experiments.scenarios import load_yaml_scenario
from experiments.sensitivity_config import load_sensitivity_sweep_config


def _append_row(path: Path, row: dict[str, float | int | str], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def _build_sweep_specs(cfg) -> list[tuple[str, str, list[float | int]]]:
    return [
        ("theta", "theta", [float(v) for v in cfg.theta_values]),
        ("alpha_cross", "alpha_cross", [float(v) for v in cfg.alpha_cross_values]),
        ("horizon", "horizon", [int(v) for v in cfg.horizon_values]),
        ("weight_stein", "weight_stein", [float(v) for v in cfg.weight_stein_values]),
    ]


def _summarize_rows(
    rows: list[dict[str, float | int | str]],
    sweep_specs: list[tuple[str, str, list[float | int]]],
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for sweep_name, _, sweep_values in sweep_specs:
        for sweep_value in sweep_values:
            group = [
                r
                for r in rows
                if str(r["sweep_name"]) == sweep_name and float(r["sweep_value"]) == float(sweep_value)
            ]
            if len(group) == 0:
                continue
            erg_vals = np.asarray([float(r["team_ergodic_error"]) for r in group], dtype=np.float64)
            summary: dict[str, float | int | str] = {
                "sweep_name": sweep_name,
                "sweep_value": float(sweep_value),
                "num_seeds": int(len(group)),
                "seed_list": str(group[0]["seed_list"]),
                "team_ergodic_error_mean": float(np.mean(erg_vals)),
                "team_ergodic_error_std": float(np.std(erg_vals)),
            }
            out.append(summary)
    return out


def run_sensitivity_sweeps(
    sensitivity_config_path: str = "configs/sweeps/open_multimodal_sensitivity.yaml",
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], list[str]]:
    cfg = load_sensitivity_sweep_config(sensitivity_config_path)
    scenario = load_yaml_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    baseline_controller = _baseline_from_scenario(scenario)
    sweep_specs = _build_sweep_specs(cfg)

    out_csv = Path(cfg.output_csv_path)
    summary_csv = Path(cfg.summary_csv_path)
    if out_csv.exists():
        out_csv.unlink()
    if summary_csv.exists():
        summary_csv.unlink()

    all_rows: list[dict[str, float | int | str]] = []
    fieldnames: list[str] | None = None
    for sweep_name, param_key, sweep_values in sweep_specs:
        for sweep_value in sweep_values:
            for seed in cfg.seeds:
                controller_config = dict(baseline_controller)
                controller_config[param_key] = int(sweep_value) if param_key == "horizon" else float(sweep_value)
                print(
                    f"Running sweep='{sweep_name}' value={sweep_value} seed={seed}..."
                )
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
                row["seed_list"] = ",".join(str(s) for s in cfg.seeds)
                all_rows.append(row)
                if fieldnames is None:
                    primary = ["sweep_name", "sweep_param", "sweep_value", "num_seeds", "seed_list"]
                    fieldnames = primary + [k for k in row.keys() if k not in set(primary)]
                _append_row(out_csv, row, fieldnames)

    summary_rows = _summarize_rows(all_rows, sweep_specs=sweep_specs)
    if len(summary_rows) > 0:
        summary_fields = list(summary_rows[0].keys())
        for summary_row in summary_rows:
            _append_row(summary_csv, summary_row, summary_fields)

    # Plotting is handled in plots/plot_sensitivity.py to keep this runner data-only.
    return all_rows, summary_rows, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BO-baseline one-factor sensitivity sweeps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweeps/open_multimodal_sensitivity.yaml",
        help="Path to sensitivity sweep YAML config.",
    )
    args = parser.parse_args()
    run_sensitivity_sweeps(args.config)

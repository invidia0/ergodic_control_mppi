from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.ablation_config import load_ablation_config
from experiments.bo_config import load_bo_config
from experiments.run_single_trial import run_single_trial
from experiments.scenarios import load_yaml_scenario

ABLATION_ORDER = ["Full", "No Curl", "No Cross", "Weak Stein", "Reduced Horizon"]


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


def _get_reduced_horizon(bo_config_path: str, default_horizon: int = 100) -> int:
    try:
        bo_cfg = load_bo_config(bo_config_path)
        if len(bo_cfg.horizon_values) == 0:
            return int(default_horizon)
        return int(min(bo_cfg.horizon_values))
    except Exception:
        return int(default_horizon)


def _build_ablations(
    baseline_controller: dict[str, float | int],
    reduced_horizon: int,
) -> list[tuple[str, dict[str, float | int]]]:
    full = dict(baseline_controller)
    no_curl = dict(full)
    no_curl["theta"] = 0.0
    no_cross = dict(full)
    no_cross["alpha_cross"] = 0.0
    weak_stein = dict(full)
    weak_stein["weight_stein"] = 0.1 * float(full["weight_stein"])
    reduced_h = dict(full)
    reduced_h["horizon"] = int(reduced_horizon)
    return [
        ("Full", full),
        ("No Curl", no_curl),
        ("No Cross", no_cross),
        ("Weak Stein", weak_stein),
        ("Reduced Horizon", reduced_h),
    ]


def _summarize_rows(
    rows: list[dict[str, float | int | str]],
    metric_cols: list[str],
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    present = {str(r["ablation_name"]) for r in rows}
    ablations = [name for name in ABLATION_ORDER if name in present]
    ablations.extend(sorted(name for name in present if name not in set(ablations)))
    for ablation_name in ablations:
        group = [r for r in rows if str(r["ablation_name"]) == ablation_name]
        summary: dict[str, float | int | str] = {
            "ablation_name": ablation_name,
            "num_seeds": int(len(group)),
            "seed_list": str(group[0]["seed_list"]) if group else "",
        }
        for col in metric_cols:
            vals = np.asarray([float(r[col]) for r in group], dtype=np.float64)
            summary[f"{col}_mean"] = float(np.mean(vals))
            summary[f"{col}_std"] = float(np.std(vals))
        out.append(summary)
    return out


def run_ablations(
    ablation_config_path: str = "configs/sweeps/open_multimodal_ablations.yaml",
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    cfg = load_ablation_config(ablation_config_path)
    scenario = load_yaml_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    baseline_controller = _baseline_from_scenario(scenario)
    reduced_horizon = _get_reduced_horizon(cfg.bo_config_path, default_horizon=100)
    ablations = _build_ablations(baseline_controller, reduced_horizon=reduced_horizon)

    out_csv = Path(cfg.output_csv_path)
    summary_csv = Path(cfg.summary_csv_path)
    if out_csv.exists():
        out_csv.unlink()
    if summary_csv.exists():
        summary_csv.unlink()
    metric_cols = [
        "team_ergodic_error",
        "redundancy_metric",
        "safety_metric",
        "R_pair",
        "D_min_pair",
    ]

    all_rows: list[dict[str, float | int | str]] = []
    fieldnames: list[str] | None = None
    for ablation_name, controller_config in ablations:
        for seed in cfg.seeds:
            print(f"Running ablation='{ablation_name}' seed={seed}...")
            row = run_single_trial(
                scenario=scenario,
                controller_config=controller_config,
                seed=seed,
                team_size=cfg.team_size,
                steps=cfg.steps,
                pairwise_d_thresh=cfg.d_thresh,
            )
            row["ablation_name"] = ablation_name
            row["num_seeds"] = int(len(cfg.seeds))
            row["seed_list"] = ",".join(str(s) for s in cfg.seeds)
            all_rows.append(row)
            if fieldnames is None:
                fieldnames = ["ablation_name", "num_seeds", "seed_list"] + [
                    k for k in row.keys() if k not in {"ablation_name", "num_seeds", "seed_list"}
                ]
            _append_row(out_csv, row, fieldnames)

    summary_rows = _summarize_rows(all_rows, metric_cols=metric_cols)
    if len(summary_rows) > 0:
        summary_fields = list(summary_rows[0].keys())
        for summary_row in summary_rows:
            _append_row(summary_csv, summary_row, summary_fields)

    return all_rows, summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments from BO-best controller config.")
    parser.add_argument(
        "--ablation-config",
        type=str,
        default="configs/sweeps/open_multimodal_ablations.yaml",
        help="Path to ablation YAML config.",
    )
    args = parser.parse_args()
    run_ablations(args.ablation_config)

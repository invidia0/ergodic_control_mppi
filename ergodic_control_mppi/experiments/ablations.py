"""Run the established controller ablation study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AblationConfig:
    """Validated ablation experiment configuration."""
    scenario_name: str
    scenario_config_path: str
    bo_config_path: str
    output_csv_path: str
    summary_csv_path: str
    team_size: int
    steps: int
    seeds: list[int]
    d_thresh: float


def _as_int(value, key: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"'{key}' must be an integer, got bool")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be an integer") from exc
    if min_value is not None and out < min_value:
        raise ValueError(f"'{key}' must be >= {min_value}")
    return out


def _as_float(value, key: str, *, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be a float") from exc
    if min_value is not None and out < min_value:
        raise ValueError(f"'{key}' must be >= {min_value}")
    return out


def _as_list_int(cfg: dict, key: str, *, min_value: int = 0) -> list[int]:
    value = cfg.get(key)
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"'{key}' must be a non-empty list")
    out = [_as_int(x, key, min_value=min_value) for x in value]
    if len(set(out)) != len(out):
        raise ValueError(f"'{key}' must not contain duplicates")
    return out


def load_ablation_config(path: str | Path) -> AblationConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("ablation config must be a YAML mapping")

    output_csv_path = str(cfg.get("output_csv_path", "results/dars2026/ablations/open_multimodal_ablations.csv"))
    summary_csv_default = str(Path(output_csv_path).with_name(Path(output_csv_path).stem + "_summary.csv"))

    return AblationConfig(
        scenario_name=str(cfg.get("scenario_name", "open_multimodal_ablation")),
        scenario_config_path=str(cfg.get("scenario_config_path", "configs/mppi_params.yaml")),
        bo_config_path=str(cfg.get("bo_config_path", "configs/experiments/open_multimodal_bo.yaml")),
        output_csv_path=output_csv_path,
        summary_csv_path=str(cfg.get("summary_csv_path", summary_csv_default)),
        team_size=_as_int(cfg.get("team_size", 4), "team_size", min_value=1),
        steps=_as_int(cfg.get("steps", 5000), "steps", min_value=1),
        seeds=_as_list_int(cfg, "seeds", min_value=0) if "seeds" in cfg else [0, 1, 2, 3, 4],
        d_thresh=_as_float(cfg.get("d_thresh", 1.0), "d_thresh", min_value=0.0),
    )
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.common import (
    append_csv,
    load_scenario,
    prepare_outputs,
    run_trial,
    summarize,
    variant_from_mapping,
)

ABLATION_ORDER = ["Full", "No Curl", "No Cross", "Weak Stein", "Reduced Horizon"]


def _theta_from_rotation(A: jnp.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(A[1, 0]), float(A[0, 0]))))


def _baseline_from_scenario(scenario) -> dict[str, float | int]:
    return {
        "alpha_cross": float(scenario.params.stein.repulsion_weight),
        "ell_x": float(scenario.params.stein.cross_bandwidth),
        "weight_stein": float(scenario.params.stein.flow_weight),
        "theta": _theta_from_rotation(scenario.params.stein.rotation),
        "history_window": int(scenario.params.mppi.history_length),
        "horizon": int(scenario.params.mppi.horizon),
    }


def _get_reduced_horizon(bo_config_path: str, default_horizon: int = 100) -> int:
    try:
        with Path(bo_config_path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream).get("horizon_values", [])
        if not values:
            return int(default_horizon)
        return int(min(values))
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
        summary.update(summarize(group, metric_cols))
        out.append(summary)
    return out


def run_ablations(
    ablation_config_path: str = "configs/experiments/open_multimodal_ablations.yaml",
    overwrite: bool = False,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    cfg = load_ablation_config(ablation_config_path)
    scenario = load_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    baseline_controller = _baseline_from_scenario(scenario)
    reduced_horizon = _get_reduced_horizon(cfg.bo_config_path, default_horizon=100)
    ablations = _build_ablations(baseline_controller, reduced_horizon=reduced_horizon)

    out_csv = Path(cfg.output_csv_path)
    summary_csv = Path(cfg.summary_csv_path)
    prepare_outputs((out_csv, summary_csv), overwrite)
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
            row = run_trial(
                scenario=scenario,
                variant=variant_from_mapping(controller_config),
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
            append_csv(out_csv, row, fieldnames)

    summary_rows = _summarize_rows(all_rows, metric_cols=metric_cols)
    if len(summary_rows) > 0:
        summary_fields = list(summary_rows[0].keys())
        for summary_row in summary_rows:
            append_csv(summary_csv, summary_row, summary_fields)

    return all_rows, summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments from BO-best controller config.")
    parser.add_argument(
        "--ablation-config",
        type=str,
        default="configs/experiments/open_multimodal_ablations.yaml",
        help="Path to ablation YAML config.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_ablations(args.ablation_config, overwrite=args.overwrite)

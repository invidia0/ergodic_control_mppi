"""Run one-factor sensitivity sweeps around the configured baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SensitivitySweepConfig:
    """Validated one-factor sensitivity experiment configuration."""
    scenario_name: str
    scenario_config_path: str
    output_csv_path: str
    summary_csv_path: str
    plot_output_dir: str
    team_size: int
    steps: int
    seeds: list[int]
    d_thresh: float
    theta_values: list[float]
    alpha_cross_values: list[float]
    horizon_values: list[int]
    weight_stein_values: list[float]
    use_log_y: bool


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


def _as_list_float(cfg: dict, key: str, *, min_value: float | None = None) -> list[float]:
    value = cfg.get(key)
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"'{key}' must be a non-empty list")
    out = [_as_float(x, key, min_value=min_value) for x in value]
    if len(set(out)) != len(out):
        raise ValueError(f"'{key}' must not contain duplicates")
    return out


def load_sensitivity_sweep_config(path: str | Path) -> SensitivitySweepConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sensitivity sweep config must be a YAML mapping")

    output_csv_path = str(
        cfg.get("output_csv_path", "results/dars2026/sensitivity/open_multimodal_sensitivity.csv")
    )
    summary_csv_default = str(Path(output_csv_path).with_name(Path(output_csv_path).stem + "_summary.csv"))
    plot_output_default = str(Path(output_csv_path).with_name(Path(output_csv_path).stem + "_plots"))

    return SensitivitySweepConfig(
        scenario_name=str(cfg.get("scenario_name", "open_multimodal_sensitivity")),
        scenario_config_path=str(cfg.get("scenario_config_path", "configs/mppi_params.yaml")),
        output_csv_path=output_csv_path,
        summary_csv_path=str(cfg.get("summary_csv_path", summary_csv_default)),
        plot_output_dir=str(cfg.get("plot_output_dir", plot_output_default)),
        team_size=_as_int(cfg.get("team_size", 4), "team_size", min_value=1),
        steps=_as_int(cfg.get("steps", 5000), "steps", min_value=1),
        seeds=_as_list_int(cfg, "seeds", min_value=0) if "seeds" in cfg else [0, 1, 2, 3, 4],
        d_thresh=_as_float(cfg.get("d_thresh", 1.0), "d_thresh", min_value=0.0),
        theta_values=_as_list_float(cfg, "theta_values"),
        alpha_cross_values=_as_list_float(cfg, "alpha_cross_values"),
        horizon_values=_as_list_int(cfg, "horizon_values", min_value=1),
        weight_stein_values=_as_list_float(cfg, "weight_stein_values", min_value=0.0),
        use_log_y=bool(cfg.get("use_log_y", True)),
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
            summary: dict[str, float | int | str] = {
                "sweep_name": sweep_name,
                "sweep_value": float(sweep_value),
                "num_seeds": int(len(group)),
                "seed_list": str(group[0]["seed_list"]),
            }
            summary.update(summarize(group, ["team_ergodic_error"]))
            out.append(summary)
    return out


def run_sensitivity_sweeps(
    sensitivity_config_path: str = "configs/experiments/open_multimodal_sensitivity.yaml",
    overwrite: bool = False,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], list[str]]:
    cfg = load_sensitivity_sweep_config(sensitivity_config_path)
    scenario = load_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    baseline_controller = _baseline_from_scenario(scenario)
    sweep_specs = _build_sweep_specs(cfg)

    out_csv = Path(cfg.output_csv_path)
    summary_csv = Path(cfg.summary_csv_path)
    prepare_outputs((out_csv, summary_csv), overwrite)

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
                row = run_trial(
                    scenario=scenario,
                    variant=variant_from_mapping(controller_config),
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
                append_csv(out_csv, row, fieldnames)

    summary_rows = _summarize_rows(all_rows, sweep_specs=sweep_specs)
    if len(summary_rows) > 0:
        summary_fields = list(summary_rows[0].keys())
        for summary_row in summary_rows:
            append_csv(summary_csv, summary_row, summary_fields)

    # Plotting is handled in ergodic_control_mppi.plotting.sensitivity.
    return all_rows, summary_rows, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BO-baseline one-factor sensitivity sweeps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/open_multimodal_sensitivity.yaml",
        help="Path to sensitivity sweep YAML config.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_sensitivity_sweeps(args.config, overwrite=args.overwrite)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SensitivitySweepConfig:
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ContinuousSearchSpace:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class BOConfig:
    scenario_name: str
    scenario_config_path: str
    output_csv_path: str
    study_name: str
    storage: str | None
    sampler_seed: int
    n_trials: int
    n_startup_trials: int
    n_jobs: int
    team_size: int
    steps: int
    search_seeds: list[int]
    reeval_seeds: list[int]
    reeval_top_n: int
    safety_max: float | None
    safety_penalty: float | None
    include_baseline: bool
    alpha_cross: ContinuousSearchSpace
    ell_x: ContinuousSearchSpace
    weight_stein: ContinuousSearchSpace
    theta: ContinuousSearchSpace
    history_window_values: list[int]
    horizon_values: list[int]


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("BO config must be a YAML mapping")
    return cfg


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


def _as_list_int(cfg: dict, key: str, *, min_value: int = 1) -> list[int]:
    value = cfg.get(key)
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"'{key}' must be a non-empty list")
    out = [_as_int(x, key, min_value=min_value) for x in value]
    if len(set(out)) != len(out):
        raise ValueError(f"'{key}' must not contain duplicates")
    return out


def _as_search_space(cfg: dict, key: str, *, allow_log: bool = True) -> ContinuousSearchSpace:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping")
    low = _as_float(value.get("low"), f"{key}.low")
    high = _as_float(value.get("high"), f"{key}.high")
    if high <= low:
        raise ValueError(f"'{key}.high' must be > '{key}.low'")
    log = bool(value.get("log", False))
    if log and not allow_log:
        raise ValueError(f"'{key}.log' is not allowed for this parameter")
    if log and low <= 0.0:
        raise ValueError(f"'{key}.low' must be > 0 when '{key}.log' is true")
    return ContinuousSearchSpace(low=low, high=high, log=log)


def load_bo_config(path: str | Path) -> BOConfig:
    cfg = _load_yaml(path)

    search_seeds = _as_list_int(cfg, "search_seeds", min_value=0)
    reeval_seeds = _as_list_int(cfg, "reeval_seeds", min_value=0)

    safety_max_raw = cfg.get("safety_max")
    safety_penalty_raw = cfg.get("safety_penalty")
    safety_max = None if safety_max_raw is None else _as_float(safety_max_raw, "safety_max")
    safety_penalty = None if safety_penalty_raw is None else _as_float(
        safety_penalty_raw, "safety_penalty", min_value=0.0
    )

    return BOConfig(
        scenario_name=str(cfg.get("scenario_name", "open_multimodal_bo")),
        scenario_config_path=str(cfg.get("scenario_config_path", "configs/mppi_params.yaml")),
        output_csv_path=str(cfg.get("output_csv_path", "results/dars2026/bo/open_multimodal_bo.csv")),
        study_name=str(cfg.get("study_name", "open_multimodal_bo")),
        storage=None if cfg.get("storage") is None else str(cfg.get("storage")),
        sampler_seed=_as_int(cfg.get("sampler_seed", 0), "sampler_seed", min_value=0),
        n_trials=_as_int(cfg.get("n_trials", 80), "n_trials", min_value=1),
        n_startup_trials=_as_int(cfg.get("n_startup_trials", 20), "n_startup_trials", min_value=1),
        n_jobs=_as_int(cfg.get("n_jobs", 1), "n_jobs", min_value=1),
        team_size=_as_int(cfg.get("team_size", 4), "team_size", min_value=1),
        steps=_as_int(cfg.get("steps", 5000), "steps", min_value=1),
        search_seeds=search_seeds,
        reeval_seeds=reeval_seeds,
        reeval_top_n=_as_int(cfg.get("reeval_top_n", 10), "reeval_top_n", min_value=1),
        safety_max=safety_max,
        safety_penalty=safety_penalty,
        include_baseline=bool(cfg.get("include_baseline", True)),
        alpha_cross=_as_search_space(cfg, "alpha_cross"),
        ell_x=_as_search_space(cfg, "ell_x"),
        weight_stein=_as_search_space(cfg, "weight_stein"),
        theta=_as_search_space(cfg, "theta", allow_log=False),
        history_window_values=_as_list_int(cfg, "history_window_values"),
        horizon_values=_as_list_int(cfg, "horizon_values"),
    )

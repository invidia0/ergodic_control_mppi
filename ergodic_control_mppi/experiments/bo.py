"""Optuna Bayesian optimization over typed controller variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ContinuousSearchSpace:
    """Continuous Optuna interval and logarithmic-sampling flag."""
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class BOConfig:
    """Validated Bayesian-optimization experiment configuration."""
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
import argparse
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.common import append_csv, load_scenario, run_trial, variant_from_mapping

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


BO_CSV_FIELDS = [
    "study_name", "phase", "trial_number", "objective", "team_ergodic_error",
    "pairwise_overlap", "safety_metric", "redundancy_metric", "runtime_ms",
    "num_seeds", "seed_list", "team_size", "steps", "alpha_cross", "ell_x",
    "weight_stein", "theta", "history_window", "horizon", "safety_penalized",
    "is_baseline", "error",
]


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


def _ensure_storage_parent(storage: str | None) -> None:
    if storage is None:
        return
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        db_path = Path(storage[len(prefix):])
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _sample_controller_config(trial: optuna.trial.Trial, cfg: BOConfig) -> dict[str, float | int]:
    return {
        "alpha_cross": float(
            trial.suggest_float(
                "alpha_cross",
                cfg.alpha_cross.low,
                cfg.alpha_cross.high,
                log=cfg.alpha_cross.log,
            )
        ),
        "ell_x": float(
            trial.suggest_float(
                "ell_x",
                cfg.ell_x.low,
                cfg.ell_x.high,
                log=cfg.ell_x.log,
            )
        ),
        "weight_stein": float(
            trial.suggest_float(
                "weight_stein",
                cfg.weight_stein.low,
                cfg.weight_stein.high,
                log=cfg.weight_stein.log,
            )
        ),
        "theta": float(
            trial.suggest_float(
                "theta",
                cfg.theta.low,
                cfg.theta.high,
                log=cfg.theta.log,
            )
        ),
        "history_window": int(trial.suggest_categorical("history_window", cfg.history_window_values)),
        "horizon": int(trial.suggest_categorical("horizon", cfg.horizon_values)),
    }


def _summarize_seed_rows(seed_rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    metrics = [
        "team_ergodic_error",
        "pairwise_overlap",
        "safety_metric",
        "redundancy_metric",
        "runtime_ms",
    ]
    return {k: float(np.mean([float(row[k]) for row in seed_rows])) for k in metrics}


def _evaluate_config(
    *,
    scenario,
    controller_config: dict[str, float | int],
    seeds: list[int],
    team_size: int,
    steps: int,
) -> tuple[dict[str, float], str]:
    seed_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        row = run_trial(
            scenario=scenario,
            variant=variant_from_mapping(controller_config),
            seed=seed,
            team_size=team_size,
            steps=steps,
        )
        seed_rows.append(row)
    return _summarize_seed_rows(seed_rows), ",".join(str(s) for s in seeds)


def _objective_from_metrics(
    metrics: dict[str, float],
    safety_max: float | None,
    safety_penalty: float | None,
) -> tuple[float, bool]:
    objective = float(metrics["team_ergodic_error"])
    if safety_max is None or safety_penalty is None:
        return objective, False
    if float(metrics["safety_metric"]) <= safety_max:
        return objective, False
    return objective + safety_penalty, True


def _as_csv_row(
    *,
    cfg: BOConfig,
    phase: str,
    trial_number: int,
    controller_config: dict[str, float | int],
    metrics: dict[str, float],
    objective: float,
    seeds: list[int],
    safety_penalized: bool,
    is_baseline: bool,
) -> dict[str, float | int | str]:
    return {
        "study_name": cfg.study_name,
        "phase": phase,
        "trial_number": int(trial_number),
        "objective": float(objective),
        "team_ergodic_error": float(metrics["team_ergodic_error"]),
        "pairwise_overlap": float(metrics["pairwise_overlap"]),
        "safety_metric": float(metrics["safety_metric"]),
        "redundancy_metric": float(metrics["redundancy_metric"]),
        "runtime_ms": float(metrics["runtime_ms"]),
        "num_seeds": int(len(seeds)),
        "seed_list": ",".join(str(s) for s in seeds),
        "team_size": int(cfg.team_size),
        "steps": int(cfg.steps),
        "alpha_cross": float(controller_config["alpha_cross"]),
        "ell_x": float(controller_config["ell_x"]),
        "weight_stein": float(controller_config["weight_stein"]),
        "theta": float(controller_config["theta"]),
        "history_window": int(controller_config["history_window"]),
        "horizon": int(controller_config["horizon"]),
        "safety_penalized": int(safety_penalized),
        "is_baseline": int(is_baseline),
    }


def _config_key(cfg: dict[str, float | int]) -> tuple[float | int, ...]:
    return (
        float(cfg["alpha_cross"]),
        float(cfg["ell_x"]),
        float(cfg["weight_stein"]),
        float(cfg["theta"]),
        int(cfg["history_window"]),
        int(cfg["horizon"]),
    )


def _top_trial_configs(study: optuna.Study, top_n: int) -> list[tuple[int, dict[str, float | int]]]:
    seen: set[tuple[float | int, ...]] = set()
    out: list[tuple[int, dict[str, float | int]]] = []
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    trials.sort(key=lambda t: float(t.value))
    for t in trials:
        params = dict(t.params)
        for key in ("alpha_cross", "ell_x", "weight_stein", "theta"):
            params[key] = float(params[key])
        params["history_window"] = int(params["history_window"])
        params["horizon"] = int(params["horizon"])
        k = _config_key(params)
        if k in seen:
            continue
        seen.add(k)
        out.append((int(t.number), params))
        if len(out) >= top_n:
            break
    return out


def run_bo(
    bo_config_path: str = "configs/experiments/open_multimodal_bo.yaml",
) -> dict[str, float | int | str]:
    if optuna is None:
        raise ImportError("Optuna is required for BO; install the 'experiments' extra")
    cfg = load_bo_config(bo_config_path)
    scenario = load_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    csv_path = Path(cfg.output_csv_path)
    fieldnames = BO_CSV_FIELDS

    _ensure_storage_parent(cfg.storage)
    sampler = optuna.samplers.TPESampler(
        seed=cfg.sampler_seed,
        n_startup_trials=cfg.n_startup_trials,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=cfg.study_name,
        storage=cfg.storage,
        load_if_exists=True,
    )

    baseline = _baseline_from_scenario(scenario) # warm-start the BO with the baseline controller config
    if cfg.include_baseline:
        baseline_is_valid = (
            baseline["history_window"] in set(cfg.history_window_values)
            and baseline["horizon"] in set(cfg.horizon_values)
            and cfg.alpha_cross.low <= float(baseline["alpha_cross"]) <= cfg.alpha_cross.high
            and cfg.ell_x.low <= float(baseline["ell_x"]) <= cfg.ell_x.high
            and cfg.weight_stein.low <= float(baseline["weight_stein"]) <= cfg.weight_stein.high
            and cfg.theta.low <= float(baseline["theta"]) <= cfg.theta.high
        )
        if baseline_is_valid:
            study.enqueue_trial(baseline)

    def objective(trial: optuna.trial.Trial) -> float:
        controller_config = _sample_controller_config(trial, cfg)
        is_baseline = _config_key(controller_config) == _config_key(baseline)
        try:
            metrics, _ = _evaluate_config(
                scenario=scenario,
                controller_config=controller_config,
                seeds=cfg.search_seeds,
                team_size=cfg.team_size,
                steps=cfg.steps,
            )
            objective_value, safety_penalized = _objective_from_metrics(
                metrics,
                safety_max=cfg.safety_max,
                safety_penalty=cfg.safety_penalty,
            )
            row = _as_csv_row(
                cfg=cfg,
                phase="search",
                trial_number=int(trial.number),
                controller_config=controller_config,
                metrics=metrics,
                objective=objective_value,
                seeds=cfg.search_seeds,
                safety_penalized=safety_penalized,
                is_baseline=is_baseline,
            )
            append_csv(csv_path, row, fieldnames)
            return float(objective_value)
        except Exception as exc:  # pragma: no cover - runtime guard
            fail_metrics = {
                "team_ergodic_error": float("inf"),
                "pairwise_overlap": float("nan"),
                "safety_metric": float("nan"),
                "redundancy_metric": float("nan"),
                "runtime_ms": float("nan"),
            }
            row = _as_csv_row(
                cfg=cfg,
                phase="search_fail",
                trial_number=int(trial.number),
                controller_config=controller_config,
                metrics=fail_metrics,
                objective=float("inf"),
                seeds=cfg.search_seeds,
                safety_penalized=False,
                is_baseline=is_baseline,
            )
            row["error"] = str(exc)
            append_csv(csv_path, row, fieldnames)
            return float("inf")

    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

    reeval_candidates = _top_trial_configs(study, cfg.reeval_top_n)
    best_row: dict[str, float | int | str] | None = None
    if len(reeval_candidates) > 0:
        for source_trial_num, controller_config in reeval_candidates:
            metrics, _ = _evaluate_config(
                scenario=scenario,
                controller_config=controller_config,
                seeds=cfg.reeval_seeds,
                team_size=cfg.team_size,
                steps=cfg.steps,
            )
            objective_value, safety_penalized = _objective_from_metrics(
                metrics,
                safety_max=cfg.safety_max,
                safety_penalty=cfg.safety_penalty,
            )
            row = _as_csv_row(
                cfg=cfg,
                phase="reeval",
                trial_number=source_trial_num,
                controller_config=controller_config,
                metrics=metrics,
                objective=objective_value,
                seeds=cfg.reeval_seeds,
                safety_penalized=safety_penalized,
                is_baseline=_config_key(controller_config) == _config_key(baseline),
            )
            append_csv(csv_path, row, fieldnames)
            if best_row is None or float(row["objective"]) < float(best_row["objective"]):
                best_row = row

    if best_row is None:
        if study.best_trial.value is None or not math.isfinite(float(study.best_trial.value)):
            raise RuntimeError("BO study completed without a finite best objective.")
        best_params = dict(study.best_trial.params)
        best_config = {
            "alpha_cross": float(best_params["alpha_cross"]),
            "ell_x": float(best_params["ell_x"]),
            "weight_stein": float(best_params["weight_stein"]),
            "theta": float(best_params["theta"]),
            "history_window": int(best_params["history_window"]),
            "horizon": int(best_params["horizon"]),
        }
        metrics, _ = _evaluate_config(
            scenario=scenario,
            controller_config=best_config,
            seeds=cfg.search_seeds,
            team_size=cfg.team_size,
            steps=cfg.steps,
        )
        objective_value, safety_penalized = _objective_from_metrics(
            metrics,
            safety_max=cfg.safety_max,
            safety_penalty=cfg.safety_penalty,
        )
        best_row = _as_csv_row(
            cfg=cfg,
            phase="best_search_only",
            trial_number=int(study.best_trial.number),
            controller_config=best_config,
            metrics=metrics,
            objective=objective_value,
            seeds=cfg.search_seeds,
            safety_penalized=safety_penalized,
            is_baseline=_config_key(best_config) == _config_key(baseline),
        )

    print("Best BO configuration:")
    print(
        {
            "objective": best_row["objective"],
            "team_ergodic_error": best_row["team_ergodic_error"],
            "alpha_cross": best_row["alpha_cross"],
            "ell_x": best_row["ell_x"],
            "weight_stein": best_row["weight_stein"],
            "theta": best_row["theta"],
            "history_window": best_row["history_window"],
            "horizon": best_row["horizon"],
        }
    )
    return best_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/open_multimodal_bo.yaml")
    run_bo(parser.parse_args().config)

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from experiments.bo_config import BOConfig, load_bo_config
from experiments.run_single_trial import run_single_trial
from experiments.scenarios import load_yaml_scenario

try:
    import optuna
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Optuna is required for experiments/run_bo.py. Install dependencies including 'optuna'."
    ) from exc


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
        row = run_single_trial(
            scenario=scenario,
            controller_config=controller_config,
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
    bo_config_path: str = "configs/sweeps/open_multimodal_bo.yaml",
) -> dict[str, float | int | str]:
    cfg = load_bo_config(bo_config_path)
    scenario = load_yaml_scenario(
        config_path=cfg.scenario_config_path,
        scenario_name=cfg.scenario_name,
    )
    csv_path = Path(cfg.output_csv_path)
    fieldnames = [
        "study_name",
        "phase",
        "trial_number",
        "objective",
        "team_ergodic_error",
        "pairwise_overlap",
        "safety_metric",
        "redundancy_metric",
        "runtime_ms",
        "num_seeds",
        "seed_list",
        "team_size",
        "steps",
        "alpha_cross",
        "ell_x",
        "weight_stein",
        "theta",
        "history_window",
        "horizon",
        "safety_penalized",
        "is_baseline",
    ]

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
            _append_row(csv_path, row, fieldnames)
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
            if "error" not in fieldnames:
                fieldnames.append("error")
            _append_row(csv_path, row, fieldnames)
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
            _append_row(csv_path, row, fieldnames)
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
    run_bo()

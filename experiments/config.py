from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SweepConfig:
    scenario_name: str
    team_size: int
    real_time_budget_ms: float
    alpha_cross_values: list[float]
    ell_x_values: list[float]
    weight_stein_values: list[float]
    theta_values: list[float]
    history_window_values: list[int]
    horizon_values: list[int]
    num_seeds: int
    seeds: list[int] | None = None

    def iter_seeds(self) -> list[int]:
        if self.seeds is not None and len(self.seeds) > 0:
            return list(self.seeds)
        return list(range(self.num_seeds))

    def parameter_grid(self) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for alpha_cross in self.alpha_cross_values:
            for ell_x in self.ell_x_values:
                for weight_stein in self.weight_stein_values:
                    for theta in self.theta_values:
                        for history_window in self.history_window_values:
                            for horizon in self.horizon_values:
                                rows.append(
                                    {
                                        "alpha_cross": float(alpha_cross),
                                        "ell_x": float(ell_x),
                                        "weight_stein": float(weight_stein),
                                        "theta": float(theta),
                                        "history_window": int(history_window),
                                        "horizon": int(horizon),
                                    }
                                )
        return rows


def _as_list(data: dict, key: str, default: list[float] | list[int]) -> list:
    value = data.get(key, default)
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"sweep key '{key}' must be a non-empty list")
    return value


def load_sweep_config(path: str | Path) -> SweepConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sweep config must be a YAML mapping")

    seeds = cfg.get("seeds")
    if seeds is not None and (not isinstance(seeds, list) or len(seeds) == 0):
        raise ValueError("'seeds' must be a non-empty list when provided")

    return SweepConfig(
        scenario_name=str(cfg.get("scenario_name", "open_multimodal")),
        team_size=int(cfg.get("team_size", 4)),
        real_time_budget_ms=float(cfg.get("real_time_budget_ms", 100.0)),
        alpha_cross_values=[float(x) for x in _as_list(cfg, "alpha_cross_values", [20.0, 50.0])],
        ell_x_values=[float(x) for x in _as_list(cfg, "ell_x_values", [0.5, 1.0, 2.0])],
        weight_stein_values=[float(x) for x in _as_list(cfg, "weight_stein_values", [20.0, 45.0])],
        theta_values=[float(x) for x in _as_list(cfg, "theta_values", [45.0, 60.0])],
        history_window_values=[int(x) for x in _as_list(cfg, "history_window_values", [50, 100])],
        horizon_values=[int(x) for x in _as_list(cfg, "horizon_values", [100, 150, 250])],
        num_seeds=int(cfg.get("num_seeds", 3)),
        seeds=[int(x) for x in seeds] if seeds is not None else None,
    )


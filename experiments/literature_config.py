from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


_ALLOWED_METHODS = {"mppi", "smc", "hedac", "traj_opt", "dec"}


@dataclass(frozen=True)
class GMMSpec:
    name: str
    weights: np.ndarray  # (M,)
    means: np.ndarray  # (M, 2)
    covariances: np.ndarray  # (M, 2, 2)


@dataclass(frozen=True)
class HedacConfig:
    grid_size: int
    jacobi_iterations: int
    diffusion_gain: float
    damping: float
    gradient_gain: float


@dataclass(frozen=True)
class TrajOptConfig:
    iterations: int
    learning_rate: float
    control_weight: float
    smoothness_weight: float
    bounds_weight: float


@dataclass(frozen=True)
class LiteratureComparisonConfig:
    scenario_name: str
    scenario_config_path: str
    output_csv_path: str
    summary_csv_path: str
    convergence_csv_path: str
    plot_output_dir: str
    team_size: int
    steps: int
    seeds: list[int]
    d_thresh: float
    methods: list[str]
    fourier_order: int
    desired_speed: float
    tracker_gain: float
    smc_gain: float
    dec_gain: float
    hedac: HedacConfig
    traj_opt: TrajOptConfig
    scenarios: list[GMMSpec]



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



def _as_list_str(cfg: dict, key: str, default: list[str]) -> list[str]:
    value = cfg.get(key, default)
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"'{key}' must be a non-empty list")
    out = [str(x).strip() for x in value]
    for method in out:
        if method not in _ALLOWED_METHODS:
            raise ValueError(f"unsupported method '{method}' in '{key}'")
    if len(set(out)) != len(out):
        raise ValueError(f"'{key}' must not contain duplicates")
    return out



def _as_gmm_spec(raw: dict, key_prefix: str) -> GMMSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"'{key_prefix}' must be a mapping")

    name = str(raw.get("name", "")).strip()
    if len(name) == 0:
        raise ValueError(f"'{key_prefix}.name' must be non-empty")

    weights = np.asarray(raw.get("weights"), dtype=np.float64)
    means = np.asarray(raw.get("means"), dtype=np.float64)
    covs = np.asarray(raw.get("covariances"), dtype=np.float64)

    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(f"'{key_prefix}.weights' must have shape (M,)")
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError(f"'{key_prefix}.means' must have shape (M, 2)")
    if covs.ndim != 3 or covs.shape[1:] != (2, 2):
        raise ValueError(f"'{key_prefix}.covariances' must have shape (M, 2, 2)")
    if means.shape[0] != weights.shape[0] or covs.shape[0] != weights.shape[0]:
        raise ValueError(
            f"'{key_prefix}' must have the same mode count for weights, means, and covariances"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"'{key_prefix}.weights' contains non-finite values")
    if not np.all(np.isfinite(means)):
        raise ValueError(f"'{key_prefix}.means' contains non-finite values")
    if not np.all(np.isfinite(covs)):
        raise ValueError(f"'{key_prefix}.covariances' contains non-finite values")
    if np.any(weights <= 0.0):
        raise ValueError(f"'{key_prefix}.weights' must contain strictly positive values")

    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) > 1e-3:
        raise ValueError(f"'{key_prefix}.weights' must sum to 1.0 (got {weight_sum:.6f})")

    for i, cov in enumerate(covs):
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError(f"'{key_prefix}.covariances[{i}]' must be symmetric")
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0.0):
            raise ValueError(f"'{key_prefix}.covariances[{i}]' must be positive definite")

    return GMMSpec(
        name=name,
        weights=weights,
        means=means,
        covariances=covs,
    )



def load_literature_comparison_config(path: str | Path) -> LiteratureComparisonConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("literature comparison config must be a YAML mapping")

    output_csv_path = str(
        cfg.get("output_csv_path", "results/dars2026/literature/literature_comparison.csv")
    )
    summary_csv_default = str(Path(output_csv_path).with_name(Path(output_csv_path).stem + "_summary.csv"))
    convergence_csv_default = str(
        Path(output_csv_path).with_name(Path(output_csv_path).stem + "_convergence.csv")
    )
    plot_output_default = str(Path(output_csv_path).with_name("plots"))

    scenarios_raw = cfg.get("scenarios")
    if not isinstance(scenarios_raw, list) or len(scenarios_raw) == 0:
        raise ValueError("'scenarios' must be a non-empty list")
    scenarios = [_as_gmm_spec(item, f"scenarios[{i}]") for i, item in enumerate(scenarios_raw)]
    scenario_names = [s.name for s in scenarios]
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario names must be unique")

    hedac_cfg = cfg.get("hedac", {})
    if hedac_cfg is None:
        hedac_cfg = {}
    if not isinstance(hedac_cfg, dict):
        raise ValueError("'hedac' must be a mapping")

    traj_cfg = cfg.get("traj_opt", {})
    if traj_cfg is None:
        traj_cfg = {}
    if not isinstance(traj_cfg, dict):
        raise ValueError("'traj_opt' must be a mapping")

    return LiteratureComparisonConfig(
        scenario_name=str(cfg.get("scenario_name", "literature_comparison")),
        scenario_config_path=str(cfg.get("scenario_config_path", "configs/mppi_params.yaml")),
        output_csv_path=output_csv_path,
        summary_csv_path=str(cfg.get("summary_csv_path", summary_csv_default)),
        convergence_csv_path=str(cfg.get("convergence_csv_path", convergence_csv_default)),
        plot_output_dir=str(cfg.get("plot_output_dir", plot_output_default)),
        team_size=_as_int(cfg.get("team_size", 4), "team_size", min_value=1),
        steps=_as_int(cfg.get("steps", 5000), "steps", min_value=1),
        seeds=_as_list_int(cfg, "seeds", min_value=0) if "seeds" in cfg else [0, 1, 2],
        d_thresh=_as_float(cfg.get("d_thresh", 1.0), "d_thresh", min_value=0.0),
        methods=_as_list_str(cfg, "methods", ["mppi", "smc", "hedac", "traj_opt", "dec"]),
        fourier_order=_as_int(cfg.get("fourier_order", 5), "fourier_order", min_value=1),
        desired_speed=_as_float(cfg.get("desired_speed", 1.5), "desired_speed", min_value=0.0),
        tracker_gain=_as_float(cfg.get("tracker_gain", 3.0), "tracker_gain", min_value=0.0),
        smc_gain=_as_float(cfg.get("smc_gain", 2.5), "smc_gain", min_value=0.0),
        dec_gain=_as_float(cfg.get("dec_gain", 2.5), "dec_gain", min_value=0.0),
        hedac=HedacConfig(
            grid_size=_as_int(hedac_cfg.get("grid_size", 80), "hedac.grid_size", min_value=8),
            jacobi_iterations=_as_int(
                hedac_cfg.get("jacobi_iterations", 60),
                "hedac.jacobi_iterations",
                min_value=1,
            ),
            diffusion_gain=_as_float(
                hedac_cfg.get("diffusion_gain", 1.0),
                "hedac.diffusion_gain",
                min_value=0.0,
            ),
            damping=_as_float(hedac_cfg.get("damping", 0.2), "hedac.damping", min_value=0.0),
            gradient_gain=_as_float(
                hedac_cfg.get("gradient_gain", 8.0),
                "hedac.gradient_gain",
                min_value=0.0,
            ),
        ),
        traj_opt=TrajOptConfig(
            iterations=_as_int(traj_cfg.get("iterations", 80), "traj_opt.iterations", min_value=1),
            learning_rate=_as_float(
                traj_cfg.get("learning_rate", 0.08),
                "traj_opt.learning_rate",
                min_value=1e-8,
            ),
            control_weight=_as_float(
                traj_cfg.get("control_weight", 1e-4),
                "traj_opt.control_weight",
                min_value=0.0,
            ),
            smoothness_weight=_as_float(
                traj_cfg.get("smoothness_weight", 5e-4),
                "traj_opt.smoothness_weight",
                min_value=0.0,
            ),
            bounds_weight=_as_float(
                traj_cfg.get("bounds_weight", 10.0),
                "traj_opt.bounds_weight",
                min_value=0.0,
            ),
        ),
        scenarios=scenarios,
    )

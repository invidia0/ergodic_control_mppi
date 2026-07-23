"""Shared scenario, variant, CSV, summary, and trial utilities."""

import csv
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import AppConfig, load_config
from ergodic_control_mppi.metrics.evaluate import TrialData, compute_all_metrics
from ergodic_control_mppi.mppi.stein import pdf
from ergodic_control_mppi.parameters import ControllerParams, ControllerVariant, RunConfig, apply_variant
from ergodic_control_mppi.simulation import run_simulation


@dataclass(frozen=True)
class Scenario:
    """Controller inputs and normalized grid representation of one scenario."""

    name: str
    params: ControllerParams
    run_config: RunConfig
    target_density_grid: np.ndarray
    map_x_limits: tuple[float, float]
    map_y_limits: tuple[float, float]
    obstacle_map: np.ndarray
    safety_radius: float


def build_target_grid(params: ControllerParams, grid_shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    """Evaluate and normalize the configured GMM on a regular grid."""
    ny, nx = grid_shape
    x = jnp.linspace(params.workspace.x_limits[0], params.workspace.x_limits[1], nx)
    y = jnp.linspace(params.workspace.y_limits[0], params.workspace.y_limits[1], ny)
    grid_x, grid_y = jnp.meshgrid(x, y)
    values = np.asarray(pdf(jnp.stack((grid_x, grid_y), axis=-1), params.gmm), dtype=np.float64)
    total = values.sum()
    if total <= 0:
        raise ValueError("target grid mass must be positive")
    return values / total


def load_scenario(
    config_path: str = "configs/mppi_params.yaml",
    scenario_name: str = "yaml_default",
    grid_shape: tuple[int, int] = (80, 80),
    safety_radius: float | None = None,
) -> Scenario:
    """Load one YAML scenario through the package configuration entrypoint."""
    config = load_config(config_path)
    params = config.controller
    workspace = params.workspace
    return Scenario(
        scenario_name,
        params,
        config.run,
        build_target_grid(params, grid_shape),
        tuple(map(float, workspace.x_limits)),
        tuple(map(float, workspace.y_limits)),
        np.asarray(workspace.obstacles),
        float(workspace.safe_distance if safety_radius is None else safety_radius),
    )


def variant_from_mapping(values: dict[str, Any]) -> ControllerVariant:
    """Validate established experiment-column names into one typed variant."""
    allowed = {"alpha_cross", "ell_x", "weight_stein", "theta", "horizon", "history_window", "history_len"}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unknown controller variant fields: {sorted(unknown)}")
    history = values.get("history_window", values.get("history_len"))
    return ControllerVariant(
        repulsion_weight=float(values["alpha_cross"]) if "alpha_cross" in values else None,
        cross_bandwidth=float(values["ell_x"]) if "ell_x" in values else None,
        flow_weight=float(values["weight_stein"]) if "weight_stein" in values else None,
        theta=float(values["theta"]) if "theta" in values else None,
        horizon=int(values["horizon"]) if "horizon" in values else None,
        history_length=int(history) if history is not None else None,
    )


def run_trial(
    scenario: Scenario,
    variant: ControllerVariant,
    seed: int,
    team_size: int | None = None,
    steps: int | None = None,
    pairwise_d_thresh: float = 1.0,
) -> dict[str, float | int | str]:
    """Run one CPU trial and return its stable flat scalar CSV row."""
    team = scenario.run_config.num_robots if team_size is None else int(team_size)
    count = scenario.run_config.steps if steps is None else int(steps)
    if team < 1 or count < 1:
        raise ValueError("team_size and steps must be >= 1")
    run = replace(scenario.run_config, seed=int(seed), num_robots=team, steps=count)
    params = apply_variant(scenario.params, variant)
    start = time.perf_counter()
    result = run_simulation(AppConfig(run, params), device="cpu")
    runtime_ms = (time.perf_counter() - start) * 1000
    metrics = compute_all_metrics(
        TrialData(
            result.paths,
            scenario.target_density_grid,
            scenario.map_x_limits,
            scenario.map_y_limits,
            scenario.obstacle_map,
            scenario.safety_radius,
        ),
        pairwise_d_thresh,
    )
    row: dict[str, float | int | str] = {
        "scenario_name": scenario.name,
        "seed": int(seed),
        "team_size": team,
        "steps": count,
        "runtime_ms": float(runtime_ms),
    }
    for name, value in (
        ("alpha_cross", variant.repulsion_weight),
        ("ell_x", variant.cross_bandwidth),
        ("weight_stein", variant.flow_weight),
        ("theta", variant.theta),
        ("horizon", variant.horizon),
        ("history_window", variant.history_length),
    ):
        if value is not None:
            row[name] = value
    row.update(metrics)
    return row


def append_csv(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one row, creating its parent and header when needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    new = not output.exists()
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow(row)


def prepare_outputs(paths: Iterable[str | Path], overwrite: bool) -> None:
    """Protect result files from replacement unless explicitly authorized."""
    outputs = [Path(path) for path in paths]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"result already exists: {existing[0]}; pass --overwrite to replace it")
    for path in existing:
        path.unlink()


def summarize(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, float]:
    """Return mean and sample standard deviation columns for scalar metrics."""
    result: dict[str, float] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows])
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    return result

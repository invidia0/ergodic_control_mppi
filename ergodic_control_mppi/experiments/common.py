"""Shared scenario, CSV, and summary utilities."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.stein import pdf
from ergodic_control_mppi.parameters import ControllerParams, RunConfig


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

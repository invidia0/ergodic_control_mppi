from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from configs import params_loader
from mppi.stein import pdf


@dataclass(frozen=True)
class Scenario:
    name: str
    params: object
    run_config: params_loader.RunConfig
    target_density_grid: np.ndarray
    map_x_limits: tuple[float, float]
    map_y_limits: tuple[float, float]
    obstacle_map: np.ndarray
    safety_radius: float


def _build_target_grid(params, grid_shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    ny, nx = grid_shape
    xs = jnp.linspace(params.map_x_limits[0], params.map_x_limits[1], nx)
    ys = jnp.linspace(params.map_y_limits[0], params.map_y_limits[1], ny)
    gx, gy = jnp.meshgrid(xs, ys)
    points = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
    vals = jax.vmap(pdf, in_axes=(0, None))(points, params.stein).reshape((ny, nx))
    arr = np.asarray(vals, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("target grid mass must be positive")
    return arr / total


def load_yaml_scenario(
    config_path: str = "configs/mppi_params.yaml",
    scenario_name: str = "yaml_default",
    grid_shape: tuple[int, int] = (80, 80),
    safety_radius: float | None = None,
) -> Scenario:
    params = params_loader.load_mppi_params(config_path)
    run_cfg = params_loader.load_run_config(config_path)
    target_grid = _build_target_grid(params, grid_shape=grid_shape)
    safe_r = float(safety_radius if safety_radius is not None else params.obstacle_params.safe_distance)
    return Scenario(
        name=scenario_name,
        params=params,
        run_config=run_cfg,
        target_density_grid=target_grid,
        map_x_limits=(float(params.map_x_limits[0]), float(params.map_x_limits[1])),
        map_y_limits=(float(params.map_y_limits[0]), float(params.map_y_limits[1])),
        obstacle_map=np.asarray(params.obstacle_params.xyr),
        safety_radius=safe_r,
    )


"""Single-pass YAML loading and validation."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ergodic_control_mppi.models.double_integrator import DoubleIntegratorParams
from ergodic_control_mppi.parameters import (
    ControllerParams,
    GMMParams,
    MPPIParams,
    RunConfig,
    SteinParams,
    WorkspaceParams,
)


@dataclass(frozen=True)
class AppConfig:
    """Validated run and controller configuration."""

    run: RunConfig
    controller: ControllerParams


def _mapping(parent: dict[str, Any], key: str, path: str = "") -> dict[str, Any]:
    full = f"{path}.{key}" if path else key
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{full}' must be a mapping")
    return value


def _required(parent: dict[str, Any], key: str, path: str) -> Any:
    if key not in parent:
        raise ValueError(f"Missing required config key: '{path}.{key}'")
    return parent[key]


def _integer(value: Any, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Config key '{path}' must be an exact integer")
    result = int(value)
    if result != value:
        raise ValueError(f"Config key '{path}' must be an exact integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"Config key '{path}' must be >= {minimum}")
    return result


def _number(value: Any, path: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Config key '{path}' must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{path}' must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"Config key '{path}' must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"Config key '{path}' must be >= {minimum}")
    return result


def _array(value: Any, path: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{path}' must be a numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Config key '{path}' contains non-finite values")
    if shape is not None and result.shape != shape:
        raise ValueError(f"Config key '{path}' must have shape {shape}, got {result.shape}")
    return result


def _limits(value: Any, path: str) -> np.ndarray:
    result = _array(value, path, (2,))
    if result[0] >= result[1]:
        raise ValueError(f"Config key '{path}' must satisfy lower < upper")
    return result


def _positive_definite(value: Any, path: str, shape: tuple[int, ...]) -> np.ndarray:
    result = _array(value, path, shape)
    matrices = result[None, ...] if result.ndim == 2 else result
    for matrix in matrices:
        if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Config key '{path}' must contain symmetric matrices")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Config key '{path}' must contain positive-definite matrices") from exc
    return result


def _generate_obstacles(
    count: int,
    x_limits: np.ndarray,
    y_limits: np.ndarray,
    radius_min: float,
    radius_max: float,
    seed: int,
) -> jax.Array:
    """Generate deterministic circular obstacles with shape ``(count, 3)``."""
    _, key_x, key_y, key_r = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.uniform(key_x, (count,), minval=x_limits[0], maxval=x_limits[1])
    y = jax.random.uniform(key_y, (count,), minval=y_limits[0], maxval=y_limits[1])
    radius = jax.random.uniform(key_r, (count,), minval=radius_min, maxval=radius_max)
    return jnp.stack((x, y, radius), axis=-1).astype(jnp.float32)


def load_config(path: str | Path) -> AppConfig:
    """Read and validate a YAML configuration exactly once.

    Args:
        path: YAML file using the repository controller schema.

    Returns:
        Immutable run and controller parameters.

    Raises:
        ValueError: If a required value is missing, malformed, non-finite, or
            outside its supported range.
        OSError: If ``path`` cannot be read.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML content must be a mapping")

    robots = _mapping(raw, "robots")
    mppi_raw = _mapping(raw, "mppi")
    noise = _mapping(mppi_raw, "noise", "mppi")
    map_raw = _mapping(raw, "map")
    obstacles_raw = _mapping(map_raw, "obstacles", "map")
    density = _mapping(raw, "density")
    stein_raw = _mapping(raw, "stein")
    model_raw = _mapping(raw, "model")
    model_specific = _mapping(model_raw, "double_integrator", "model")

    for removed in ("dim_x", "dim_u"):
        if removed in mppi_raw:
            raise ValueError(f"Config key 'mppi.{removed}' was removed; the model dimensions are fixed")
    if "weight_pdf" in stein_raw:
        raise ValueError("Config key 'stein.weight_pdf' was removed because it was inactive")

    seed = _integer(raw.get("seed", 0), "seed")
    resolution = _number(_required(map_raw, "resolution", "map"), "map.resolution", 1e-12)
    run = RunConfig(
        seed=seed,
        steps=_integer(raw.get("steps", 5000), "steps", 1),
        num_robots=_integer(robots.get("num_robots", 1), "robots.num_robots", 1),
        resolution=resolution,
    )

    samples = _integer(_required(mppi_raw, "K", "mppi"), "mppi.K", 1)
    horizon = _integer(_required(mppi_raw, "T", "mppi"), "mppi.T", 1)
    temperature = _number(_required(mppi_raw, "lambda", "mppi"), "mppi.lambda", 1e-12)
    alpha = _number(_required(mppi_raw, "alpha", "mppi"), "mppi.alpha", 0.0)
    if alpha > 1:
        raise ValueError("Config key 'mppi.alpha' must be <= 1")
    exploration = _number(_required(mppi_raw, "exploration", "mppi"), "mppi.exploration", 0.0)
    if exploration > 1:
        raise ValueError("Config key 'mppi.exploration' must be <= 1")
    ess_target = _number(mppi_raw.get("ess_target", 0.3), "mppi.ess_target", 1e-12)
    if ess_target >= 1:
        raise ValueError("Config key 'mppi.ess_target' must be < 1")
    temperature_min = _number(mppi_raw.get("lam_min", 0.05), "mppi.lam_min", 1e-12)
    temperature_max = _number(mppi_raw.get("lam_max", 20.0), "mppi.lam_max", 1e-12)
    if temperature_max <= temperature_min:
        raise ValueError("Config key 'mppi.lam_max' must exceed mppi.lam_min")
    covariance_np = _positive_definite(
        _required(noise, "sigma", "mppi.noise"), "mppi.noise.sigma", (3, 3)
    )

    means = _array(_required(density, "means", "density"), "density.means")
    weights = _array(_required(density, "weights", "density"), "density.weights")
    if means.ndim != 2 or means.shape[1] != 2 or means.shape[0] == 0:
        raise ValueError(f"Config key 'density.means' must have shape (M, 2), got {means.shape}")
    covariances = _positive_definite(
        _required(density, "covariances", "density"),
        "density.covariances",
        (means.shape[0], 2, 2),
    )
    if weights.shape != (means.shape[0],):
        raise ValueError(f"Config key 'density.weights' must have shape ({means.shape[0]},)")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Config key 'density.weights' must be positive and sum to 1")
    covariance_inverse = np.linalg.inv(covariances).astype(np.float32)
    log_normalizers = (-0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(covariances)[1])).astype(np.float32)
    gmm = GMMParams(
        means=jnp.asarray(means),
        covariance_inverse=jnp.asarray(covariance_inverse),
        log_weights=jnp.log(jnp.asarray(weights)),
        log_normalizers=jnp.asarray(log_normalizers),
    )

    theta = _number(_required(stein_raw, "theta", "stein"), "stein.theta", 0.0)
    if theta > 90:
        raise ValueError("Config key 'stein.theta' must be in [0, 90]")
    theta_radians = np.deg2rad(theta)
    stein = SteinParams(
        rotation=jnp.asarray(
            [[np.cos(theta_radians), -np.sin(theta_radians)],
             [np.sin(theta_radians), np.cos(theta_radians)]],
            dtype=jnp.float32,
        ),
        self_bandwidth=_number(_required(stein_raw, "ell_self", "stein"), "stein.ell_self", 1e-12),
        cross_bandwidth=_number(_required(stein_raw, "ell_x", "stein"), "stein.ell_x", 1e-12),
        flow_weight=_number(_required(stein_raw, "weight_stein", "stein"), "stein.weight_stein", 0.0),
        repulsion_weight=_number(_required(stein_raw, "alpha_cross", "stein"), "stein.alpha_cross", 0.0),
    )

    map_x = _limits(_required(map_raw, "x_limits", "map"), "map.x_limits")
    map_y = _limits(_required(map_raw, "y_limits", "map"), "map.y_limits")
    obstacle_x = _limits(_required(obstacles_raw, "x_limits", "map.obstacles"), "map.obstacles.x_limits")
    obstacle_y = _limits(_required(obstacles_raw, "y_limits", "map.obstacles"), "map.obstacles.y_limits")
    obstacle_count = _integer(
        _required(obstacles_raw, "num_obstacles", "map.obstacles"),
        "map.obstacles.num_obstacles",
        0,
    )
    radius_min = _number(
        _required(obstacles_raw, "min_radius", "map.obstacles"),
        "map.obstacles.min_radius",
        0.0,
    )
    radius_max = _number(
        _required(obstacles_raw, "max_radius", "map.obstacles"),
        "map.obstacles.max_radius",
        0.0,
    )
    if radius_max < radius_min:
        raise ValueError("Config key 'map.obstacles.max_radius' must be >= min_radius")
    workspace = WorkspaceParams(
        x_limits=jnp.asarray(map_x),
        y_limits=jnp.asarray(map_y),
        out_of_map_cost=_number(_required(map_raw, "oom_cost", "map"), "map.oom_cost", 0.0),
        obstacles=_generate_obstacles(
            obstacle_count, obstacle_x, obstacle_y, radius_min, radius_max, seed
        ),
        obstacle_cost=_number(
            _required(obstacles_raw, "weight", "map.obstacles"),
            "map.obstacles.weight",
            0.0,
        ),
        safe_distance=_number(
            _required(obstacles_raw, "safe_distance", "map.obstacles"),
            "map.obstacles.safe_distance",
            0.0,
        ),
    )
    mppi = MPPIParams(
        samples=samples,
        horizon=horizon,
        history_length=_integer(mppi_raw.get("history_len", 100), "mppi.history_len", 1),
        temperature=temperature,
        alpha=alpha,
        exploration=exploration,
        ess_target=ess_target,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        covariance=jnp.asarray(covariance_np),
        covariance_inverse=jnp.asarray(np.linalg.inv(covariance_np).astype(np.float32)),
    )
    model = DoubleIntegratorParams(
        delta_t=_number(_required(model_raw, "delta_t", "model"), "model.delta_t", 1e-12),
        max_accel_lin_abs=_number(
            _required(model_specific, "max_accel_lin_abs", "model.double_integrator"),
            "model.double_integrator.max_accel_lin_abs",
            0.0,
        ),
        max_accel_ang_abs=_number(
            _required(model_specific, "max_accel_ang_abs", "model.double_integrator"),
            "model.double_integrator.max_accel_ang_abs",
            0.0,
        ),
    )
    return AppConfig(run=run, controller=ControllerParams(mppi, gmm, stein, workspace, model))

import yaml

import jax
import jax.numpy as jnp

from mppi.core import MPPIParams, ObstacleParams
from mppi.stein import SteinParams
from models.double_integrator import DoubleIntegratorParams


def _require_dict(parent: dict, key: str, parent_path: str) -> dict:
    path = f"{parent_path}.{key}" if parent_path else key
    value = parent.get(key)
    if value is None:
        raise ValueError(f"Missing required config section: '{path}'")
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{path}' must be a mapping")
    return value


def _require_value(parent: dict, key: str, parent_path: str):
    path = f"{parent_path}.{key}" if parent_path else key
    if key not in parent:
        raise ValueError(f"Missing required config key: '{path}'")
    return parent[key]


def _as_int(value, path: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Config key '{path}' must be an integer, got bool")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{path}' must be an integer") from exc
    if min_value is not None and out < min_value:
        raise ValueError(f"Config key '{path}' must be >= {min_value}")
    return out


def _as_float(value, path: str, *, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{path}' must be a float") from exc
    if min_value is not None and out < min_value:
        raise ValueError(f"Config key '{path}' must be >= {min_value}")
    return out


def _float32_array(value, path: str) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=jnp.float32)
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"Config key '{path}' contains non-finite values")
    return arr


def _expect_shape(arr: jnp.ndarray, shape: tuple[int, ...], path: str) -> None:
    if arr.shape != shape:
        raise ValueError(f"Config key '{path}' must have shape {shape}, got {arr.shape}")


def _validate_limits(limits: jnp.ndarray, path: str) -> None:
    _expect_shape(limits, (2,), path)
    if float(limits[0]) >= float(limits[1]):
        raise ValueError(f"Config key '{path}' must satisfy lower < upper")


def _obstacle_generator(
    num_obstacles: int,
    x_limits: jnp.ndarray,
    y_limits: jnp.ndarray,
    r_min: float,
    r_max: float,
    key: jax.Array,
) -> tuple[jnp.ndarray, jax.Array]:
    """
    Generate random circular obstacles.

    Returns:
        xyr: jnp.ndarray of shape (num_obstacles, 3) representing (x, y, r)
        key: updated PRNG key
    """
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    xs = jax.random.uniform(
        subkey1,
        shape=(num_obstacles,),
        minval=x_limits[0],
        maxval=x_limits[1],
    )

    ys = jax.random.uniform(
        subkey2,
        shape=(num_obstacles,),
        minval=y_limits[0],
        maxval=y_limits[1],
    )

    rs = jax.random.uniform(
        subkey3,
        shape=(num_obstacles,),
        minval=r_min,
        maxval=r_max,
    )

    xyr = jnp.stack([xs, ys, rs], axis=1)
    return xyr, key


def load_mppi_params(yaml_path: str) -> MPPIParams:
    """
    Load canonical MPPI config from YAML and build runtime params.

    Canonical schema (strict, no legacy aliases):
      - mppi.noise
      - map.obstacles
      - density
      - stein.drift
      - model.delta_t (single source of truth for timestep)
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML content must be a mapping")

    mppi_cfg = _require_dict(cfg, "mppi", "")
    noise_cfg = _require_dict(mppi_cfg, "noise", "mppi")
    stein_cfg = _require_dict(cfg, "stein", "")
    drift_cfg = _require_dict(stein_cfg, "drift", "stein")
    map_cfg = _require_dict(cfg, "map", "")
    obstacle_cfg = _require_dict(map_cfg, "obstacles", "map")
    density_cfg = _require_dict(cfg, "density", "")
    model_cfg = _require_dict(cfg, "model", "")
    model_spec = _require_dict(model_cfg, "double_integrator", "model")

    seed = _as_int(cfg.get("seed", 0), "seed")
    key = jax.random.PRNGKey(seed)

    K = _as_int(_require_value(mppi_cfg, "K", "mppi"), "mppi.K", min_value=1)
    T = _as_int(_require_value(mppi_cfg, "T", "mppi"), "mppi.T", min_value=1)
    dim_x = _as_int(_require_value(mppi_cfg, "dim_x", "mppi"), "mppi.dim_x", min_value=1)
    dim_u = _as_int(_require_value(mppi_cfg, "dim_u", "mppi"), "mppi.dim_u", min_value=1)
    lam = _as_float(_require_value(mppi_cfg, "lambda", "mppi"), "mppi.lambda", min_value=0.0)
    alpha = _as_float(_require_value(mppi_cfg, "alpha", "mppi"), "mppi.alpha", min_value=0.0)
    exploration = _as_float(_require_value(mppi_cfg, "exploration", "mppi"), "mppi.exploration", min_value=0.0)
    if exploration > 1.0:
        raise ValueError("Config key 'mppi.exploration' must be <= 1.0")

    Sigma = _float32_array(_require_value(noise_cfg, "sigma", "mppi.noise"), "mppi.noise.sigma")
    _expect_shape(Sigma, (dim_u, dim_u), "mppi.noise.sigma")
    Sigma_inv = jnp.linalg.inv(Sigma + 1e-8 * jnp.eye(dim_u, dtype=jnp.float32))
    if not bool(jnp.all(jnp.isfinite(Sigma_inv))):
        raise ValueError("Computed Sigma_inv contains non-finite values; check mppi.noise.sigma")

    means = _float32_array(_require_value(density_cfg, "means", "density"), "density.means")
    covs = _float32_array(_require_value(density_cfg, "covariances", "density"), "density.covariances")
    weights = _float32_array(_require_value(density_cfg, "weights", "density"), "density.weights")

    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError(f"Config key 'density.means' must have shape (M, 2), got {means.shape}")
    if covs.ndim != 3 or covs.shape[1:] != (2, 2):
        raise ValueError(f"Config key 'density.covariances' must have shape (M, 2, 2), got {covs.shape}")
    if weights.ndim != 1:
        raise ValueError(f"Config key 'density.weights' must have shape (M,), got {weights.shape}")
    if means.shape[0] != covs.shape[0] or means.shape[0] != weights.shape[0]:
        raise ValueError(
            "density.means, density.covariances, and density.weights must share the same leading dimension"
        )
    if bool(jnp.any(weights <= 0.0)):
        raise ValueError("Config key 'density.weights' must contain strictly positive values")
    weight_sum = float(jnp.sum(weights))
    if abs(weight_sum - 1.0) > 1e-3:
        raise ValueError(f"Config key 'density.weights' must sum to 1.0 (got {weight_sum:.6f})")

    cov_inv = jnp.linalg.inv(covs)
    if not bool(jnp.all(jnp.isfinite(cov_inv))):
        raise ValueError("Computed density cov_inv contains non-finite values; check density.covariances")

    log_weights = jnp.log(weights + 1e-10)
    log_det = jnp.log(jnp.linalg.det(covs))
    d = means.shape[1]
    log_norm = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det)

    drift_sigma = _as_float(_require_value(drift_cfg, "sigma", "stein.drift"), "stein.drift.sigma", min_value=0.0)
    S = _float32_array(_require_value(drift_cfg, "S", "stein.drift"), "stein.drift.S")
    _expect_shape(S, (2, 2), "stein.drift.S")
    # Keep existing behavior: linear scaling of diffusion by sigma.
    D = 0.5 * drift_sigma * jnp.eye(2, dtype=jnp.float32)

    map_x_limits = _float32_array(_require_value(map_cfg, "x_limits", "map"), "map.x_limits")
    map_y_limits = _float32_array(_require_value(map_cfg, "y_limits", "map"), "map.y_limits")
    _validate_limits(map_x_limits, "map.x_limits")
    _validate_limits(map_y_limits, "map.y_limits")
    resolution = _as_float(_require_value(map_cfg, "resolution", "map"), "map.resolution", min_value=1e-12)
    oom_cost = _as_float(_require_value(map_cfg, "oom_cost", "map"), "map.oom_cost", min_value=0.0)

    num_obstacles = _as_int(_require_value(obstacle_cfg, "num_obstacles", "map.obstacles"), "map.obstacles.num_obstacles", min_value=1)
    obs_x_limits = _float32_array(_require_value(obstacle_cfg, "x_limits", "map.obstacles"), "map.obstacles.x_limits")
    obs_y_limits = _float32_array(_require_value(obstacle_cfg, "y_limits", "map.obstacles"), "map.obstacles.y_limits")
    _validate_limits(obs_x_limits, "map.obstacles.x_limits")
    _validate_limits(obs_y_limits, "map.obstacles.y_limits")
    min_radius = _as_float(_require_value(obstacle_cfg, "min_radius", "map.obstacles"), "map.obstacles.min_radius", min_value=0.0)
    max_radius = _as_float(_require_value(obstacle_cfg, "max_radius", "map.obstacles"), "map.obstacles.max_radius", min_value=0.0)
    if max_radius < min_radius:
        raise ValueError("Config key 'map.obstacles.max_radius' must be >= map.obstacles.min_radius")
    obstacle_weight = _as_float(_require_value(obstacle_cfg, "weight", "map.obstacles"), "map.obstacles.weight", min_value=0.0)
    safe_distance = _as_float(_require_value(obstacle_cfg, "safe_distance", "map.obstacles"), "map.obstacles.safe_distance", min_value=0.0)

    delta_t = _as_float(_require_value(model_cfg, "delta_t", "model"), "model.delta_t", min_value=1e-12)
    max_accel_lin_abs = _as_float(
        _require_value(model_spec, "max_accel_lin_abs", "model.double_integrator"),
        "model.double_integrator.max_accel_lin_abs",
        min_value=0.0,
    )
    max_accel_ang_abs = _as_float(
        _require_value(model_spec, "max_accel_ang_abs", "model.double_integrator"),
        "model.double_integrator.max_accel_ang_abs",
        min_value=0.0,
    )

    num_nominal = jnp.asarray((1.0 - exploration) * K, dtype=jnp.int32)
    use_nominal = jnp.arange(K) < num_nominal
    gamma = lam * (1.0 - alpha)

    xyr, key = _obstacle_generator(
        num_obstacles=num_obstacles,
        x_limits=obs_x_limits,
        y_limits=obs_y_limits,
        r_min=min_radius,
        r_max=max_radius,
        key=key,
    )

    obstacle_params = ObstacleParams(
        xyr=xyr,
        weight=obstacle_weight,
        safe_distance=safe_distance,
    )

    stein_params = SteinParams(
        means=means,
        cov_inv=cov_inv,
        log_weights=log_weights,
        log_norm=log_norm,
        D=D,
        S=S,
        gamma=_as_float(_require_value(stein_cfg, "gamma", "stein"), "stein.gamma"),
        h=_as_float(_require_value(stein_cfg, "h", "stein"), "stein.h", min_value=1e-12),
        weight=_as_float(_require_value(stein_cfg, "weight", "stein"), "stein.weight", min_value=0.0),
        weight_pdf=_as_float(_require_value(stein_cfg, "weight_pdf", "stein"), "stein.weight_pdf", min_value=0.0),
    )

    model_params = DoubleIntegratorParams(
        delta_t=delta_t,
        max_accel_lin_abs=max_accel_lin_abs,
        max_accel_ang_abs=max_accel_ang_abs,
    )

    return MPPIParams(
        T=T,
        K=K,
        dim_x=dim_x,
        dim_u=dim_u,
        lam=lam,
        alpha=alpha,
        gamma=gamma,
        Sigma=Sigma,
        delta_t=delta_t,
        Sigma_inv=Sigma_inv,
        map_x_limits=map_x_limits,
        map_y_limits=map_y_limits,
        resolution=resolution,
        oom_cost=oom_cost,
        stein=stein_params,
        model_params=model_params,
        obstacle_params=obstacle_params,
        use_nominal=use_nominal,
        seed=seed,
    )
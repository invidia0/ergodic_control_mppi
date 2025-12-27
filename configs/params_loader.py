import yaml

import jax
import jax.numpy as jnp

from mppi.mppi_core import MPPIParams, ObstacleParams
from mppi.stein import SteinParams
from models.double_integrator import DoubleIntegratorParams


def obstacle_generator(
num_obstacles: int,
x_limits: jnp.ndarray,
y_limits: jnp.ndarray,
r_min: float,
r_max: float,
key: jax.Array) -> tuple[jnp.ndarray, jax.Array]:
    """
    Generate random circular obstacles.

    Returns:
        xyr: jnp.ndarray of shape (num_obstacles, 3) representing (x, y, r) of each obstacle
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

    xyr = jnp.stack([xs, ys, rs], axis=1)  # (num_obstacles, 3)

    return xyr, key


def load_mppi_params(yaml_path: str) -> MPPIParams:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    mppi_cfg = cfg["mppi"]
    noise_cfg = cfg["noise"]
    stein_cfg = cfg["stein"]
    mp_cfg = cfg["map"]
    gmm2d_cfg = cfg["gmm2d"]
    seed = cfg.get("seed", 0)

    key = jax.random.PRNGKey(seed)

    model_cfg = cfg["model"]

    Sigma = jnp.array(noise_cfg["sigma"], dtype=jnp.float32)
    Sigma_inv = jnp.linalg.inv(Sigma + 1e-8 * jnp.eye(Sigma.shape[0])) # just an approx, should use full inv and tune noise properly

    means = jnp.array(gmm2d_cfg["means"], dtype=jnp.float32)
    covs = jnp.array(gmm2d_cfg["covariances"], dtype=jnp.float32)
    weights = jnp.array(gmm2d_cfg["weights"], dtype=jnp.float32)
    cov_inv = jnp.linalg.inv(covs)
    log_weights = jnp.log(weights + 1e-10)
    log_det = jnp.log(jnp.linalg.det(covs))
    d = means.shape[1]
    log_norm = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det)

    drift_cfg = gmm2d_cfg["drift"]
    sigma = jnp.array(drift_cfg["sigma"], dtype=jnp.float32)
    D = 0.5 * (sigma**2) * jnp.eye(2)
    S = jnp.array(drift_cfg["S"], dtype=jnp.float32)

    num_nominal = jnp.asarray((1.0 - mppi_cfg["exploration"]) * mppi_cfg["K"], dtype=jnp.int32)
    use_nominal = jnp.arange(mppi_cfg["K"]) < num_nominal  # (K,)

    gamma = mppi_cfg["lambda"] * (1.0 - mppi_cfg["alpha"])

    obav_cfg = cfg.get("obstacles", None)
    if obav_cfg is not None:
        xyr, key = obstacle_generator(
            num_obstacles=obav_cfg["num_obstacles"],
            x_limits=jnp.array(obav_cfg["x_limits"], dtype=jnp.float32),
            y_limits=jnp.array(obav_cfg["y_limits"], dtype=jnp.float32),
            r_min=obav_cfg["min_radius"],
            r_max=obav_cfg["max_radius"],
            key=key,
        )
    
    _obav_params = ObstacleParams(
        xyr=xyr,
        weight=obav_cfg["weight"],
        safe_distance=obav_cfg["safe_distance"],
    ) if obav_cfg is not None else None

    _stein = SteinParams(
        means=means,
        cov_inv=cov_inv,
        log_weights=log_weights,
        log_norm=log_norm,
        D=D,
        S=S,
        gamma=stein_cfg["gamma"],
        h=stein_cfg["h"],
        weight=stein_cfg["weight"],
        weight_pdf=stein_cfg["weight_pdf"],
    )

    model_spec = model_cfg["double_integrator"]
    _model_params = DoubleIntegratorParams(
        delta_t=model_cfg["delta_t"],
        max_accel_lin_abs=model_spec["max_accel_lin_abs"],
        max_accel_ang_abs=model_spec["max_accel_ang_abs"],
    )

    mppi_params = MPPIParams(
        T=int(mppi_cfg["T"]),
        K=int(mppi_cfg["K"]),
        dim_x=int(mppi_cfg["dim_x"]),
        dim_u=int(mppi_cfg["dim_u"]),
        lam=mppi_cfg["lambda"],
        alpha=mppi_cfg["alpha"],
        gamma=gamma,
        Sigma=Sigma,
        delta_t=mppi_cfg["delta_t"],
        Sigma_inv=Sigma_inv,
        map_x_limits=jnp.array(mp_cfg["x_limits"]),
        map_y_limits=jnp.array(mp_cfg["y_limits"]),
        resolution=mp_cfg["resolution"],
        oom_cost=mp_cfg["oom_cost"],
        stein=_stein,
        model_params=_model_params,
        obstacle_params=_obav_params,
        use_nominal=use_nominal,
    )

    return mppi_params, seed
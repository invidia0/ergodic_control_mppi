import yaml

import jax
import jax.numpy as jnp

from mppi.mppi_core import MPPIParams
from mppi.stein import SteinParams
from models.double_integrator import DoubleIntegratorParams


def load_mppi_params(yaml_path: str) -> MPPIParams:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    mppi_cfg = cfg["mppi"]
    noise_cfg = cfg["noise"]
    stein_cfg = cfg["stein"]
    mp_cfg = cfg["map"]
    gmm2d_cfg = cfg["gmm2d"]

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
        use_nominal=use_nominal,
    )

    return mppi_params, cfg["seed"]
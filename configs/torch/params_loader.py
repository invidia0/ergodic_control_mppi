import yaml
import torch
import math

from mppi.torch.mppi_core import MPPIParams, ObstacleParams
from mppi.torch.stein import SteinParams
from models.torch.double_integrator import DoubleIntegratorParams

# Global device setting (can be passed as arg in production)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def obstacle_generator(
    num_obstacles: int,
    x_limits: torch.Tensor,
    y_limits: torch.Tensor,
    r_min: float,
    r_max: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate random circular obstacles.
    
    Returns:
        xyr: torch.Tensor of shape (num_obstacles, 3) representing (x, y, r)
    """
    # Generate random values on [0, 1) and scale to limits
    xs = torch.rand(num_obstacles, device=device) * (x_limits[1] - x_limits[0]) + x_limits[0]
    ys = torch.rand(num_obstacles, device=device) * (y_limits[1] - y_limits[0]) + y_limits[0]
    rs = torch.rand(num_obstacles, device=device) * (r_max - r_min) + r_min

    # Stack into (num_obstacles, 3)
    xyr = torch.stack([xs, ys, rs], dim=1)
    
    return xyr

def load_mppi_params(
        yaml_path: str,
        device: torch.device
    ) -> tuple[MPPIParams, int]:
    
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    mppi_cfg = cfg["mppi"]
    noise_cfg = cfg["noise"]
    stein_cfg = cfg["stein"]
    mp_cfg = cfg["map"]
    gmm2d_cfg = cfg["gmm2d"]
    seed = cfg.get("seed", 0)

    # Set Global Seed for Reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_cfg = cfg["model"]

    # --- Noise Parameters ---
    # Convert lists to tensors immediately on device
    Sigma = torch.tensor(noise_cfg["sigma"], dtype=torch.float32, device=device)
    
    # Inverse with regularization
    eye = torch.eye(Sigma.shape[0], device=device)
    Sigma_inv = torch.linalg.inv(Sigma + 1e-8 * eye)

    # --- GMM / Stein Parameters ---
    means = torch.tensor(gmm2d_cfg["means"], dtype=torch.float32, device=device)
    covs = torch.tensor(gmm2d_cfg["covariances"], dtype=torch.float32, device=device)
    weights = torch.tensor(gmm2d_cfg["weights"], dtype=torch.float32, device=device)
    
    cov_inv = torch.linalg.inv(covs)
    log_weights = torch.log(weights + 1e-10)
    
    # Batch determinant
    log_det = torch.log(torch.det(covs)) 
    d = means.shape[1]
    # log_norm = -0.5 * (d * log(2pi) + log_det)
    log_norm = -0.5 * (d * math.log(2 * math.pi) + log_det)

    drift_cfg = gmm2d_cfg["drift"]
    sigma_drift = torch.tensor(drift_cfg["sigma"], dtype=torch.float32, device=device)
    
    D = 0.5 * (sigma_drift**2) * torch.eye(2, device=device)
    S = torch.tensor(drift_cfg["S"], dtype=torch.float32, device=device)

    # --- MPPI Logic ---
    num_nominal = int((1.0 - mppi_cfg["exploration"]) * mppi_cfg["K"])
    # Create boolean mask: True for indices < num_nominal
    use_nominal = torch.arange(mppi_cfg["K"], device=device) < num_nominal

    gamma = mppi_cfg["lambda"] * (1.0 - mppi_cfg["alpha"])

    # --- Obstacles ---
    obav_cfg = cfg.get("obstacles", None)
    _obav_params = None
    
    if obav_cfg is not None:
        # Load limits as tensors for the generator
        x_lim_obs = torch.tensor(obav_cfg["x_limits"], dtype=torch.float32, device=device)
        y_lim_obs = torch.tensor(obav_cfg["y_limits"], dtype=torch.float32, device=device)
        
        xyr = obstacle_generator(
            num_obstacles=obav_cfg["num_obstacles"],
            x_limits=x_lim_obs,
            y_limits=y_lim_obs,
            r_min=obav_cfg["min_radius"],
            r_max=obav_cfg["max_radius"],
            device=device,
        )
        
        _obav_params = ObstacleParams(
            xyr=xyr,
            weight=obav_cfg["weight"],
            safe_distance=obav_cfg["safe_distance"],
        )

    # --- Sub-Structs ---
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

    # --- Final Config ---
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
        map_x_limits=torch.tensor(mp_cfg["x_limits"], device=device),
        map_y_limits=torch.tensor(mp_cfg["y_limits"], device=device),
        resolution=mp_cfg["resolution"],
        oom_cost=mp_cfg["oom_cost"],
        stein=_stein,
        model_params=_model_params,
        obstacle_params=_obav_params,
        use_nominal=use_nominal,
    )

    return mppi_params, seed

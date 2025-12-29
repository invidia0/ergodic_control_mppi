import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, List

from mppi.torch.stein import SteinParams, stein_grad_traj, logpdf
from models.torch import double_integrator as model

# Assumes previous definitions are available in scope:
# from double_integrator import DoubleIntegratorParams, step, clamp
# from stein import SteinParams, stein_grad_traj, logpdf, drift, score_pdf

# Helper to ensure device consistency
def to_device(t, device):
    return t.to(device)

@dataclass(frozen=True)
class ObstacleParams:
    xyr: torch.Tensor  # (num_obstacles, 3) (x, y, r)
    weight: float
    safe_distance: float

@dataclass(frozen=True)
class MPPIParams:
    K: int
    T: int
    dim_u: int
    dim_x: int

    # MPPI scalars
    lam: float
    alpha: float
    gamma: float

    # (K,) boolean tensor indicating which particles use nominal control
    use_nominal: torch.Tensor 

    Sigma: torch.Tensor      # (dim_u, dim_u)
    Sigma_inv: torch.Tensor  # (dim_u, dim_u)

    map_x_limits: torch.Tensor   # (2,)
    map_y_limits: torch.Tensor   # (2,)
    resolution: float
    oom_cost: float

    delta_t: float

    stein: SteinParams
    model_params: model.DoubleIntegratorParams
    obstacle_params: ObstacleParams

def sample_epsilon(params: MPPIParams, device: torch.device) -> torch.Tensor:
    """
    eps ~ N(0, Sigma) with shape (K, T, dim_u)
    """
    # Create multivariate normal distribution
    # Note: For speed in a loop, you might want to pre-compute Cholesky and use torch.randn
    mean = torch.zeros(params.dim_u, device=device)
    dist = torch.distributions.MultivariateNormal(mean, params.Sigma)
    
    # Sample (K, T)
    eps = dist.sample((params.K, params.T))
    return eps

def _is_collided_batch(x: torch.Tensor, obs_params: ObstacleParams) -> torch.Tensor:
    """
    Check if position x is in collision with any obstacle.
    x: (K, 2)
    obs_params.xyr: (N_obs, 3)
    Returns: (K,) boolean tensor
    """
    # Broadcast subtraction: (K, 1, 2) - (1, N_obs, 2)
    diff = x.unsqueeze(1) - obs_params.xyr[:, :2].unsqueeze(0)
    
    # Distances: (K, N_obs)
    dist = torch.norm(diff, dim=2)
    
    # Radii + safety: (1, N_obs)
    thresholds = obs_params.xyr[:, 2].unsqueeze(0) + obs_params.safe_distance
    
    # Check collisions: (K, N_obs)
    in_collision = dist <= thresholds
    
    # If collided with ANY obstacle: (K,)
    return torch.any(in_collision, dim=1)

def stage_cost_batch(x: torch.Tensor, u: torch.Tensor, params: MPPIParams) -> torch.Tensor:
    """
    x: (K, dim_x)
    Returns: (K,)
    """
    collided = _is_collided_batch(x[:, :2], params.obstacle_params)
    cost = torch.where(collided, 
                       torch.tensor(params.obstacle_params.weight, device=x.device), 
                       torch.tensor(0.0, device=x.device))
    return cost

def terminal_cost_batch(x: torch.Tensor, params: MPPIParams) -> torch.Tensor:
    """
    x: (K, dim_x)
    Returns: (K,)
    """
    return torch.zeros(x.shape[0], device=x.device)

def rollout_batch(params: MPPIParams, x0: torch.Tensor, U_prev: torch.Tensor, eps: torch.Tensor):
    """
    Simulates K trajectories in parallel.
    x0: (dim_x,)
    U_prev: (T, dim_u)
    eps: (K, T, dim_u)
    
    Returns: 
        S: (K,) Total cost
        V: (K, T, dim_u) Perturbed control sequence
        trajs: (K, T, dim_x) Resulting trajectories
    """
    K, T, _ = eps.shape
    device = x0.device
    
    # Initialize state for the whole batch: (K, dim_x)
    x = x0.unsqueeze(0).expand(K, -1).clone()
    
    S = torch.zeros(K, device=device)
    
    # We need to collect the trajectory and controls
    traj_list = []
    V_list = []
    
    # Expand U_prev to batch dim for easier indexing: (K, T, dim_u)
    U_expanded = U_prev.unsqueeze(0).expand(K, -1, -1)
    
    # Ensure boolean mask handles broadcasting: (K, 1)
    use_nominal = params.use_nominal.unsqueeze(1)
    
    for t in range(T):
        u_nom_t = U_expanded[:, t, :] # (K, dim_u)
        e_t = eps[:, t, :]            # (K, dim_u)
        
        # Mix controls based on 'use_nominal'
        # v_t = where(use_nominal, u_nom + e, e)
        v_t = torch.where(use_nominal, u_nom_t + e_t, e_t)
        
        # Clamp controls
        u_t = model.clamp(v_t, params.model_params)
        V_list.append(u_t)
        
        # Control cost contribution
        # cross = gamma * (u_nom @ Sigma_inv @ v_t)
        # We need batch-wise dot product. 
        # (K, 1, dim_u) @ (1, dim_u, dim_u) @ (K, dim_u, 1) -> (K, 1, 1) roughly
        term1 = torch.matmul(u_nom_t.unsqueeze(1), params.Sigma_inv) # (K, 1, dim_u)
        cross = params.gamma * torch.matmul(term1, v_t.unsqueeze(2)).squeeze() # (K,)
        
        step_c = stage_cost_batch(x, u_t, params)
        S = S + step_c + cross
        
        # Dynamics step
        x = model.step(x.t(), u_t, params.model_params).t() # Transpose logic due to step() exptecting dim 0 as components
        
        # Store state (usually MPPI stores x_{t+1}, sometimes x_t. JAX code stored x_{t+1})
        traj_list.append(x)

    # Add terminal cost
    S = S + terminal_cost_batch(x, params)
    
    # Stack lists into tensors
    V = torch.stack(V_list, dim=1)      # (K, T, dim_u)
    trajs = torch.stack(traj_list, dim=1) # (K, T, dim_x)
    
    return S, V, trajs

def shift_U(U: torch.Tensor) -> torch.Tensor:
    """
    Shifts U by 1 timestep.
    U: (T, dim_u)
    """
    U_next = torch.roll(U, shifts=-1, dims=0)
    # Copy the now-last element from the previous last element
    U_next[-1] = U[-1]
    return U_next

# @torch.compile(mode="reduce-overhead")
def mppi_step(
    params: MPPIParams,
    U_prev: torch.Tensor,
    x0: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: u0, U_next, trajs, opt_traj
    """
    device = x0.device
    
    # 1. Sample Noise
    eps = sample_epsilon(params, device) # (K, T, dim_u)
    
    # 2. Parallel Rollout
    S, V, trajs = rollout_batch(params, x0, U_prev, eps)
    
    # 3. Stein / Spatial calculations
    spatial_trajs = trajs[:, :, :2] # (K, T, 2)
    
    # Calculate median trajectory (T, 2)
    # torch.median returns (values, indices), we want values
    spatial_median = torch.median(spatial_trajs, dim=0).values 
    
    # Compute Gradient of Target Distribution at the Median Trajectory
    # Assumes stein_grad_traj handles (T, 2) inputs
    h_target = stein_grad_traj(spatial_median, params.stein) # (T, 2)
    
    # S_flow calculation
    # JAX: einsum('ktn,tn->k', spatial_trajs, h_target)
    S_flow = -torch.einsum('ktn,tn->k', spatial_trajs, h_target)
    S = S + params.stein.weight * S_flow
    
    # Log PDF calculation
    # We must iterate over time because Stein logpdf usually expects batch of particles (K, 2)
    # JAX used nested vmap. PyTorch logic:
    # Reshape (K, T, 2) -> (K*T, 2) to batch process, then reshape back
    K, T, _ = spatial_trajs.shape
    flat_spatial = spatial_trajs.reshape(-1, 2)
    
    # Assuming logpdf can handle (N, 2) or we use vmap
    # Using the 'logpdf' from previous turn which expects (2,) or (N,2) if vectorized manually
    # Let's map it safely:
    flat_logp = torch.vmap(lambda x: logpdf(x, params.stein))(flat_spatial)
    logp = flat_logp.reshape(K, T)
    
    S_pdf = -torch.sum(logp, dim=1) # (K,)
    
    # 4. Total Cost & Weighting
    S = S + params.stein.weight * S_flow + params.stein.weight_pdf * S_pdf
    
    rho = torch.min(S)
    w_unnorm = torch.exp(-(S - rho) / params.lam)
    w = w_unnorm / (torch.sum(w_unnorm) + 1e-12) # (K,)
    
    # 5. Update Control Sequence
    # delta = V - U_prev (broadcasting U_prev across K)
    delta = V - U_prev.unsqueeze(0)
    
    # w_eps = sum(w * delta)
    w_eps = torch.einsum("k,ktu->tu", w, delta)
    U = U_prev + w_eps
    
    # 6. Generate Optimal Trajectory (Single Rollout without noise)
    opt_traj_list = []
    x_opt = x0.clone()
    
    for t in range(T):
        u_opt = U[t]
        # step expects 1D tensors to be components, not (1, dim) usually, 
        # but our previous `step` definition used stack. 
        x_opt = model.step(x_opt, u_opt, params.model_params)
        opt_traj_list.append(x_opt)
        
    opt_traj = torch.stack(opt_traj_list) # (T, dim_x)
    
    u0 = U[0]
    U_next = shift_U(U)
    
    return u0, U_next, trajs, opt_traj

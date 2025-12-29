import torch
import torch.nn.functional as F
from torch.func import vmap, grad  # Requires PyTorch 2.0+
from dataclasses import dataclass

# Setup device agnostic helper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class SteinParams:
    # GMM in R^2
    means: torch.Tensor      # (M, 2)
    cov_inv: torch.Tensor    # (M, 2, 2)
    log_weights: torch.Tensor # (M,)
    log_norm: torch.Tensor    # (M,)

    # drift matrices
    D: torch.Tensor          # (2, 2)
    S: torch.Tensor          # (2, 2)
    gamma: float

    # kernel
    h: float

    # overall weight
    weight: float
    weight_pdf: float

def component_logpdf(x: torch.Tensor, p: SteinParams):
    """
    Computes log-probabilities for GMM components.
    
    Shapes:
    x: (..., dim_x)  -> can be (2,), (Batch, 2), (Batch, 6), (H, W, 2)
    p.means: (M, 2)
    p.cov_inv: (M, 2, 2)
    Output: (..., M)
    """
    # 1. Extract position (last dimension, first 2 elements)
    # This works for both (2,) and (Batch, 6) inputs.
    pos = x[..., :2] 
    
    # 2. Broadcast 'pos' against Mixture Components 'M'
    # p.means is (M, 2).
    # We unsqueeze dimension -2 to create a "slot" for the M dimension.
    # If pos is (2,), it becomes (1, 2) -> broadcasts with (M, 2) -> (M, 2)
    # If pos is (B, 2), it becomes (B, 1, 2) -> broadcasts with (M, 2) -> (B, M, 2)
    dz = pos.unsqueeze(-2) - p.means 
    
    # 3. Compute Mahalanobis distance using Einsum with Ellipsis
    # dz:      (..., M, 2)  [denoted as ...ki]
    # cov_inv: (M, 2, 2)    [denoted as kij]
    # Result:  (..., M)     [denoted as ...k]
    quad = torch.einsum("...ki, kij, ...kj -> ...k", dz, p.cov_inv, dz)
    
    return p.log_norm - 0.5 * quad


def logpdf(x: torch.Tensor, p: SteinParams):
    log_comps = p.log_weights + component_logpdf(x, p)
    # torch.logsumexp requires a dim argument
    return torch.logsumexp(log_comps, dim=-1)

def score_pdf(x: torch.Tensor, p: SteinParams):
    # JAX: jax.grad(logpdf, argnums=0)(x, p)
    # PyTorch: grad transform from torch.func
    # We use a lambda to close over 'p' so grad only sees 'x'
    return grad(lambda x_in: logpdf(x_in, p))(x)

def drift(x: torch.Tensor, p: SteinParams):
    s = score_pdf(x, p)
    # JAX: r = p.gamma * (p.S @ s.T).T
    # Since inputs are 1D vectors here, s is (2,). 
    # Matrix vector multiplication handles the shapes naturally.
    r = p.gamma * (p.S @ s) 
    return p.D @ s + r

def pdf(x: torch.Tensor, p: SteinParams):
    return torch.exp(logpdf(x, p))

def kernel(x: torch.Tensor, y: torch.Tensor, p: SteinParams):
    diff = x - y
    # dot product for 1D vectors
    return torch.exp(-torch.dot(diff, diff) / p.h)

def stein_grad_unit(x1: torch.Tensor, x2: torch.Tensor, p: SteinParams):
    """
    Computes contribution of particle x2 onto x1.
    """
    # JAX: ker_grad = jax.grad(kernel, argnums=0)(x2, x1, p)
    # Differentiate kernel w.r.t first argument (x2)
    ker_grad_fn = grad(lambda a: kernel(a, x1, p))
    ker_grad = ker_grad_fn(x2)
    
    val = kernel(x2, x1, p) * drift(x2, p) + ker_grad
    return val

def stein_grad_state(x: torch.Tensor, x_traj: torch.Tensor, p: SteinParams):
    """
    Compute the Stein gradient for a single particle x against all particles in x_traj.
    """
    # We vmap over x_traj (the second argument of stein_grad_unit)
    # x and p are fixed for this operation
    batched_unit = vmap(lambda xt: stein_grad_unit(x, xt, p))
    stein_grads = batched_unit(x_traj)
    
    return torch.mean(stein_grads, dim=0)

# def stein_grad_traj(x_traj: torch.Tensor, p: SteinParams):
#     """
#     Compute Stein gradient for every particle in x_traj against the whole set.
#     """
#     # We vmap over the first argument of stein_grad_state
#     # The second argument (x_traj) acts as the fixed 'population'
#     batched_state = vmap(lambda x: stein_grad_state(x, x_traj, p))
#     return batched_state(x_traj)

def stein_grad_traj(x_traj: torch.Tensor, p: SteinParams):
    """
    Vectorized computation of Stein gradients.
    x_traj: (T, 2)
    Returns: (T, 2)
    """
    T = x_traj.shape[0]
    
    # 1. Pre-compute drift for all particles (T, 2)
    #    (You might need to adjust 'drift' to handle (T, 2) input if it doesn't already)
    drift_vals = torch.vmap(lambda x: drift(x, p))(x_traj)
    
    # 2. Compute pairwise differences & Kernel Matrix
    # diff[i, j] = x_traj[j] - x_traj[i]
    x_i = x_traj.unsqueeze(1)  # (T, 1, 2)
    x_j = x_traj.unsqueeze(0)  # (1, T, 2)
    diff = x_j - x_i           # (T, T, 2)
    
    dist_sq = torch.sum(diff**2, dim=-1) # (T, T)
    K_mat = torch.exp(-dist_sq / p.h)    # (T, T)
    
    # 3. Term 1: Kernel * Drift
    # Average over j: sum_j ( K[i, j] * drift[j] ) / T
    term1 = torch.matmul(K_mat, drift_vals) / T
    
    # 4. Term 2: Grad of Kernel
    # grad_{x_j} k(x_j, x_i) = k_ij * (-2/h * (x_j - x_i))
    grad_k = K_mat.unsqueeze(-1) * (-2.0 / p.h * diff) # (T, T, 2)
    term2 = torch.mean(grad_k, dim=1) # Average over j
    
    return term1 + term2

import torch
from dataclasses import dataclass

# Set default device to cuda if available, otherwise cpu
# (This variable is for initialization of new tensors if needed outside functions)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class DoubleIntegratorParams:
    delta_t: float
    max_accel_lin_abs: float
    max_accel_ang_abs: float

def clamp(u: torch.Tensor, p: DoubleIntegratorParams) -> torch.Tensor:
    """
    Clamps the control input u.
    Supports input shapes: (3,) or (Batch, 3) or (Batch, Time, 3)
    """
    # Unbind splits the tensor along the last dimension.
    # u[..., 0] -> ax, u[..., 1] -> ay, u[..., 2] -> w
    ax, ay, w = u.unbind(dim=-1)

    ax_c = torch.clamp(ax, -p.max_accel_lin_abs, p.max_accel_lin_abs)
    ay_c = torch.clamp(ay, -p.max_accel_lin_abs, p.max_accel_lin_abs)
    w_c  = torch.clamp(w,  -p.max_accel_ang_abs, p.max_accel_ang_abs)

    # Stack back along the last dimension to preserve input shape
    return torch.stack([ax_c, ay_c, w_c], dim=-1)

def step(x: torch.Tensor, u: torch.Tensor, p: DoubleIntegratorParams) -> torch.Tensor:
    """
    x = [px, py, vx, vy, yaw, yaw_rate]
    u = [ax, ay, w]
    """
    u = clamp(u, p)
    px, py, vx, vy, yaw, yaw_rate = x
    ax, ay, w = u.unbind(dim=-1)
    dt = p.delta_t

    new_px = px + vx * dt + 0.5 * ax * dt**2
    new_py = py + vy * dt + 0.5 * ay * dt**2
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_yaw = yaw + yaw_rate * dt + 0.5 * w * dt**2
    new_yaw_rate = yaw_rate + w * dt

    # Return stacked tensor to maintain device and gradient history
    return torch.stack([new_px, new_py, new_vx, new_vy, new_yaw, new_yaw_rate])
    
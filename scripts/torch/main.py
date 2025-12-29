import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import time
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import previously translated modules
# Assumes the file structure is maintained or all classes are in the same scope
from mppi.torch.mppi_core import mppi_step, MPPIParams
from models.torch import double_integrator as model
from mppi.torch.stein import pdf, drift, SteinParams
from configs.torch import params_loader # Assumes you have the PyTorch version of this


# Global device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def closed_loop(params: MPPIParams, x0: torch.Tensor, U0: torch.Tensor, N: int):
    """
    Simulates the closed-loop control system for N steps.
    """
    x = x0.clone()
    U_prev = U0.clone()
    
    # Pre-allocate storage for history to avoid list appending overhead
    xs_list = []
    us_list = []
    
    # We only store the LAST step's trajectories for visualization to save memory,
    # or you can store all if you have enough VRAM/RAM.
    # For exact parity with JAX code which returned all, be careful with memory on GPU.
    # Here, I'll return lists which is standard PyTorch practice.
    trajs_all_list = [] 
    opt_trajs_all_list = []

    # No need for jax.jit; PyTorch runs eagerly (or you could use torch.compile)
    # mppi_step_compiled = torch.compile(mppi_step, mode="reduce-overhead")
    for i in range(N):
        # mppi_step returns: u0, U_next, trajs, opt_traj
        # Note: 'key' is removed as PyTorch manages RNG state globally
        u0, U_next, trajs, opt_traj = mppi_step(params, U_prev, x)
        
        # Step dynamics
        x_next = model.step(x, u0, params.model_params)
        
        # Store data
        xs_list.append(x)
        us_list.append(u0)
        # if i == N - 1:  # Store only the last step's trajectories
        trajs_all_list.append(trajs)
        opt_trajs_all_list.append(opt_traj)
        
        # Update state
        x = x_next
        U_prev = U_next

    # Stack results
    xs = torch.stack(xs_list)
    us = torch.stack(us_list)
    trajs_all = torch.stack(trajs_all_list)
    opt_trajs_all = torch.stack(opt_trajs_all_list)

    return xs, us, U_prev, trajs_all, opt_trajs_all

def setup_canvas(fig, ax, params):
    # Move limits to CPU for plotting
    x_limits = params.map_x_limits.cpu().numpy()
    y_limits = params.map_y_limits.cpu().numpy()
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("MPPI Double Integrator Benchmark (PyTorch)")
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig, ax

def visualize(params, xs=None, trajs_all=None, opt_trajs_all=None):
    # 1. Grid Generation (on CPU for plotting logic usually, or GPU then transfer)
    # Using numpy for grid generation to interface cleanly with matplotlib
    x_lim = params.map_x_limits.cpu().numpy()
    y_lim = params.map_y_limits.cpu().numpy()
    
    n_x = int((x_lim[1] - x_lim[0]) / params.resolution)
    n_y = int((y_lim[1] - y_lim[0]) / params.resolution)

    x_lin = np.linspace(x_lim[0], x_lim[1], n_x)
    y_lin = np.linspace(y_lim[0], y_lim[1], n_y)
    grids_x, grids_y = np.meshgrid(x_lin, y_lin)

    # Convert grid to tensor for PDF computation
    # Flatten: (N_points, 2)
    flat_grid = np.stack([grids_x.ravel(), grids_y.ravel()], axis=1)
    grid_tensor = torch.tensor(flat_grid, dtype=torch.float32, device=device)
    print("Grid tensor shape: ", grid_tensor.shape)

    # 2. Compute PDF and Drift Field
    # We can batch process the grid
    with torch.no_grad():
        # logpdf usually returns log values, so exp it for contour
        # Assuming we defined pdf = exp(logpdf) in stein.py
        pdf_vals = pdf(grid_tensor, params.stein)
        print("PDF vals shape: ", pdf_vals.shape)
        
        # Reshape back to grid
        pdf_grids_np = pdf_vals.reshape(grids_x.shape).cpu().numpy()

        # Optional: Compute drift for streamplot
        # fs = drift(grid_tensor, params.stein) # You might need vmap for drift if not vectorized natively
        # But 'drift' in previous steps was element-wise for vectors.
        # Let's skip streamplot for speed as it was commented out in original too.

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    fig, ax = setup_canvas(fig, ax, params)

    # Contour
    cs = ax.contourf(grids_x, grids_y, pdf_grids_np, cmap='Blues', alpha=0.8)

    # Obstacles
    if params.obstacle_params is not None:
        obs_cpu = params.obstacle_params.xyr.cpu().numpy()
        for obs in obs_cpu:
            # obs: [x, y, r]
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
            ax.add_artist(circle)

    # Trajectories (Move to CPU)
    if trajs_all is not None:
        # Plot only the last step's swarm for clarity
        last_step_trajs = trajs_all[-1].cpu().numpy() # (K, T, dim_x)
        # Plot spatial components (x, y)
        for i in range(last_step_trajs.shape[0]):
            ax.plot(last_step_trajs[i, :, 0], last_step_trajs[i, :, 1], color='gray', alpha=0.1)
    
    if opt_trajs_all is not None:
        last_opt_traj = opt_trajs_all[-1].cpu().numpy()
        ax.plot(last_opt_traj[:, 0], last_opt_traj[:, 1], color='green', alpha=1, linewidth=2, label="Optimal Plan")
    
    if xs is not None:
        xs_cpu = xs.cpu().numpy()
        ax.plot(xs_cpu[:, 0], xs_cpu[:, 1], label="Executed Path", color='k')
        ax.scatter(xs_cpu[-1, 0], xs_cpu[-1, 1], color='red', marker='x', label='End')
    
    ax.legend()
    plt.show()

def main(N=2000):
    # Load Params (returns tuple in PyTorch version)
    params, seed = params_loader.load_mppi_params("configs/mppi_params.yaml", device=device)
    
    print(f"Device: {device}")

    # Initialize State
    x0 = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, math.radians(45.0), 0.0],
        dtype=torch.float32,
        device=device
    )
    
    U_prev = torch.zeros((params.T, params.dim_u), dtype=torch.float32, device=device)

    # Warmup / Compilation (Optional in PyTorch, but good practice to run one step before timing)
    print("Warming up...")
    _ = mppi_step(params, U_prev, x0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    print("Running closed-loop simulation...")
    t0 = time.perf_counter()
    
    # Run Loop
    xs, us, U_final, trajs_all, opt_trajs_all = closed_loop(
        params,
        x0,
        U_prev,
        N=N
    )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    
    print(f"Simulation completed in {t1 - t0:.2f} seconds.")
    rate = N / (t1 - t0)
    print(f"Average control rate: {rate:.2f} Hz")
    print("Done.")

    visualize(params, xs, trajs_all, opt_trajs_all)

if __name__ == "__main__":
    # N = 5000
    main(N=5000)

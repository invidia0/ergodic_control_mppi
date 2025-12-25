import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
print(sys.path)

import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from mppi.mppi_controller import MPPIController
from utils.utility_functions import generate_obstacle_map
import tqdm

lims = [-10, 10, -10, 10] # x_min, x_max, y_min, y_max
delta_t = 0.01
WB = 0.1

# Sigma_u = jnp.diag(jnp.array([1**2, jnp.deg2rad(8.0)**2]))
Sigma_u = jnp.diag(jnp.array([1, 1]))
map_size = (20.0, 20.0)

obstacle_map, obstacle_points, obstacles, boundary_indices, boundary_points = generate_obstacle_map(num_obstacles=25, 
                                  map_size=map_size,
                                  obstacle_radius_range=(0.5, 1),
                                  seed=42)

grids_x, grids_y = jnp.meshgrid(
    jnp.linspace(lims[0], lims[1], 100),
    jnp.linspace(lims[2], lims[3], 100)
)



params = {
    "delta_t": delta_t,
    "WB": WB,
    "Sigma_u": Sigma_u,
    "max_steer_abs": jnp.deg2rad(75.0),
    "max_accel_abs": 3.0,
    "horizon_step_T": 100,
    "number_of_samples_K": 500,
    "param_exploration": 0.1,
    "param_lambda": 1,
    "param_alpha": 0.1,
    "param_flow_alpha": 20,
}

mppi = MPPIController(
    delta_t=params["delta_t"],
    max_steer_abs=params["max_steer_abs"],
    max_accel_abs=params["max_accel_abs"],
    horizon_step_T=params["horizon_step_T"],
    number_of_samples_K=params["number_of_samples_K"],
    param_exploration=params["param_exploration"],
    param_lambda=params["param_lambda"],
    param_alpha=params["param_alpha"],
    param_flow_alpha=params["param_flow_alpha"],
    sigma=params["Sigma_u"],
    stage_cost_weights = jnp.array([0.1, 0.1]),
    terminal_cost_weights = jnp.array([10., 10., 0.2, 0.2]),
    use_gpu=True,
    map_resolution=0.2,
    sensing_range=10.0,
    safety_distance=0.5,
    min_safety_distance=0.1,
    occupancy_points=jnp.array(obstacle_points),
    collision_cost=0.0,
    sensing_cost=0.0,
    oom_cost=0.1,
    seed=0,
)

# x0 = jnp.array([-2.5, 0.0, jnp.deg2rad(0.0), 4])  # initial state
# x0 = jnp.array([-2.5, 0.0, 0.0, 0.0, jnp.deg2rad(0.0), 0.0])  # initial state [x, y, vx, vy, yaw, omega]
x0 = jnp.array([-2.5, 0.0])  # initial state [x, y, vx, vy]

steps = 2000
plot_steps = 1

history = []
optimal_traj_segments = []
sampled_traj_segments = []

for k in tqdm.trange(steps):
    optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list, epsilon = mppi.calc_control_input(x0=x0)
    # store current state position (on host)
    history.append(np.array(x0[:2]))

    # only save full optimal trajectory every 'plot_steps' iterations
    if k % plot_steps == 0:
        optimal_traj_segments.append(np.array(optimal_traj[:, :2]))
        sampled_traj_segments.append(np.array(sampled_traj_list))

    # advance the state
    x0 = optimal_traj[1]

# convert to arrays once at the end
history = np.stack(history, axis=0)  # (steps, 2)
# New axis for segments
optimal_traj_plot = np.stack(optimal_traj_segments, axis=0)  # (num_segments, T, 2)
sampled_traj_stack = np.stack(sampled_traj_segments, axis=0)  # (num_segments, K, T, 4)

print("Simulation complete.")
print(history.shape)
print(optimal_traj_plot.shape)
print(sampled_traj_stack.shape)


# PLOTS
step_to_plot = steps -1

# Check availability
if step_to_plot % plot_steps != 0:
    raise ValueError(f"step_to_plot {step_to_plot} is not available; must be multiple of plot_steps {plot_steps}.")

optimal_traj_plot = optimal_traj_segments[step_to_plot // plot_steps]
sampled_traj_plot = sampled_traj_stack[step_to_plot // plot_steps]
print(sampled_traj_plot.shape)
history_plot = history[:step_to_plot+1]
pos = history_plot[-1]

grids = jnp.array([grids_x.ravel(), grids_y.ravel()]).T
pdf_grids = vmap(mppi.pdf)(grids).reshape(grids_x.shape)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal', 'box')
# ax.set_xlim(lims[0], lims[1])
# ax.set_ylim(lims[2], lims[3])
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.contourf(grids_x, grids_y, pdf_grids, cmap='Blues', levels=10, alpha=0.8, zorder=0)

for k in range(sampled_traj_plot.shape[0]):
    if k >= int((1.0 - params["param_exploration"]) * params["number_of_samples_K"]):
        continue
    ax.plot(sampled_traj_plot[k,:, 0], sampled_traj_plot[k,:, 1], color='gray', alpha=1, zorder=9, linewidth=1)

ax.plot(optimal_traj_plot[:,0], optimal_traj_plot[:,1], color='purple',linewidth=2, zorder=10, marker='o', markersize=3, markevery=5)
ax.plot(history_plot[:,0], history_plot[:,1], 'k', linewidth=1.5)
ax.scatter(pos[0], pos[1], color='purple', s=100, zorder=10)
ax.scatter(history_plot[0,0], history_plot[0,1], color='black', s=25, zorder=10)
plt.show()
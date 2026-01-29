import numpy as np
from utils import utilities
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, params, xs, agent_radius = 5):
        self.params = params
        self.steps = xs.shape[0]
        self.agent_radius = agent_radius
        # self.trajectory = np.zeros((self.steps, 2))
        self.trajectory = np.array(xs[:, 0:2])  # Store the trajectory positions
        self.dt = params.delta_t

        ## Initialize metrics
        self.x_min, self.x_max = params.map_x_limits
        self.y_min, self.y_max = params.map_y_limits
        self.res = params.resolution
        self.means = params.stein.means - np.array([self.x_min, self.y_min])
        cov_inv = params.stein.cov_inv
        log_weights = params.stein.log_weights
        self.covariances = np.linalg.inv(np.array(cov_inv))
        self.weights = np.exp(np.array(log_weights))
        x_grid, y_grid = np.meshgrid(
            np.arange(0, 2*self.x_max, self.res),
            np.arange(0, 2*self.y_max, self.res),
        )
        self.grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        self.sigma = 1 / self.agent_radius

        # density_map = utilities.gauss_pdf(grid, means[0], covariances[0]) * weights[0] + \
        #                 utilities.gauss_pdf(grid, means[1], covariances[1]) * weights[1] + \
        #                 utilities.gauss_pdf(grid, means[2], covariances[2]) * weights[2]
        # density_map = utilities.gmm_eval(self.grid, self.means, self.covariances, self.weights)
        density_map = np.ones_like(x_grid)
        # density_map = utilities.gauss_pdf(self.grid, self.means[0], self.covariances[0])
        # density_map = utilities.min_max_normalize(density_map)
        self.goal_density = utilities.normalize_mat(density_map).reshape(x_grid.shape)
        free_density = self.goal_density.copy()
        # fig, ax = plt.subplots()
        for obs in params.obstacle_params.xyr:
            # find occupied cells
            obs_center = np.array([obs[0] - self.x_min, obs[1] - self.y_min])
            adj_obs = obs_center / self.res
            x_id, y_id = int(adj_obs[0]), int(adj_obs[1])
            num_cells_range = int(obs[2] / self.res)
            free_density[y_id-num_cells_range : y_id+num_cells_range, x_id-num_cells_range : x_id+num_cells_range] = 0.0
            # circle = plt.Circle((obs_center[0], obs_center[1]), obs[2], color='red', alpha=0.5)
            # ax.add_artist(circle)

        # ax.contourf(x_grid, y_grid, free_density, cmap='Blues', alpha=0.8)
        self.goal_density = free_density
        self.coverage_block = utilities.agent_block(2, 1e-6, self.agent_radius)
        self.kernel_size = self.coverage_block.shape[0]
        self.coverage_density = np.zeros_like(self.goal_density)
        self.ergodic_metric = np.zeros(self.steps)

    def update_trajectory(self, t: int, position: jax.Array):
        self.trajectory[t, :] = np.array(position)

    def compute_ergodicity(self, step: int, position: jax.Array) -> np.array:
        t = step * self.dt
        coverage = np.zeros_like(self.goal_density)
        adj_pos = (position[:2] - np.array([self.x_min, self.y_min])) / self.res
        # adj_pos = (position[:2] - np.array([self.x_min, self.y_min]))
        x, y = int(adj_pos[0]), int(adj_pos[1])
        # x, y = position[0], position[1]

        row_indices, row_start_kernel, num_kernel_rows = utilities.clamp_kernel_1d(
            x, 0, int(2*self.x_max/self.res), self.kernel_size
        )
        col_indices, col_start_kernel, num_kernel_cols = utilities.clamp_kernel_1d(
            y, 0, int(2*self.y_max/self.res), self.kernel_size
        )

        self.coverage_density[row_indices, col_indices] += self.coverage_block[
            row_start_kernel : row_start_kernel + num_kernel_rows,
            col_start_kernel : col_start_kernel + num_kernel_cols,
        ] # Eq. 3 - Coverage density
        
        # coverage = utilities.normalize_mat(self.coverage_density / (t + 1e-12))
        coverage = utilities.normalize_mat(self.coverage_density)
        em_diff = np.linalg.norm(self.goal_density - coverage)
        self.ergodic_metric[step] = em_diff

    def get_ergodic_metric(self) -> np.array:
        for step, pos in enumerate(self.trajectory):
            self.compute_ergodicity(step, pos)
        
        return self.ergodic_metric
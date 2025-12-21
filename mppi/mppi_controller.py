import numpy as jnp
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import patches
from matplotlib.animation import ArtistAnimation

import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
# import numpy as jnp
import tqdm

from jax import jit, grad, vmap, jacfwd
from jax.scipy.stats import gaussian_kde as kde
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp
import jax
from functools import partial
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
except:
    gpu = cpu

jnp.set_printoptions(precision=4)

jax.config.update("jax_enable_x64", False)


import numpy as np
from mppi.dynamics import Bicycle, DoubleIntegrator, DoubleIntegratorNoYaw, SingleIntegrator


class MPPIController():
    def __init__(
            self,
            delta_t: float = 0.05,
            max_steer_abs: float = 0.523,
            max_accel_abs: float = 2.000,
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            param_flow_alpha: float = 1,
            sigma: jnp.ndarray = jnp.array([[0.5, 0.0], [0.0, 0.1]]),
            stage_cost_weights: jnp.ndarray = jnp.array([50.0, 50.0]),
            terminal_cost_weights: jnp.ndarray = jnp.array([50.0, 50.0, 1.0, 20.0]),
            v_ref: float = 3.0,
            wheel_base: float = 2.5,
            use_gpu: bool = True,
            map_resolution: float = 0.2,
            sensing_range: float = 2.0,
            safety_distance: float = 1.0,
            min_safety_distance: float = 0.1,
            occupancy_points: jnp.ndarray = None,
            collision_cost: float = 100.0,
            sensing_cost: float = 1.0,
            oom_cost: float = 10.0,
            seed: int = 0,
    ) -> None:
        devices = jax.devices("gpu" if use_gpu and jax.devices("gpu") else "cpu")
        self.device = devices[0]
        print(f"[INFO] Using device: {self.device}")
        with jax.default_device(self.device):
            self.dim_x = 4 
            self.dim_u = 2
            self.T = horizon_step_T
            self.K = number_of_samples_K
            self.param_exploration = param_exploration
            self.param_lambda = param_lambda
            self.param_alpha = param_alpha
            self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
            self.Sigma = sigma
            # self.Sigma = jnp.array([[0.5, 0.0, 0.0],
            #                         [0.0, 0.5, 0.0],
            #                         [0.0, 0.0, 0.1]])
            # self.Sigma = jnp.diag(jnp.array([1, 1, jnp.deg2rad(8.0)**2]))
            # self.Sigma_inv = jnp.linalg.inv(self.Sigma)
            self.Sigma_inv = jnp.linalg.cholesky(self.Sigma + 1e-8 * jnp.eye(self.dim_u))
            self.stage_cost_weights = stage_cost_weights
            self.terminal_cost_weights = terminal_cost_weights
            self.v_ref = v_ref
            self.flow_alpha = param_flow_alpha
            self.collision_cost = collision_cost
            self.sensing_cost = sensing_cost
            self.oom_cost = oom_cost

            # vehicle parameters
            self.delta_t = delta_t #[s]
            self.max_steer_abs = max_steer_abs # [rad]
            self.max_accel_abs = max_accel_abs # [m/s^2]
            self.wheel_base = wheel_base # [m]
            self.sensing_range = sensing_range  # [m]
            self.safety_distance = safety_distance  # [m] DESIRED safety distance
            self.min_safety_distance = min_safety_distance  # [m] MINIMUM safety distance, collision if below
            # Filter
            self.window_size = 10
            kernel_1d = jnp.ones((self.window_size,), dtype=jnp.float32) / self.window_size
            self.ma_kernel = jnp.tile(kernel_1d[None, None, :], (self.dim_u, 1, 1))  # (dim_u, 1, w)

            # Stochasticity params
            self.key = jax.random.PRNGKey(seed)

            self.u_prev = jnp.zeros((self.T, self.dim_u))

            # Safety checks
            if self.Sigma.shape[0] != self.Sigma.shape[1] \
                or self.Sigma.shape[0] != self.dim_u \
                or self.dim_u < 1:
                raise ValueError("[ERROR] Sigma must be (size_dim_u x size_dim_u).")
            
            # Map related

            self.occupancy_points = occupancy_points
            self.map_resolution = map_resolution
            self.map_size = jnp.array([self.occupancy_points.shape[0] * self.map_resolution,
                                       self.occupancy_points.shape[1] * self.map_resolution])
            self.map_x_limits = jnp.array([-self.map_size[0]/2, self.map_size[0]/2])
            self.map_y_limits = jnp.array([-self.map_size[1]/2, self.map_size[1]/2])

            # PDF params
            # ----------- replace with params -----------
            means = jnp.stack([
                jnp.array([-2.5, -3.8]),
                jnp.array([5.0, -2.5]),
                jnp.array([3.5, 3.5]),
            ])  # (3, 2)

            covs = jnp.stack([
                jnp.array([[4.0,  2.0],
                        [2.0,  4.0]]),
                jnp.array([[3.0, -1.5],
                        [-1.5, 3.0]]),
                jnp.array([[4.0,  0.0],
                        [0.0,  2.0]]),
            ])  # (3, 2, 2)

            weights = jnp.array([0.5, 0.3, 0.2])   # sum to 1
            # ----------------------------------------------
            
            self.means = means
            self.covs = covs
            self.weights = weights
            self.log_weights = jnp.log(self.weights + 1e-12)  # avoid log(0) just in case
            # Precompute inverses and normalization constants
            self.cov_inv = jnp.linalg.inv(self.covs)                       # (3, 2, 2)
            self.log_det = jnp.log(jnp.linalg.det(self.covs))              # (3,)
            d = 2
            self.log_norm = -0.5 * (d * jnp.log(2.0 * jnp.pi) + self.log_det)  # (3,)

            self.score_pdf = grad(self.log_pdf)
            sigma = 1
            self.D = 0.5 * (sigma**2) * jnp.eye(2)

            # Skew-symmetric matrix S for circulation r = gamma * S * score
            self.S = jnp.array([[0.0, -1.0],
                        [1.0,  0.0]])  # 90° rotation
        
            self.d_kernel = jax.grad(self.kernel, argnums=(0)) # This is needed for the Stein operator
            
            self.stein_grad = jax.jit(self.stein_grad, device=gpu)
            self.pdf = jax.jit(lambda x: jnp.exp(self.log_pdf(x)))

            # self.model = Bicycle(
            #     delta_t=self.delta_t,
            #     max_steer_abs=self.max_steer_abs,
            #     max_accel_abs=self.max_accel_abs,
            #     wheelbase=self.wheel_base,
            #     w_accel=self.stage_cost_weights[0],
            #     w_steer=self.stage_cost_weights[1],
            # )

            self.model = SingleIntegrator(
                delta_t=self.delta_t,
                max_vel_abs=self.max_accel_abs,
                w_vel=self.stage_cost_weights[0],
            )

            


    @partial(jax.jit, static_argnames=("self",))
    def _mppi_step(self, x0, U_prev, occupancy_points, key):
        self.current_state = x0

        # 1) sample epsilon
        key, subkey = jax.random.split(key)

        epsilon = jax.random.multivariate_normal(
            key=subkey,
            mean=jnp.zeros(self.dim_u),
            cov=self.Sigma,
            shape=(self.K, self.T),
        )  # (K, T, dim_u)

        # 2) rollouts
        num_exloration = int((1.0 - self.param_exploration) * self.K)
        use_nominal = jnp.arange(self.K) < num_exloration

        S_V = jax.vmap(self.single_rollout, in_axes=(None, None, 0, 0, None), out_axes=(0, 0, 0))
        S, V, trajs = S_V(x0, U_prev, epsilon, use_nominal, occupancy_points) # S: (K,); V: (K, T, dim_u); trajs: (K, T, dim_x)

        # 3) Flow matching: compute Stein gradient on spatial positions only
        spatial_trajs = trajs[:, :, :2]  # Extract (x, y) positions: (K, T, 2)
        spatial_mean = jnp.median(spatial_trajs, axis=0)  # (T, 2)
        h_target = self.stein_grad(spatial_mean, h=1.0)  # (T, 2) - Stein gradient for spatial positions

        # Align shapes: compare spatial_trajs (K, T, 2) with h_target (T, 2)
        # Deviation from target flow in spatial dimensions only
        S_flow = -jnp.einsum('ktn,tn->k', spatial_trajs, h_target)  # (K,)
        S = S + self.flow_alpha * S_flow

        # 3) update controls
        U = self.update_controls(U_prev, S, V)

        # 4) trajectories
        optimal_traj = self._optimal_trajectory(x0, U)

        # 5) shift control sequence
        U_prev_next = jnp.roll(U, shift=-1, axis=0).at[-1].set(U[-1])

        return U[0], U, optimal_traj, trajs, epsilon, key, U_prev_next


    def calc_control_input(self, x0: jnp.ndarray):
        # x0 should already be on the right device, or you can device_put here
        u0, U, optimal_traj, sampled_traj_list, epsilon, key_new, U_prev_next= \
            self._mppi_step(x0, self.u_prev, self.occupancy_points, self.key)

        # update mutable fields outside jit
        self.key = key_new
        self.u_prev = U_prev_next

        return u0, U, optimal_traj, sampled_traj_list, epsilon


    @partial(jax.jit, static_argnames=("self",))
    def single_rollout(self, 
                       x0: jnp.ndarray, 
                       U: jnp.ndarray, 
                       epsilon: jnp.ndarray,
                       use_nominal: bool,
                       occupancy_points: jnp.ndarray) -> jnp.ndarray:
        """
        Perform a single rollout of the system dynamics given initial state, 
        control sequence, and noise.
        :param x0: (dim_x,) Initial state
        :param U: (T_horizon, dim_u) Control sequence
        :param epsilon: (T_horizon, dim_u) Noise sequence
        :return: S: (,) Cumulative cost of the rollout
        """
        def step(carry, inputs):
            x, S = carry
            U_t, epsilon_t = inputs
            v_t = jnp.where(use_nominal, U_t + epsilon_t, epsilon_t)
            u_t_clamped = self.model._g(v_t)

            cross = self.param_gamma * (U_t @ self.Sigma_inv @ v_t)
            # cross = self.param_gamma * (U_t @ v_t)
            S = S + self._c(x, u_t_clamped, occupancy_points) + cross
            # S = S + self._c(x, u_t_clamped, occupancy_points) #+ cross

            x = self.model._F(x, u_t_clamped)
            return (x, S), (u_t_clamped, x)

        (x, S), (V, traj) = jax.lax.scan(step, 
                                 (x0, jnp.array(0.0)),
                                 (U, epsilon))

        # S = S + self._phi(x, V[-1])
        return S, V, traj


    @partial(jax.jit, static_argnames=("self",))
    def update_controls(self, U, S, V):
        rho = jnp.min(S)
        scaled = (S - rho) / self.param_lambda
        weights_unnorm = jnp.exp(-scaled)
        eta = jnp.sum(weights_unnorm)
        w = weights_unnorm / eta

        deltas = V - U[None, :, :] # (K, T, dim_u)
        w_eps = jnp.einsum('k,ktu->tu', w, deltas) # (T, dim_u)
        # w_eps = self.moving_average(w_eps)
        U_new = U + w_eps
        return U_new


    @partial(jax.jit, static_argnames=("self",))
    def _optimal_trajectory(self, x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """Compute optimal trajectory given initial state and control sequence."""
        def step(x, u):
            x_next = self.model._F(x, self.model._g(u))
            return x_next, x_next

        _, traj = jax.lax.scan(step, x0, U)
        return traj


    # def _g(self, v: jnp.ndarray) -> jnp.ndarray:
    #     """clamp input: u = [accel, steer] - JAX-compatible version"""
    #     v_clamped = jnp.array([
    #         jnp.clip(v[0], -self.max_accel_abs, self.max_accel_abs),
    #         jnp.clip(v[1], -self.max_steer_abs, self.max_steer_abs)
    #     ])
    #     return v_clamped


    @partial(jax.jit, static_argnames=("self",))
    def safety_costs(self, x_t: jnp.ndarray, occupancy_points: jnp.ndarray) -> float:
        """
        Safety cost for a single state x_t = [x, y, yaw, v].
        
        - Distance-based cost w.r.t. occupied cells (0.0 in occupancy_map)
        - Soft penalty when outside sensing range
        """
        x, y, yaw, v = x_t
        pos = jnp.array([x, y])
        

        # Compute all distances at once
        diffs = occupancy_points - pos  # (H, W, 2)
        dists = jnp.linalg.norm(diffs, axis=-1)  # (H, W)
        
        min_dist = jnp.min(dists)
        
        # Three-way cost: collision (min_dist < min_safety), 
        # safe (min_dist > safety), or transition zone
        below_min = min_dist <= self.min_safety_distance
        above_safe = min_dist >= self.safety_distance
        
        # Compute safety cost (only applies in transition zone)
        safety_cost = self.sensing_cost * ((min_dist - self.safety_distance) ** -1) ** 2
        
        # Select appropriate cost
        obstacle_cost = jnp.where(
            below_min,
            self.collision_cost,  # Collision
            jnp.where(above_safe, 0.0, safety_cost)  # Safe or transition
        )
        
        # # Only apply if obstacles exist
        # obstacle_cost = jnp.where(True, obstacle_cost, 0.0)
        
        # ---------- Sensing range penalty ----------
        dist_from_robot = jnp.linalg.norm(pos - self.current_state[:2])
        range_cost = jnp.where(
            dist_from_robot <= self.sensing_range,
            0.0,
            self.sensing_cost * (dist_from_robot - self.sensing_range) ** 2
        )
        
        return obstacle_cost + range_cost


    def _c(self, x_t: jnp.ndarray, u_t: jnp.ndarray, occupancy_points: jnp.ndarray) -> float:
        """
        Instantaneous running cost for a single state/control pair.
        """
        # accel, steer = u_t
        x, y, *_ = x_t              # unpack state: x, y must be first two elements

        # w_accel = self.stage_cost_weights[0]
        # w_steer = self.stage_cost_weights[1]

        # safety cost (from obstacles + sensing range)
        safety_cost = 0.0 # self.safety_costs(x_t, occupancy_points)
        # out-of-bounds penalty (JAX-friendly, no Python ifs)
        out_x_low  = x < self.map_x_limits[0]
        out_x_high = x > self.map_x_limits[1]
        out_y_low  = y < self.map_y_limits[0]
        out_y_high = y > self.map_y_limits[1]

        out_of_bounds = out_x_low | out_x_high | out_y_low | out_y_high
        out_of_bounds_cost = jnp.where(out_of_bounds, self.oom_cost, 0.0)

        return (
            self.model.stage_cost(x_t, u_t) +
            safety_cost +
            out_of_bounds_cost
        )


    @partial(jax.jit, static_argnames=("self",))
    def _phi(self, x_T, u_T: jnp.ndarray) -> float:
        """Terminal cost: penalize out-of-bounds states."""
        x, y, yaw, v = x_T
        
        # Check boundary violations using JAX-compatible comparisons
        out_x = (x < self.map_x_limits[0]) | (x > self.map_x_limits[1])
        out_y = (y < self.map_y_limits[0]) | (y > self.map_y_limits[1])
        
        # Compute cost: 100.0 for each violated dimension
        cost_x = jnp.where(out_x, self.oom_cost, 0.0)
        cost_y = jnp.where(out_y, self.oom_cost, 0.0)
        
        return 0.0


    # def _F(self, x_t: jnp.ndarray, v_t: jnp.ndarray) -> jnp.ndarray:
    #     """calculate next state of the vehicle"""
    #     # get previous state variables
    #     x, y, yaw, v = x_t
    #     accel, steer = v_t

    #     # prepare params
    #     l = self.wheel_base
    #     dt = self.delta_t

    #     # update state variables
    #     new_x = x + v * jnp.cos(yaw) * dt
    #     new_y = y + v * jnp.sin(yaw) * dt
    #     new_yaw = yaw + v / l * jnp.tan(steer) * dt
    #     new_v = v + accel * dt

    #     # return updated state
    #     x_t_plus_1 = jnp.array([new_x, new_y, new_yaw, new_v])
    #     return x_t_plus_1


    @partial(jax.jit, static_argnames=("self",))
    def moving_average(self, xx: jnp.ndarray) -> jnp.ndarray:
        T, dim = xx.shape
        xx_ncl = jnp.swapaxes(xx, 0, 1)[None, :, :]  # (1, dim, T)

        yy_ncl = jax.lax.conv_general_dilated(
            lhs=xx_ncl,
            rhs=self.ma_kernel,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NCL", "OIL", "NCL"),
            feature_group_count=dim,
        )
        yy = jnp.swapaxes(yy_ncl[0], 0, 1)
        return yy

    @partial(jax.jit, static_argnames=("self",))
    def component_logpdf(self, x):
        """
        x: (...,) at least 2D, we use x[:2]
        returns: (3,) log N_k(x)
        """
        z = x[:2] - self.means                # (3, 2)
        quad = jnp.einsum('ki,kij,kj->k', z, self.cov_inv, z)  # (3,)
        return self.log_norm - 0.5 * quad     # (3,)

    @partial(jax.jit, static_argnames=("self",))
    def log_pdf(self, x):
        """
        Stable log mixture density.
        """
        log_comps = self.component_logpdf(x)              # (3,)
        return logsumexp(self.log_weights + log_comps)    # scalar
    
    @partial(jax.jit, static_argnames=("self",))
    def drift(self, x, gamma=0.0):
        s = self.score_pdf(x)
        r = gamma * (self.S @ s.T).T  # apply S to each vector
        return self.D @ s + r
    
    # Define function to calculate Stein variational gradient (RBF kernel)
    @partial(jax.jit, static_argnames=("self",))
    def kernel(self, x1, x2, h):
        # same as in the pdf function, only evaluate the first two dimensions
        """
        Here h is the bandwidth parameter for the RBF kernel
        """
        return jnp.exp(-1.0 * jnp.sum(jnp.square(x1[:2]-x2[:2])) / h)




    @partial(jax.jit, static_argnames=("self",))
    def stein_grad_unit(self, x1, x2, h):
        val = self.kernel(x2, x1, h) * self.drift(x2, gamma=0.5) + self.d_kernel(x2, x1, h)
        return val


    @partial(jax.jit, static_argnames=("self",))
    def stein_grad_state(self, x, x_traj, h):
        """
        Here we compute the Stein gradient for a single particle x against all particles in x_traj
        and then average the results.
        The vmap has in_axes=(None, 0, None) because we want to keep x fixed and vary x_traj
        over its first axis (the particles).
        """
        vals = jax.vmap(self.stein_grad_unit, in_axes=(None, 0, None))(x, x_traj, h)
        return jnp.mean(vals, axis=0)

    @partial(jax.jit, static_argnames=("self",))
    def stein_grad(self, traj, h):
        """
        Here we vmap over all particles in traj to compute their Stein gradients.
        """
        return jax.vmap(self.stein_grad_state, in_axes=(0, None, None))(traj, traj, h)
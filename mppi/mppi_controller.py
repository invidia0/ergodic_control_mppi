
import dataclasses
import jax
import jax.numpy as jnp

cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
    print(f"[INFO] CUDA device found: {gpu}")
except:
    gpu = cpu
    print("[INFO] No CUDA device found, using CPU.")

jnp.set_printoptions(precision=4)

from .core import MPPIParams, mppi_step

_ESS_RATE = 0.05  # must match scripts/main.py


class MPPIController:
    def __init__(self, params: MPPIParams, seed: int = 0):
        self.params = params
        self.lam = float(params.lam)  # mutable; adapted each step
        self.key = jax.random.PRNGKey(seed)

        self.U_prev = jnp.zeros(
            (params.T, params.dim_u), dtype=jnp.float32
        )

    def reset(self):
        self.U_prev = jnp.zeros_like(self.U_prev)
        self.lam = float(self.params.lam)

    def step(self, x0):
        """
        One MPPI step with ESS-adaptive temperature.

        x0: (dim_x,)

        Returns:
            u0:             (dim_u,)    first control to apply
            trajs:          (K, T, 2)  sampled position trajectories
            opt_traj:       (T, dim_x) optimal (weighted) trajectory
            spatial_median: (T, 2)     median of sampled trajectories
                                       — Phase 2: broadcast to other robots
        """
        p = dataclasses.replace(self.params, lam=jnp.asarray(self.lam, dtype=jnp.float32))
        u0, U_next, key_next, trajs, opt_traj, spatial_median, w = mppi_step(
            p, self.U_prev, x0, self.key,
        )

        ess_frac = float(1.0 / (jnp.sum(w ** 2) * self.params.K))
        self.lam = float(jnp.clip(
            self.lam * jnp.exp(_ESS_RATE * (self.params.ess_target - ess_frac)),
            self.params.lam_min,
            self.params.lam_max,
        ))
        self.U_prev = U_next
        self.key = key_next

        return u0, trajs, opt_traj, spatial_median
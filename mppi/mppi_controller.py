
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

from .mppi_core import MPPIParams, mppi_step

class MPPIController:
    def __init__(self, params: MPPIParams, seed: int = 0):
        self.params = params
        self.key = jax.random.PRNGKey(seed)

        self.U_prev = jnp.zeros(
            (params.T, params.dim_u), dtype=jnp.float32
        )

    def reset(self):
        self.U_prev = jnp.zeros_like(self.U_prev)

    def step(self, x0):
        """
        One MPPI step.

        x0: (dim_x,)
        """
        self.key, _ = jax.random.split(self.key)

        out, self.key = mppi_step(
            self.params,
            self.U_prev,
            x0,
            self.key,
        )

        self.U_prev = out.U_prev

        return out
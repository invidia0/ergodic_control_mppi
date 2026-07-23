"""Batch-compatible six-state double-integrator dynamics."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DoubleIntegratorParams:
    """Integration timestep and absolute acceleration limits."""

    delta_t: float
    max_accel_lin_abs: float
    max_accel_ang_abs: float


def clamp(control: jax.Array, params: DoubleIntegratorParams) -> jax.Array:
    """Clamp controls with shape ``(..., 3)`` to configured limits."""
    limits = jnp.array(
        [params.max_accel_lin_abs, params.max_accel_lin_abs, params.max_accel_ang_abs],
        dtype=control.dtype,
    )
    return jnp.clip(control, -limits, limits)


def step(state: jax.Array, control: jax.Array, params: DoubleIntegratorParams) -> jax.Array:
    """Advance double-integrator states, supporting arbitrary leading batches.

    Args:
        state: State array with shape ``(..., 6)``.
        control: Acceleration array broadcastable to shape ``(..., 3)``.
        params: Dynamics parameters.

    Returns:
        Integrated states with shape ``(..., 6)``.
    """
    control = clamp(control, params)
    dt = params.delta_t
    position = state[..., :2] + state[..., 2:4] * dt + 0.5 * control[..., :2] * dt**2
    velocity = state[..., 2:4] + control[..., :2] * dt
    yaw = state[..., 4:5] + state[..., 5:6] * dt + 0.5 * control[..., 2:3] * dt**2
    yaw_rate = state[..., 5:6] + control[..., 2:3] * dt
    return jnp.concatenate((position, velocity, yaw, yaw_rate), axis=-1)

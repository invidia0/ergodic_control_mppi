"""Batch-compatible double-integrator dynamics.

State is ``[p(d), v(d), yaw, yaw_rate]`` with shape ``(..., 2d + 2)``. Control is
``[a(d), angular_acceleration]`` with shape ``(..., d + 1)``. Planar ``d = 2`` is
``(..., 6)`` / ``(..., 3)``.
"""

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
    """Clamp controls with shape ``(..., d + 1)``, ``d`` linear axes then yaw."""
    dim_u = control.shape[-1]
    if dim_u == 3:
        limits = jnp.array(
            [params.max_accel_lin_abs, params.max_accel_lin_abs, params.max_accel_ang_abs],
            dtype=control.dtype,
        )
    else:
        limits = jnp.array(
            [params.max_accel_lin_abs] * (dim_u - 1) + [params.max_accel_ang_abs],
            dtype=control.dtype,
        )
    return jnp.clip(control, -limits, limits)


def step(state: jax.Array, control: jax.Array, params: DoubleIntegratorParams) -> jax.Array:
    """Advance double-integrator states, supporting arbitrary leading batches.

    Args:
        state: State array with shape ``(..., 2d + 2)``.
        control: Acceleration array broadcastable to shape ``(..., d + 1)``.
        params: Dynamics parameters.

    Returns:
        Integrated states with shape ``(..., 2d + 2)``.
    """
    control = clamp(control, params)
    dt = params.delta_t
    if state.shape[-1] == 6:
        position = state[..., :2] + state[..., 2:4] * dt + 0.5 * control[..., :2] * dt**2
        velocity = state[..., 2:4] + control[..., :2] * dt
        yaw = state[..., 4:5] + state[..., 5:6] * dt + 0.5 * control[..., 2:3] * dt**2
        yaw_rate = state[..., 5:6] + control[..., 2:3] * dt
        return jnp.concatenate((position, velocity, yaw, yaw_rate), axis=-1)
    d = (state.shape[-1] - 2) // 2
    position = state[..., :d] + state[..., d:2 * d] * dt + 0.5 * control[..., :d] * dt**2
    velocity = state[..., d:2 * d] + control[..., :d] * dt
    yaw = state[..., 2 * d:2 * d + 1] + state[..., 2 * d + 1:] * dt + 0.5 * control[..., d:] * dt**2
    yaw_rate = state[..., 2 * d + 1:] + control[..., d:] * dt
    return jnp.concatenate((position, velocity, yaw, yaw_rate), axis=-1)

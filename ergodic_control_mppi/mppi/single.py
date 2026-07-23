"""Single-robot closed-loop MPPI orchestration."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import step
from ergodic_control_mppi.mppi.core import adapt_temperature, mppi_step
from ergodic_control_mppi.parameters import ControllerParams


class SingleRunResult(NamedTuple):
    """Single-robot path and final planning outputs."""

    path: jax.Array
    optimal_trajectory: jax.Array
    surrogate: jax.Array


def run_single(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
    steps: int,
) -> SingleRunResult:
    """Run a single-robot closed loop.

    Args:
        params: Controller parameters.
        initial_state: State with shape ``(6,)``.
        initial_controls: Zero or warm-start controls with shape ``(T, 3)``.
        key: JAX PRNG key.
        steps: Positive scan length; static under JIT.

    Returns:
        Executed path ``(N, 6)`` and final optimal/surrogate trajectories.
    """
    history = jnp.broadcast_to(initial_state[:2], (params.mppi.history_length, 2))
    temperature = jnp.asarray(params.mppi.temperature, dtype=jnp.float32)
    initial_optimal = jnp.broadcast_to(initial_state, (params.mppi.horizon, 6))
    initial_surrogate = jnp.broadcast_to(initial_state[:2], (params.mppi.horizon, 2))

    def scan_step(carry, _):
        state, controls, current_key, current_temperature, current_history, _, _ = carry
        result = mppi_step(
            params,
            controls,
            state,
            current_key,
            current_temperature,
            current_history,
        )
        next_state = step(state, result.control, params.model)
        next_history = jnp.concatenate((current_history[1:], state[None, :2]), axis=0)
        next_temperature = adapt_temperature(current_temperature, result.weights, params)
        next_carry = (
            next_state,
            result.controls,
            result.key,
            next_temperature,
            next_history,
            result.optimal_trajectory,
            result.surrogate,
        )
        return next_carry, next_state

    initial = (
        initial_state,
        initial_controls,
        key,
        temperature,
        history,
        initial_optimal,
        initial_surrogate,
    )
    (*_, final_optimal, final_surrogate), path = jax.lax.scan(
        scan_step, initial, xs=None, length=steps
    )
    return SingleRunResult(path, final_optimal, final_surrogate)

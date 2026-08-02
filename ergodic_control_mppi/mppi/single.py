"""Single-robot closed-loop MPPI orchestration."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import step
from ergodic_control_mppi.mppi.core import (
    MPPIStepResult,
    adapt_temperature,
    effective_sample_fraction,
    mppi_step,
)
from ergodic_control_mppi.parameters import ControllerParams


class SingleRunResult(NamedTuple):
    """Single-robot path and final planning outputs."""

    path: jax.Array
    optimal_trajectory: jax.Array
    surrogate: jax.Array
    ess_fraction: jax.Array
    temperature: jax.Array


class SingleControllerState(NamedTuple):
    """Everything the closed loop carries from one control step to the next.

    Attributes:
        state: Current state with shape ``(6,)``.
        controls: Warm-start control sequence with shape ``(T, 3)``.
        key: JAX PRNG key.
        temperature: Scalar ESS-adapted MPPI temperature.
        memory: Fading-memory positions with shape ``(P, 2)``, oldest first.
        step_index: Number of completed control steps.
    """

    state: jax.Array
    controls: jax.Array
    key: jax.Array
    temperature: jax.Array
    memory: jax.Array
    step_index: jax.Array


def initialize_single(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
) -> SingleControllerState:
    """Build the initial closed-loop carry.

    Args:
        params: Controller parameters.
        initial_state: State with shape ``(6,)``.
        initial_controls: Zero or warm-start controls with shape ``(T, 3)``.
        key: JAX PRNG key.

    Returns:
        The carry consumed by :func:`single_step`.
    """
    return SingleControllerState(
        state=initial_state,
        controls=initial_controls,
        key=key,
        temperature=jnp.asarray(params.mppi.temperature, dtype=jnp.float32),
        memory=jnp.broadcast_to(initial_state[:2], (params.mppi.memory_length, 2)),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def single_step(
    params: ControllerParams, carry: SingleControllerState
) -> tuple[SingleControllerState, MPPIStepResult]:
    """Advance the closed loop by one control step.

    The appended memory sample is the position the robot actually reaches. Online
    callers must therefore overwrite ``state`` with the measured observation before
    the next call, so the buffer keeps executed positions rather than predicted ones.

    Args:
        params: Controller parameters.
        carry: Current closed-loop carry.

    Returns:
        The next carry and the planning outputs of this step.
    """
    result = mppi_step(
        params,
        carry.controls,
        carry.state,
        carry.key,
        carry.temperature,
        carry.memory,
    )
    next_state = step(carry.state, result.control, params.model)
    next_carry = SingleControllerState(
        state=next_state,
        controls=result.controls,
        key=result.key,
        temperature=adapt_temperature(carry.temperature, result.weights, params),
        memory=jnp.concatenate((carry.memory[1:], next_state[None, :2]), axis=0),
        step_index=carry.step_index + 1,
    )
    return next_carry, result


def stationary_step(
    params: ControllerParams,
    carry: SingleControllerState,
    state: jax.Array,
) -> tuple[SingleControllerState, MPPIStepResult]:
    """Plan once while holding the executed state and fading memory stationary."""
    next_carry, result = single_step(params, carry)
    held = next_carry._replace(
        state=state,
        memory=jnp.broadcast_to(state[:2], next_carry.memory.shape),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )
    return held, result


def run_single(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
    steps: int,
    progress: bool = False,
    preflight_steps: int = 0,
) -> SingleRunResult:
    """Run a single-robot closed loop.

    Args:
        params: Controller parameters.
        initial_state: State with shape ``(6,)``.
        initial_controls: Zero or warm-start controls with shape ``(T, 3)``.
        key: JAX PRNG key.
        steps: Positive scan length; static under JIT.
        progress: Whether to print approximately one update per percent.
        preflight_steps: Stationary planning iterations retained before motion starts.

    Returns:
        Executed path, final planning outputs, and per-step ESS/temperature histories.
    """
    initial_optimal = jnp.broadcast_to(initial_state, (params.mppi.horizon, 6))
    initial_surrogate = jnp.broadcast_to(initial_state[:2], (params.mppi.horizon, 2))
    progress_interval = max(1, (steps + 99) // 100)

    def scan_step(carry, index):
        if progress:
            current = index + 1
            jax.lax.cond(
                (current % progress_interval == 0) | (current == steps),
                lambda _: jax.debug.print(
                    "Progress: {current}/{total} ({percent}%)",
                    current=current,
                    total=steps,
                    percent=current * 100 // steps,
                ),
                lambda _: None,
                operand=None,
            )
        controller, _, _ = carry
        next_controller, result = single_step(params, controller)
        next_carry = (next_controller, result.optimal_trajectory, result.surrogate)
        diagnostics = (
            next_controller.state,
            effective_sample_fraction(result.weights, params.mppi.samples),
            next_controller.temperature,
        )
        return next_carry, diagnostics

    controller = initialize_single(params, initial_state, initial_controls, key)

    def preflight_step(carry, _):
        held, _ = stationary_step(params, carry, initial_state)
        return held, None

    controller, _ = jax.lax.scan(
        preflight_step, controller, xs=None, length=preflight_steps
    )
    initial = (
        controller,
        initial_optimal,
        initial_surrogate,
    )
    (_, final_optimal, final_surrogate), diagnostics = jax.lax.scan(
        scan_step, initial, xs=jnp.arange(steps)
    )
    path, ess_fraction, temperature = diagnostics
    return SingleRunResult(
        path, final_optimal, final_surrogate, ess_fraction, temperature
    )

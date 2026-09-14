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
from ergodic_control_mppi.mppi.field import responsibilities
from ergodic_control_mppi.parameters import ControllerParams


class SingleRunResult(NamedTuple):
    """Single-robot path and final planning outputs."""

    path: jax.Array
    optimal_trajectory: jax.Array
    surrogate: jax.Array
    ess_fraction: jax.Array
    temperature: jax.Array


class SingleControllerState(NamedTuple):
    """Closed-loop carry from one control step to the next.

    Attributes:
        state: Current state, shape ``(2d + 2,)`` (planar ``(6,)``).
        controls: Warm-start controls, shape ``(T, d + 1)`` (planar ``(T, 3)``).
        key: JAX PRNG key.
        temperature: ESS-adapted MPPI temperature.
        memory: Fading-memory positions, shape ``(P, d)``.
        step_index: Completed control steps.
        service_mass: Per-component visit mass for the service gate, shape ``(J,)``.
    """

    state: jax.Array
    controls: jax.Array
    key: jax.Array
    temperature: jax.Array
    memory: jax.Array
    step_index: jax.Array
    service_mass: jax.Array


def initialize_single(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
) -> SingleControllerState:
    """Build the initial closed-loop carry."""
    d = params.gmm.means.shape[-1]
    return SingleControllerState(
        state=initial_state,
        controls=initial_controls,
        key=key,
        temperature=jnp.asarray(params.mppi.temperature, dtype=jnp.float32),
        memory=jnp.broadcast_to(initial_state[:d], (params.mppi.memory_length, d)),
        service_mass=responsibilities(initial_state[:d], params.gmm)
        / jnp.maximum(1.0 - params.field.service_decay, 1e-12),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def single_step(
    params: ControllerParams, carry: SingleControllerState
) -> tuple[SingleControllerState, MPPIStepResult]:
    """
    Advance the closed loop by one control step.
    
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
        carry.service_mass,
    )
    next_state = step(carry.state, result.control, params.model)
    d = params.gmm.means.shape[-1]
    next_carry = SingleControllerState(
        state=next_state,
        controls=result.controls,
        key=result.key,
        temperature=adapt_temperature(carry.temperature, result.weights, params),
        memory=jnp.concatenate((carry.memory[1:], next_state[None, :d]), axis=0),
        step_index=carry.step_index + 1,
        service_mass=params.field.service_decay * carry.service_mass
        + responsibilities(next_state[:d], params.gmm),
    )
    return next_carry, result


def measured_step(
    params: ControllerParams, carry: SingleControllerState, observation: jax.Array
) -> tuple[SingleControllerState, MPPIStepResult]:
    """
    Advance the closed loop from a measured state instead of the model's prediction.

    The newest fading-memory sample is overwritten too, so the buffer holds positions the
    vehicle actually reached.

    Args:
            params: Controller parameters.
            carry: Current closed-loop carry.
            observation: Measured state with shape ``(2d + 2,)``.
    Returns:
            The next carry and the planning outputs of this step.
    """
    d = params.gmm.means.shape[-1]
    corrected = carry._replace(
        state=observation, memory=carry.memory.at[-1].set(observation[:d])
    )
    return single_step(params, corrected)


def stationary_step(
    params: ControllerParams,
    carry: SingleControllerState,
    state: jax.Array,
) -> tuple[SingleControllerState, MPPIStepResult]:
    """Plan once while holding the executed state and fading memory stationary."""
    next_carry, result = single_step(params, carry)
    d = params.gmm.means.shape[-1]
    held = next_carry._replace(
        state=state,
        memory=jnp.broadcast_to(state[:d], next_carry.memory.shape),
        step_index=jnp.asarray(0, dtype=jnp.int32),
        service_mass=responsibilities(state[:d], params.gmm)
        / jnp.maximum(1.0 - params.field.service_decay, 1e-12),
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
    """
    Run a single-robot closed loop.
    
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
    d = params.gmm.means.shape[-1]
    initial_optimal = jnp.broadcast_to(initial_state, (params.mppi.horizon, initial_state.shape[-1]))
    initial_surrogate = jnp.broadcast_to(initial_state[:d], (params.mppi.horizon, d))
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


def stack_params(per_lane: list[ControllerParams]) -> ControllerParams:
    """
    Stack per-lane controller parameters into one batched pytree.
    
    Args:
            per_lane: One parameter set per lane; all must share the static signature.
    Returns:
            Parameters whose every leaf carries a leading lane axis.
    """
    if not per_lane:
        raise ValueError("stack_params needs at least one lane")
    reference = jax.tree_util.tree_structure(per_lane[0])
    for index, candidate in enumerate(per_lane[1:], start=1):
        if jax.tree_util.tree_structure(candidate) != reference:
            raise ValueError(
                f"lane {index} has a different static signature than lane 0; group arms by "
                "(samples, horizon, memory_length, smooth_window) before stacking"
            )
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *per_lane)


def run_batch(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    keys: jax.Array,
    steps: int,
    preflight_steps: int = 0,
) -> SingleRunResult:
    """
    Run one closed loop per lane, in a single fused scan.
    
    Args:
            params: Batched parameters from :func:`stack_params`, lane axis leading.
            initial_state: Shared start state with shape ``(6,)``.
            initial_controls: Shared warm start with shape ``(T, 3)``.
            keys: Per-lane PRNG keys, lane axis leading.
            steps: Positive scan length; static under JIT.
            preflight_steps: Stationary planning iterations retained before motion starts.
    Returns:
            : class:`SingleRunResult` with a leading lane axis on every field.
    """
    return jax.vmap(
        lambda lane_params, key: run_single(
            lane_params,
            initial_state,
            initial_controls,
            key,
            steps=steps,
            preflight_steps=preflight_steps,
        )
    )(params, keys)

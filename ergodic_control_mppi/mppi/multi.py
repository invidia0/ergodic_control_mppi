"""Synchronous decentralized multi-robot MPPI orchestration."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import step
from ergodic_control_mppi.mppi.core import adapt_temperature, mppi_step
from ergodic_control_mppi.parameters import ControllerParams


class MultiRunResult(NamedTuple):
    """Multi-robot paths and final shared planning outputs."""

    paths: jax.Array
    optimal_trajectories: jax.Array
    surrogates: jax.Array


def initialize_surrogates(initial_states: jax.Array, horizon: int) -> jax.Array:
    """Broadcast actual initial positions to shape ``(R, T, 2)``."""
    return jnp.broadcast_to(initial_states[:, None, :2], (initial_states.shape[0], horizon, 2))


def build_cross_particles(surrogates: jax.Array, histories: jax.Array) -> jax.Array:
    """Build each robot's synchronous other-robot particles.

    Args:
        surrogates: Shared predicted paths with shape ``(R, T, 2)``.
        histories: Executed positions with shape ``(R, H, 2)``.

    Returns:
        Particles with shape ``(R, (R-1) * (T+H), 2)``.
    """
    robots = surrogates.shape[0]
    all_particles = jnp.concatenate((surrogates, histories), axis=1)
    offsets = jnp.arange(1, robots)[None, :]
    other_indices = (jnp.arange(robots)[:, None] + offsets) % robots
    return all_particles[other_indices].reshape(robots, -1, 2)


def run_multi(
    params: ControllerParams,
    initial_states: jax.Array,
    initial_controls: jax.Array,
    keys: jax.Array,
    steps: int,
) -> MultiRunResult:
    """Run synchronous decentralized MPPI for a robot batch.

    Args:
        params: Shared controller parameters.
        initial_states: States with shape ``(R, 6)``.
        initial_controls: Controls with shape ``(R, T, 3)``.
        keys: Per-robot JAX keys with shape ``(R, 2)``.
        steps: Positive scan length; static under JIT.

    Returns:
        Executed paths ``(N, R, 6)`` and final planned/shared trajectories.
    """
    robots = initial_states.shape[0]
    histories = jnp.broadcast_to(
        initial_states[:, None, :2], (robots, params.mppi.history_length, 2)
    )
    surrogates = initialize_surrogates(initial_states, params.mppi.horizon)
    optimal = jnp.broadcast_to(
        initial_states[:, None, :], (robots, params.mppi.horizon, 6)
    )
    temperatures = jnp.full((robots,), params.mppi.temperature, dtype=jnp.float32)

    def scan_step(carry, _):
        states, controls, current_keys, current_temperatures, current_histories, current_surrogates, _ = carry
        cross_particles = build_cross_particles(current_surrogates, current_histories)
        results = jax.vmap(mppi_step, in_axes=(None, 0, 0, 0, 0, 0))(
            params,
            controls,
            states,
            current_keys,
            current_temperatures,
            cross_particles,
        )
        next_states = step(states, results.control, params.model)
        next_histories = jnp.concatenate(
            (current_histories[:, 1:], states[:, None, :2]), axis=1
        )
        next_temperatures = jax.vmap(adapt_temperature, in_axes=(0, 0, None))(
            current_temperatures, results.weights, params
        )
        next_carry = (
            next_states,
            results.controls,
            results.key,
            next_temperatures,
            next_histories,
            results.surrogate,
            results.optimal_trajectory,
        )
        return next_carry, next_states

    initial = (
        initial_states,
        initial_controls,
        keys,
        temperatures,
        histories,
        surrogates,
        optimal,
    )
    (*_, final_surrogates, final_optimal), paths = jax.lax.scan(
        scan_step, initial, xs=None, length=steps
    )
    return MultiRunResult(paths, final_optimal, final_surrogates)

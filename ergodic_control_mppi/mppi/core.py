"""Pure functional MPPI sampling, rollout scoring, and control update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import clamp, step
from ergodic_control_mppi.mppi.stein import stein_gradient, stein_repulsion
from ergodic_control_mppi.parameters import ControllerParams


class MPPIStepResult(NamedTuple):
    """Outputs of one MPPI update."""

    control: jax.Array
    controls: jax.Array
    key: jax.Array
    optimal_trajectory: jax.Array
    surrogate: jax.Array
    weights: jax.Array


def adapt_temperature(
    temperature: jax.Array, weights: jax.Array, params: ControllerParams
) -> jax.Array:
    """Adapt MPPI temperature toward the configured effective sample size."""
    ess_fraction = 1.0 / (jnp.sum(weights * weights) * params.mppi.samples)
    updated = temperature * jnp.exp(0.05 * (params.mppi.ess_target - ess_fraction))
    return jnp.clip(updated, params.mppi.temperature_min, params.mppi.temperature_max)


def sample_epsilon(key: jax.Array, params: ControllerParams) -> tuple[jax.Array, jax.Array]:
    """Sample controls with shape ``(K, T, 3)`` and return the advanced key."""
    key, sample_key = jax.random.split(key)
    epsilon = jax.random.multivariate_normal(
        sample_key,
        jnp.zeros((3,), dtype=jnp.float32),
        params.mppi.covariance,
        shape=(params.mppi.samples, params.mppi.horizon),
        dtype=jnp.float32,
    )
    return epsilon, key


def stage_cost(state: jax.Array, params: ControllerParams) -> jax.Array:
    """Return obstacle and workspace penalties for states with shape ``(..., 6)``."""
    position = state[..., :2]
    obstacles = params.workspace.obstacles
    distances = jnp.linalg.norm(position[..., None, :] - obstacles[:, :2], axis=-1)
    collision = jnp.any(
        distances <= obstacles[:, 2] + params.workspace.safe_distance,
        axis=-1,
    )
    outside = (
        (position[..., 0] < params.workspace.x_limits[0])
        | (position[..., 0] > params.workspace.x_limits[1])
        | (position[..., 1] < params.workspace.y_limits[0])
        | (position[..., 1] > params.workspace.y_limits[1])
    )
    return collision * params.workspace.obstacle_cost + outside * params.workspace.out_of_map_cost


def _rollouts(
    params: ControllerParams,
    state: jax.Array,
    previous_controls: jax.Array,
    epsilon: jax.Array,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Roll out all samples and return costs, controls, and positions."""
    samples = params.mppi.samples
    states = jnp.broadcast_to(state, (samples, 6))
    costs = jnp.zeros((samples,), dtype=jnp.float32)
    nominal_count = jnp.asarray((1.0 - params.mppi.exploration) * samples, dtype=jnp.int32)
    use_nominal = jnp.arange(samples) < nominal_count
    coefficient = temperature * (1.0 - params.mppi.alpha)

    def scan_step(carry, inputs):
        current_states, current_costs = carry
        nominal, noise = inputs
        raw_controls = jnp.where(use_nominal[:, None], nominal + noise, noise)
        controls = clamp(raw_controls, params.model)
        cross_cost = coefficient * jnp.einsum(
            "i,ij,kj->k", nominal, params.mppi.covariance_inverse, raw_controls
        )
        current_costs = current_costs + stage_cost(current_states, params) + cross_cost
        current_states = step(current_states, controls, params.model)
        return (current_states, current_costs), (controls, current_states[:, :2])

    (_, costs), (controls, positions) = jax.lax.scan(
        scan_step,
        (states, costs),
        (previous_controls, jnp.swapaxes(epsilon, 0, 1)),
    )
    return costs, jnp.swapaxes(controls, 0, 1), jnp.swapaxes(positions, 0, 1)


def mppi_step(
    params: ControllerParams,
    previous_controls: jax.Array,
    state: jax.Array,
    key: jax.Array,
    temperature: jax.Array,
    cross_particles: jax.Array,
) -> MPPIStepResult:
    """Compute one adaptive-temperature MPPI control update.

    Args:
        params: Shared immutable controller parameters.
        previous_controls: Warm-start controls with shape ``(T, 3)``.
        state: Current state with shape ``(6,)``.
        key: JAX PRNG key.
        temperature: Current positive MPPI temperature.
        cross_particles: History or other-robot positions with shape ``(P, 2)``.

    Returns:
        Named outputs containing the first control, shifted controls, advanced
        key, optimal state trajectory ``(T, 6)``, shared surrogate ``(T, 2)``,
        and normalized weights ``(K,)``.
    """
    epsilon, key = sample_epsilon(key, params)
    costs, sampled_controls, sampled_positions = _rollouts(
        params, state, previous_controls, epsilon, temperature
    )
    surrogate = jnp.median(sampled_positions, axis=0)
    differences = surrogate[:, None, :] - surrogate[None, :, :]
    bandwidth = jnp.maximum(
        jnp.median(jnp.sum(differences * differences, axis=-1)),
        params.stein.self_bandwidth,
    )
    target_flow = stein_gradient(surrogate, surrogate, params.gmm, params.stein, bandwidth)
    if cross_particles.shape[0]:
        target_flow += params.stein.repulsion_weight * stein_repulsion(
            surrogate, cross_particles, params.stein
        )
    costs += params.stein.flow_weight * -jnp.einsum(
        "kti,ti->k", sampled_positions, target_flow
    )
    shifted_costs = costs - jnp.min(costs)
    unnormalized = jnp.exp(-shifted_costs / temperature)
    weights = unnormalized / jnp.sum(unnormalized)
    controls = previous_controls + jnp.einsum(
        "k,kti->ti", weights, sampled_controls - previous_controls
    )

    def optimal_step(current, control):
        next_state = step(current, control, params.model)
        return next_state, next_state

    _, optimal_trajectory = jax.lax.scan(optimal_step, state, controls)
    shifted_controls = jnp.concatenate((controls[1:], controls[-1:]), axis=0)
    return MPPIStepResult(
        controls[0], shifted_controls, key, optimal_trajectory, surrogate, weights
    )

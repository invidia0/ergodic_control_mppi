"""Pure functional MPPI sampling, rollout scoring, and control update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import clamp, step
from ergodic_control_mppi.mppi.field import (
    attraction_target,
    kde_repulsion,
    memory_repulsion,
    scheduled_speed,
    score_pdf,
    service_ratio,
    service_ratio_from_mass,
)
from ergodic_control_mppi.parameters import ControllerParams


class MPPIStepResult(NamedTuple):
    """Outputs of one MPPI update."""

    control: jax.Array
    controls: jax.Array
    key: jax.Array
    optimal_trajectory: jax.Array
    surrogate: jax.Array
    weights: jax.Array


def effective_sample_fraction(weights: jax.Array, samples: int) -> jax.Array:
    """Return MPPI effective sample size divided by the rollout count."""
    return 1.0 / (jnp.sum(weights * weights) * samples)


def adapt_temperature(
    temperature: jax.Array, weights: jax.Array, params: ControllerParams
) -> jax.Array:
    """Adapt MPPI temperature toward the configured effective sample size."""
    ess_fraction = effective_sample_fraction(weights, params.mppi.samples)
    updated = temperature * jnp.exp(0.05 * (params.mppi.ess_target - ess_fraction))
    return jnp.clip(updated, params.mppi.temperature_min, params.mppi.temperature_max)


def sample_epsilon(key: jax.Array, params: ControllerParams) -> tuple[jax.Array, jax.Array]:
    """Sample controls with shape ``(K, T, d + 1)`` and return the advanced key."""
    dim_u = params.mppi.covariance.shape[0]
    key, sample_key = jax.random.split(key)
    epsilon = jax.random.multivariate_normal(
        sample_key,
        jnp.zeros((dim_u,), dtype=jnp.float32),
        params.mppi.covariance,
        shape=(params.mppi.samples, params.mppi.horizon),
        dtype=jnp.float32,
    )
    return epsilon, key


def _axis_limits(params: ControllerParams) -> jax.Array:
    """Stack ``(d, 2)`` workspace bounds from x, y, and extra axes."""
    return jnp.concatenate(
        (
            params.workspace.x_limits[None, :],
            params.workspace.y_limits[None, :],
            params.workspace.extra_limits,
        ),
        axis=0,
    )


def stage_cost(state: jax.Array, params: ControllerParams) -> jax.Array:
    """Return obstacle and workspace penalties for states with shape ``(..., 2d + 2)``."""
    d = params.gmm.means.shape[-1]
    position = state[..., :d]
    obstacles = params.workspace.obstacles
    distances = jnp.linalg.norm(position[..., None, :2] - obstacles[:, :2], axis=-1)
    collision = distances <= obstacles[:, 2] + params.workspace.safe_distance
    if obstacles.shape[-1] == 4:
        collision = collision & (
            position[..., 2:3] < obstacles[:, 3] + params.workspace.safe_distance
        )
    collision = jnp.any(collision, axis=-1)
    if d == 2:
        outside = (
            (position[..., 0] < params.workspace.x_limits[0])
            | (position[..., 0] > params.workspace.x_limits[1])
            | (position[..., 1] < params.workspace.y_limits[0])
            | (position[..., 1] > params.workspace.y_limits[1])
        )
        gap = jnp.minimum(
            jnp.minimum(position[..., 0] - params.workspace.x_limits[0],
                        params.workspace.x_limits[1] - position[..., 0]),
            jnp.minimum(position[..., 1] - params.workspace.y_limits[0],
                        params.workspace.y_limits[1] - position[..., 1]),
        )
    else:
        limits = _axis_limits(params)
        outside = jnp.any((position < limits[:, 0]) | (position > limits[:, 1]), axis=-1)
        gap = jnp.min(jnp.minimum(position - limits[:, 0], limits[:, 1] - position), axis=-1)
    encroach = jnp.maximum(params.workspace.boundary_margin - gap, 0.0)
    margin_cost = params.workspace.boundary_weight * encroach * encroach
    total = (
        collision * params.workspace.obstacle_cost
        + outside * params.workspace.out_of_map_cost
        + margin_cost
    )
    return total + _grid_cost(position, params)


def _grid_cost(position: jax.Array, params: ControllerParams) -> jax.Array:
    """Charge the runtime occupancy grid, if one was supplied."""
    grid = params.workspace.grid
    if not grid.size:
        return jnp.zeros(position.shape[:-1], dtype=jnp.float32)
    cell = jnp.floor(
        (position[..., : grid.ndim] - params.workspace.grid_origin)
        / params.workspace.grid_resolution
    ).astype(jnp.int32)
    if grid.ndim == 2:
        height, width = grid.shape
        column = jnp.clip(cell[..., 0], 0, width - 1)
        row = jnp.clip(cell[..., 1], 0, height - 1)
        occupied = grid[row, column]
    else:
        depth, height, width = grid.shape
        column = jnp.clip(cell[..., 0], 0, width - 1)
        row = jnp.clip(cell[..., 1], 0, height - 1)
        layer = jnp.clip(cell[..., 2], 0, depth - 1)
        occupied = grid[layer, row, column]
    return occupied * params.workspace.obstacle_cost


def _smooth(controls: jax.Array, window: int) -> jax.Array:
    """Moving-average the control sequence along the horizon."""
    if window <= 1:
        return controls
    kernel = jnp.ones(window, dtype=controls.dtype)
    # Dividing by the same kernel convolved with ones renormalises the truncated windows at
    # both ends exactly, for odd and even windows alike -- no special cases, no loop.
    norm = jnp.convolve(jnp.ones(controls.shape[0], controls.dtype), kernel, mode="same")
    smoothed = jax.vmap(
        lambda channel: jnp.convolve(channel, kernel, mode="same"), in_axes=1, out_axes=1
    )(controls)
    return smoothed / norm[:, None]


def _rollouts(
    params: ControllerParams,
    state: jax.Array,
    previous_controls: jax.Array,
    epsilon: jax.Array,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Roll out all samples and return costs, controls, and positions."""
    samples = params.mppi.samples
    d = params.gmm.means.shape[-1]
    states = jnp.broadcast_to(state, (samples, 6 if d == 2 else state.shape[-1]))
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
        return (current_states, current_costs), (controls, current_states[:, :2] if d == 2 else current_states[:, :d])

    (_, costs), (controls, positions) = jax.lax.scan(
        scan_step,
        (states, costs),
        (previous_controls, jnp.swapaxes(epsilon, 0, 1)),
    )
    return costs, jnp.swapaxes(controls, 0, 1), jnp.swapaxes(positions, 0, 1)


def _flow_tracking_cost(
    flow: jax.Array, displacements: jax.Array, time_step: float
) -> jax.Array:
    """
    Compute the displacement/velocity flow cost for each rollout.
    
    Args:
            flow: Reference field -grad Phi at rollout evaluation states, shape ``(K, T, 2)``.
            displacements: Consecutive position changes with shape ``(K, T, 2)``.
            time_step: Dynamics integration step.
    Returns:
            Flow-tracking costs with shape ``(K,)``.
    """
    alignment = -time_step * jnp.sum(flow * displacements, axis=-1)
    effort = 0.5 * jnp.sum(displacements * displacements, axis=-1)
    return jnp.sum(alignment + effort, axis=-1)


_tracking_cost = _flow_tracking_cost


def field_at(
    params: ControllerParams,
    queries: jax.Array,
    plan: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the gauged reference field at ``queries``, shape ``(Q, d)``."""
    field = params.field
    flow = score_pdf(queries, attraction_target(params.gmm, field, service_mass))
    ages = jnp.arange(memory.shape[0])[::-1]
    recency = field.memory_decay ** ages
    spans = _axis_limits(params)
    if params.workspace.extra_limits.shape[0] == 0:
        density_floor = 1.0 / (
            (params.workspace.x_limits[1] - params.workspace.x_limits[0])
            * (params.workspace.y_limits[1] - params.workspace.y_limits[0])
        )
    else:
        density_floor = 1.0 / jnp.prod(spans[:, 1] - spans[:, 0])
    flow += field.memory_gain * memory_repulsion(
        queries, memory, recency, params.gmm, field, density_floor
    )
    flow += field.plan_gain * jnp.sqrt(
        0.5 * jnp.e * field.fine_bandwidth
    ) * kde_repulsion(
        queries, plan, jnp.ones((plan.shape[0],), dtype=flow.dtype), field.fine_bandwidth
    )
    speed = field.reference_speed
    sigma = (service_ratio(memory, recency, params.gmm) if service_mass is None
             else service_ratio_from_mass(service_mass, memory[-1], params.gmm))
    local = scheduled_speed(queries, params.gmm, field, sigma)
    norm = jnp.linalg.norm(flow, axis=-1, keepdims=True)
    return jnp.where(speed > 0, local[:, None] * flow / jnp.maximum(norm, 1e-3), flow)


def reference_flow(
    params: ControllerParams,
    evaluation_positions: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """Return the median-source reference field, shape ``(T, d)``."""
    source_particles = jnp.median(evaluation_positions, axis=0)
    return field_at(params, source_particles, source_particles, memory, service_mass)


reference_velocity = reference_flow


def mppi_step(
    params: ControllerParams,
    previous_controls: jax.Array,
    state: jax.Array,
    key: jax.Array,
    temperature: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
) -> MPPIStepResult:
    """Compute one adaptive-temperature MPPI control update."""
    d = params.gmm.means.shape[-1]
    epsilon, key = sample_epsilon(key, params)
    costs, sampled_controls, sampled_positions = _rollouts(
        params, state, previous_controls, epsilon, temperature
    )
    surrogate = jnp.median(sampled_positions, axis=0)
    initial_positions = jnp.broadcast_to(
        state[:2] if d == 2 else state[:d], (params.mppi.samples, 1, 2 if d == 2 else d)
    )
    evaluation_positions = jnp.concatenate(
        (initial_positions, sampled_positions[:, :-1]), axis=1
    )
    displacements = sampled_positions - evaluation_positions
    target_flow = reference_flow(params, evaluation_positions, memory, service_mass)
    costs += params.field.track_weight * _flow_tracking_cost(
        target_flow[None], displacements, params.model.delta_t
    )
    shifted_costs = costs - jnp.min(costs)
    unnormalized = jnp.exp(-shifted_costs / temperature)
    weights = unnormalized / jnp.sum(unnormalized)
    controls = previous_controls + jnp.einsum(
        "k,kti->ti", weights, sampled_controls - previous_controls
    )
    controls = _smooth(controls, params.mppi.smooth_window)

    def optimal_step(current, control):
        next_state = step(current, control, params.model)
        return next_state, next_state

    _, optimal_trajectory = jax.lax.scan(optimal_step, state, controls)
    shifted_controls = jnp.concatenate((controls[1:], controls[-1:]), axis=0)
    return MPPIStepResult(
        controls[0], shifted_controls, key, optimal_trajectory, surrogate, weights
    )

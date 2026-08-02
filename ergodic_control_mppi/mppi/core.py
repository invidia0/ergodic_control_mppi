"""Pure functional MPPI sampling, rollout scoring, and control update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import clamp, step
from ergodic_control_mppi.mppi.stein import multiscale_memory_flow, stein_gradient
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
    # Soft inward margin: penalize hugging the *inside* of the map edge so the
    # flow can't park the robot in a corner (a corner is otherwise cost-free).
    gap = jnp.minimum(
        jnp.minimum(position[..., 0] - params.workspace.x_limits[0],
                    params.workspace.x_limits[1] - position[..., 0]),
        jnp.minimum(position[..., 1] - params.workspace.y_limits[0],
                    params.workspace.y_limits[1] - position[..., 1]),
    )
    encroach = jnp.maximum(params.workspace.boundary_margin - gap, 0.0)
    margin_cost = params.workspace.boundary_weight * encroach * encroach
    total = (
        collision * params.workspace.obstacle_cost
        + outside * params.workspace.out_of_map_cost
        + margin_cost
    )
    return total + _grid_cost(position, params)


def _grid_cost(position: jax.Array, params: ControllerParams) -> jax.Array:
    """Charge the runtime occupancy grid, if one was supplied.

    The grid shape is static under JIT, so an empty grid compiles this away and leaves
    the circular-obstacle cost byte-identical. Nearest-cell lookup is sufficient because
    rollout samples are spaced ``speed * delta_t`` apart, well under one cell; positions
    outside the grid are clamped, since the grid covers the workspace and leaving it is
    already charged by the out-of-map term.
    """
    grid = params.workspace.grid
    if not grid.size:
        return jnp.zeros(position.shape[:-1], dtype=jnp.float32)
    height, width = grid.shape
    cell = jnp.floor(
        (position - params.workspace.grid_origin) / params.workspace.grid_resolution
    ).astype(jnp.int32)
    column = jnp.clip(cell[..., 0], 0, width - 1)
    row = jnp.clip(cell[..., 1], 0, height - 1)
    return grid[row, column] * params.workspace.obstacle_cost


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


def _flow_tracking_cost(
    flow: jax.Array, displacements: jax.Array, time_step: float
) -> jax.Array:
    """Compute the displacement/velocity flow cost for each rollout.

    Args:
        flow: Stein flow at rollout evaluation states with shape ``(K, T, 2)``.
        displacements: Consecutive position changes with shape ``(K, T, 2)``.
        time_step: Dynamics integration step.

    Returns:
        Flow-tracking costs with shape ``(K,)``.
    """
    alignment = -time_step * jnp.sum(flow * displacements, axis=-1)
    effort = 0.5 * jnp.sum(displacements * displacements, axis=-1)
    return jnp.sum(alignment + effort, axis=-1)


def mppi_step(
    params: ControllerParams,
    previous_controls: jax.Array,
    state: jax.Array,
    key: jax.Array,
    temperature: jax.Array,
    memory: jax.Array,
) -> MPPIStepResult:
    """Compute one adaptive-temperature MPPI control update.

    Args:
        params: Shared immutable controller parameters.
        previous_controls: Warm-start controls with shape ``(T, 3)``.
        state: Current state with shape ``(6,)``.
        key: JAX PRNG key.
        temperature: Current positive MPPI temperature.
        memory: Executed-position ring buffer with shape ``(memory_length, 2)``;
            oldest first. The reference flow is repelled from a recency- and
            density-weighted (fading) memory of it to drive ergodic coverage.

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
    initial_positions = jnp.broadcast_to(
        state[:2], (params.mppi.samples, 1, 2)
    )
    evaluation_positions = jnp.concatenate(
        (initial_positions, sampled_positions[:, :-1]), axis=1
    )
    displacements = sampled_positions - evaluation_positions
    # Broadcast median-source field: ~0.995 rank-correlated with the faithful per-rollout
    # eq.-25 cost in the warm-started regime; source choice barely matters. Only a genuine
    # per-step ensemble split (rollouts branching) would need per-cluster representatives.
    source_particles = jnp.median(evaluation_positions, axis=0)
    differences = source_particles[:, None, :] - source_particles[None, :, :]
    bandwidth = jnp.maximum(
        jnp.median(jnp.sum(differences * differences, axis=-1)),
        params.stein.self_bandwidth,
    )
    target_flow = stein_gradient(
        source_particles, source_particles, params.gmm, params.stein, bandwidth
    )
    # Multiscale over-coverage feedback from the fading memory: one scale bank, two
    # knobs (memory_gain, memory_balance). The relative excess is stabilized by the
    # workspace-uniform density, so memory points dropped in transit corridors
    # (where p* ~ 0) cannot take over the normalized excess field.
    ages = jnp.arange(memory.shape[0])[::-1]  # 0 = newest (buffer is oldest-first)
    recency = params.stein.memory_decay ** ages
    workspace = params.workspace
    density_floor = 1.0 / (
        (workspace.x_limits[1] - workspace.x_limits[0])
        * (workspace.y_limits[1] - workspace.y_limits[0])
    )
    target_flow += params.stein.memory_gain * multiscale_memory_flow(
        source_particles, memory, recency, params.gmm, params.stein, density_floor
    )
    # Optional constant-speed tracking: follow the flow *direction* at a fixed
    # reference speed (LQR-flow-matching style). reference_speed=0 keeps the raw
    # velocity-magnitude cost.
    #
    # Reparameterization notes (why the tuning space is smaller than the knob count):
    #  - Magnitude gauge: when reference_speed>0 this normalization discards |target_flow|,
    #    so only the DIRECTION of (attraction + memory feedback) matters. memory_gain is
    #    therefore a RATIO to the attraction (implicit weight 1), not an absolute magnitude;
    #    the overall flow scale is a free gauge.
    #  - Horizon reach L = reference_speed * T * dt is the geometric span of one plan; reference_speed
    #    and the MPPI horizon T are partially redundant through it (tune one against the reported T).
    speed = params.stein.reference_speed
    norm = jnp.linalg.norm(target_flow, axis=-1, keepdims=True)
    target_flow = jnp.where(
        speed > 0, speed * target_flow / jnp.maximum(norm, 1e-3), target_flow
    )
    costs += params.stein.flow_weight * _flow_tracking_cost(
        target_flow[None], displacements, params.model.delta_t
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

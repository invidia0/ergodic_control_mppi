"""Pure functional MPPI sampling, rollout scoring, and control update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import clamp, step
from ergodic_control_mppi.mppi.field import (
    attraction_target,
    kde_repulsion,
    memory_flow,
    pdf,
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


def _smooth(controls: jax.Array, window: int) -> jax.Array:
    """Moving-average the control sequence along the horizon.

    Post-hoc smoothing of the weighted update, as the original MPPI paper does with a
    Savitzky-Golay filter. It trades a broken optimality reading -- the returned sequence is
    no longer the argmin of anything -- for lower variance in a plan built from a weighted
    average of noisy rollouts. ``window <= 1`` returns the input untouched, and since the
    field is static that branch is resolved at trace time and costs nothing.

    If smoothness is wanted for its own sake, colored sampling noise buys it *inside* the
    optimization instead of after it; this exists to measure whether the variance is worth
    attacking at all.
    """
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


def field_at(
    params: ControllerParams,
    queries: jax.Array,
    plan: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the gauged reference field ``Gamma_v(grad Phi)`` at ``queries``.

    ``Phi`` is written down in :func:`ergodic_control_mppi.mppi.field.potential`; the three
    gradients are taken here in closed form. Split out from :func:`reference_flow` with the
    query set and the plan set as explicit arguments so
    :mod:`ergodic_control_mppi.experiments.surrogate_fidelity` can evaluate the *same*
    expression per rollout instead of transcribing it -- a transcription is exactly what
    drifts, and this loop amplifies float-level differences into metres.

    Args:
        params: Shared immutable controller parameters.
        queries: Positions to evaluate at, shape ``(Q, 2)``.
        plan: The point set the plan repels itself from, shape ``(Z, 2)``.
        memory: Executed-position ring buffer with shape ``(P, 2)``; oldest first.
        service_mass: Per-component visit mass, shape ``(J,)``, or ``None`` to read the
            service ratio out of the trail instead.

    Returns:
        The reference flow at ``queries``, shape ``(Q, 2)``.
    """
    field = params.field
    # Score attraction: the bare analytic grad log p* at the query, of the mixture the
    # destination bias may have bent toward under-served modes. No kernel weighting and no
    # estimated bandwidth -- `h` is a design constant, not a median heuristic.
    flow = score_pdf(queries, attraction_target(params.gmm, field, service_mass))
    # Over-coverage feedback from the fading memory. The relative excess is stabilized by
    # the workspace-uniform density, so memory points dropped in transit corridors (where
    # p* ~ 0) cannot take over the normalized excess field.
    ages = jnp.arange(memory.shape[0])[::-1]  # 0 = newest (buffer is oldest-first)
    recency = field.memory_decay ** ages
    workspace = params.workspace
    density_floor = 1.0 / (
        (workspace.x_limits[1] - workspace.x_limits[0])
        * (workspace.y_limits[1] - workspace.y_limits[0])
    )
    flow += field.memory_gain * memory_flow(
        queries, memory, recency, params.gmm, field, density_floor
    )
    # Plan self-repulsion -- what the fading memory cannot supply. `memory` holds the
    # executed trail, so it can only repel from the past, and no weighting of it makes the
    # *plan* space-filling: avoiding your own wake still admits a compact repeated circuit.
    # Repelling the horizon points from each other is what fills a basin. Same
    # `sqrt(he/2)` gauge as the memory term, so `plan_gain` is commensurate with
    # `memory_gain`; measured at 1.3% of a step.
    flow += field.plan_gain * jnp.sqrt(
        0.5 * jnp.e * field.fine_bandwidth
    ) * kde_repulsion(
        queries, plan, jnp.ones((plan.shape[0],), dtype=flow.dtype), field.fine_bandwidth
    )
    # Constant-speed tracking: follow the field *direction* at a scheduled speed
    # (LQR-flow-matching style). reference_speed=0 keeps the raw velocity-magnitude cost.
    #
    # Reparameterization notes (why the tuning space is smaller than the knob count):
    #  - Magnitude gauge: when reference_speed>0 this normalization discards |grad Phi|,
    #    so only its DIRECTION matters. memory_gain and plan_gain are therefore RATIOS to
    #    the attraction (implicit weight 1), not absolute magnitudes; the overall scale of
    #    Phi is a free gauge.
    #  - Horizon reach L = reference_speed * T * dt is the geometric span of one plan;
    #    reference_speed and the MPPI horizon T are partially redundant through it.
    #
    # Density-scheduled speed. `in_mode_fraction` is a *time* ratio, so at roughly uniform
    # speed it is the share of arclength laid inside the modes, and the gaps between them
    # are overhead paid at whatever speed the vehicle crosses them. Scaling speed as 1/p* is
    # the classical condition for a path's time density to match the target, and it costs
    # nothing where the target already wants the time: at `transit_speedup = beta` the
    # vehicle keeps `reference_speed` at a mode peak and reaches `beta x` it where p*
    # vanishes. `dwell_slowdown` is the other half -- slowing the modes lengthens the
    # payload; only the pair reallocates time rather than also raising the mean speed.
    speed = field.reference_speed
    peak = jnp.max(pdf(params.gmm.means, params.gmm))
    share = jnp.clip(pdf(queries, params.gmm) / jnp.maximum(peak, 1e-12), 0.0, 1.0)
    # Service gate. With a static schedule the vehicle is slow wherever p* is high, which
    # buys dense filling but never releases: measured, every arm that dwells well has zero
    # tours. The release has to come from the buffer, not from position. sigma > 1 means the
    # mode the vehicle is in has already had more than its target share of recent path, so
    # `hold` decays from 1 to 0 and the in-mode speed climbs to the transit speed -- the
    # vehicle leaves under the same schedule that made it stay. service_floor = 0 disables
    # the gate exactly (hold == 1).
    floor = field.service_floor
    sigma = (service_ratio(memory, recency, params.gmm) if service_mass is None
             else service_ratio_from_mass(service_mass, memory[-1], params.gmm))
    served = jnp.maximum(sigma - 1.0, 0.0)
    release = jnp.where(floor > 0, served / (served + jnp.maximum(floor, 1e-12)), 0.0)
    hold = 1.0 - release
    local = speed * jnp.power(field.transit_speedup, 1.0 - share * hold) / jnp.power(
        field.dwell_slowdown, share * hold
    )
    norm = jnp.linalg.norm(flow, axis=-1, keepdims=True)
    return jnp.where(speed > 0, local[:, None] * flow / jnp.maximum(norm, 1e-3), flow)


def reference_flow(
    params: ControllerParams,
    evaluation_positions: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """Return the reference velocity field on the shared surrogate path.

    The median path over rollouts serves as both the query set and the plan the plan term
    repels from, and the resulting ``(T, 2)`` field is broadcast to every rollout. That
    compression is the paper's ``eps_comp``; :mod:`surrogate_fidelity` measures what it
    costs by calling :func:`field_at` per rollout instead.

    Args:
        params: Shared immutable controller parameters.
        evaluation_positions: Positions the rollout increments start from, shape ``(K, T, 2)``.
        memory: Executed-position ring buffer with shape ``(P, 2)``; oldest first.
        service_mass: Per-component visit mass, shape ``(J,)``, or ``None``.

    Returns:
        The reference flow at the ``T`` surrogate source particles, shape ``(T, 2)``.
    """
    # Broadcast median-source field: ~0.995 rank-correlated with the faithful per-rollout
    # cost in the warm-started regime; source choice barely matters. Only a genuine
    # per-step ensemble split (rollouts branching) would need per-cluster representatives.
    source_particles = jnp.median(evaluation_positions, axis=0)
    return field_at(params, source_particles, source_particles, memory, service_mass)


def mppi_step(
    params: ControllerParams,
    previous_controls: jax.Array,
    state: jax.Array,
    key: jax.Array,
    temperature: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
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
        service_mass: Recency-weighted per-component visit mass with shape ``(J,)``,
            or ``None`` to read the service ratio out of the trail instead.

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

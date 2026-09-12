"""
Measure every term the closed-loop analysis names, on the loop it analyses.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.models.double_integrator import step
from ergodic_control_mppi.mppi.core import (
    _rollouts,
    mppi_step,
    reference_flow,
    sample_epsilon,
)
from ergodic_control_mppi.mppi.single import SingleControllerState
from ergodic_control_mppi.parameters import ControllerParams

#: Per-step fields :func:`step_residuals` returns, in report order.
RESIDUAL_FIELDS = (
    "eps_track", "eps_avg", "eps_fm_k0", "eps_fm_full",
    "rhs_k0", "rhs_full", "jensen_slack",
    "flow_speed", "gauge_regularized", "saturated_fraction",
)


class StepResiduals(NamedTuple):
    """One replanning step's error-budget terms (m^2/s^2 unless noted).

    Attributes:
        eps_track: Squared executed tracking error.
        eps_avg: Squared control-to-motion averaging gap.
        eps_fm_k0: k=0 Euler residual term in the tracking bound.
        eps_fm_full: Full-horizon Euler residual.
        rhs_k0: Sharp bound ``(sqrt(eps_avg) + sqrt(eps_fm_k0))^2``.
        rhs_full: Same with ``eps_fm_full``.
        jensen_slack: Weighted first-slot velocity spread.
        flow_speed: Reference flow speed in m/s.
        gauge_regularized: 1.0 when the speed gauge used the regularized branch.
        saturated_fraction: Fraction of control channels at their bound.
    """

    eps_track: float
    eps_avg: float
    eps_fm_k0: float
    eps_fm_full: float
    rhs_k0: float
    rhs_full: float
    jensen_slack: float
    flow_speed: float
    gauge_regularized: float
    saturated_fraction: float


def _sharp(first: jax.Array, second: jax.Array) -> jax.Array:
    """
    ``(sqrt(a) + sqrt(b))^2``: the sharp L^2 triangle bound on ``||u + w||^2``.
    """
    return jnp.square(jnp.sqrt(first) + jnp.sqrt(second))


def _residuals(params: ControllerParams, carry: SingleControllerState) -> jax.Array:
    """Return the residual terms for one carry, as a stacked array under JIT."""
    epsilon, _ = sample_epsilon(carry.key, params)
    _, _, sampled_positions = _rollouts(
        params, carry.state, carry.controls, epsilon, carry.temperature
    )
    result = mppi_step(
        params, carry.controls, carry.state, carry.key, carry.temperature, carry.memory,
        carry.service_mass,
    )

    d = params.gmm.means.shape[-1]
    origin = carry.state[:2] if d == 2 else carry.state[:d]
    initial = jnp.broadcast_to(origin, (params.mppi.samples, 1, 2 if d == 2 else d))
    evaluation = jnp.concatenate((initial, sampled_positions[:, :-1]), axis=1)
    displacements = sampled_positions - evaluation
    flow = reference_flow(params, evaluation, carry.memory, carry.service_mass)

    delta_t = params.model.delta_t
    weights = result.weights
    # Every rollout starts at z_t, so the field's first source particle -- a median over
    # identical values -- *is* z_t, and flow[0] is h(z_t) with no interpolation.
    reference = flow[0]
    rollout_velocity = displacements[:, 0] / delta_t
    executed_velocity = (step(carry.state, result.control, params.model)[:2 if d == 2 else d] - origin) / delta_t
    average_velocity = jnp.einsum("k,ki->i", weights, rollout_velocity)

    residual = displacements / delta_t - flow[None]
    squared = jnp.sum(residual * residual, axis=-1)
    eps_fm_k0 = jnp.einsum("k,k->", weights, squared[:, 0])
    eps_fm_full = jnp.einsum("k,k->", weights, jnp.sum(squared, axis=-1))

    gap = executed_velocity - average_velocity
    eps_avg = jnp.sum(gap * gap)
    error = executed_velocity - reference
    eps_track = jnp.sum(error * error)
    spread = rollout_velocity - average_velocity[None]
    jensen = jnp.einsum("k,k->", weights, jnp.sum(spread * spread, axis=-1))

    limits = jnp.array(
        [params.model.max_accel_lin_abs] * d + [params.model.max_accel_ang_abs],
        dtype=jnp.float32,
    )
    saturated = jnp.mean(jnp.abs(result.control) >= limits * (1.0 - 1e-6))
    speed = jnp.linalg.norm(reference)
    return jnp.stack([
        eps_track, eps_avg, eps_fm_k0, eps_fm_full,
        _sharp(eps_avg, eps_fm_k0), _sharp(eps_avg, eps_fm_full), jensen,
        speed, (speed < params.field.reference_speed - 1e-6).astype(jnp.float32), saturated,
    ])


def step_residuals(params: ControllerParams, carry: SingleControllerState) -> StepResiduals:
    """
    Compute the Prop. "executed_flow_tracking" budget for one recorded planning step.
    
    Args:
            params: The parameters the step ran under.
            carry: A recorded closed-loop carry, from ``mppi.replay.restore_snapshot``.
    Returns:
            The step's residual terms. ``carry`` is not advanced.
    """
    return StepResiduals(*(float(value) for value in _residuals(params, carry)))


def endpoint_jacobian(
    params: ControllerParams, state: jax.Array, controls: jax.Array
) -> np.ndarray:
    """
    Jacobian of the ``n``-step endpoint map of As. "endpoint", shape ``(6, 3n)``.
    
    Args:
            params: Controller parameters, for the dynamics.
            state: The state to linearize at, shape ``(6,)``.
            controls: The control sequence witness, shape ``(n, 3)``.
    Returns:
            The endpoint Jacobian with respect to the flattened control sequence.
    """
    def endpoint(sequence: jax.Array) -> jax.Array:
        return jax.lax.scan(
            lambda current, control: (step(current, control, params.model),) * 2,
            state,
            sequence.reshape(controls.shape),
        )[0]

    return np.asarray(jax.jacfwd(endpoint)(controls.reshape(-1)))


def residual_walk(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
    steps: int,
    stride: int,
    preflight_steps: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """
    Run one closed loop, returning the executed path and a strided residual history.
    
    Args:
        params: Controller parameters.
        initial_state: State with shape ``(2d + 2,)``.
        initial_controls: Warm-start controls with shape ``(T, d + 1)``.
        key: JAX PRNG key.
        steps: Total control steps; must be divisible by ``stride``.
        stride: Steps between residual evaluations.
        preflight_steps: Stationary planning iterations before motion starts.

    Returns:
        ``(path, residuals)`` with shapes ``(steps, 2d + 2)`` and
        ``(steps // stride, len(RESIDUAL_FIELDS))``.
    """
    if steps % stride:
        raise ValueError(f"steps {steps} is not divisible by stride {stride}")
    from ergodic_control_mppi.mppi.single import (
        initialize_single,
        single_step,
        stationary_step,
    )

    def advance(carry, _):
        next_carry, _ = single_step(params, carry)
        return next_carry, next_carry.state

    def measure(carry, _):
        residual = _residuals(params, carry)
        carry, states = jax.lax.scan(advance, carry, xs=None, length=stride)
        return carry, (states, residual)

    carry = initialize_single(params, initial_state, initial_controls, key)
    carry, _ = jax.lax.scan(
        lambda held, _: (stationary_step(params, held, initial_state)[0], None),
        carry,
        xs=None,
        length=preflight_steps,
    )
    _, (states, residuals) = jax.lax.scan(
        measure, carry, xs=None, length=steps // stride
    )
    return states.reshape(steps, -1), residuals


def project_admissible(position: jax.Array, workspace) -> jax.Array:
    """
    Nearest admissible point: inside the workspace box, and outside any circular pillar.
    """
    lower = jnp.stack((workspace.x_limits[0], workspace.y_limits[0]))
    upper = jnp.stack((workspace.x_limits[1], workspace.y_limits[1]))
    position = jnp.clip(position, lower, upper)

    obstacles = workspace.obstacles
    if obstacles.shape[0] == 0:            # static under JIT; rasterized maps take this branch
        return position
    offset = position - obstacles[:, :2]
    distance = jnp.linalg.norm(offset, axis=-1)
    keepout = obstacles[:, 2] + workspace.safe_distance
    # Push out of the deepest violation only. Pillars are generated at a minimum separation
    # of 1.2 m against keepout radii well under half that, so at most one can be violated;
    # iterating would buy nothing and would not be a fixed point either.
    depth = keepout - distance
    worst = jnp.argmax(depth)
    direction = offset[worst] / jnp.maximum(distance[worst], 1e-9)
    return jnp.where(depth[worst] > 0.0, obstacles[worst, :2] + direction * keepout[worst],
                     position)


def ideal_step(params: ControllerParams, carry):
    """
    One step of the ideal kernel of As. "ideal_kernel_stability", tracking the flow exactly.
    """
    from ergodic_control_mppi.mppi.field import responsibilities
    from ergodic_control_mppi.mppi.single import SingleControllerState, adapt_temperature

    result = mppi_step(
        params, carry.controls, carry.state, carry.key, carry.temperature, carry.memory,
        carry.service_mass,
    )
    # Same construction as _residuals: one key, so these rollouts are the ones the step used,
    # and flow[0] is h(z_t) exactly because every rollout starts at z_t.
    epsilon, _ = sample_epsilon(carry.key, params)
    _, _, sampled_positions = _rollouts(
        params, carry.state, carry.controls, epsilon, carry.temperature
    )
    origin = carry.state[:2]
    initial = jnp.broadcast_to(origin, (params.mppi.samples, 1, 2))
    evaluation = jnp.concatenate((initial, sampled_positions[:, :-1]), axis=1)
    flow = reference_flow(params, evaluation, carry.memory, carry.service_mass)[0]

    nominal = step(carry.state, result.control, params.model)
    advanced = project_admissible(origin + params.model.delta_t * flow, params.workspace)
    next_state = jnp.concatenate((
        advanced,     # Euler step along the field, projected back into the admissible set
        flow,         # velocity *is* the reference; its gap to the realized motion is
                      # eps_track, recoverable from the path since both are recorded
        nominal[4:],  # yaw is unconstrained by a planar field
    ))
    next_carry = SingleControllerState(
        state=next_state,
        controls=result.controls,
        key=result.key,
        temperature=adapt_temperature(carry.temperature, result.weights, params),
        memory=jnp.concatenate((carry.memory[1:], next_state[None, :2]), axis=0),
        step_index=carry.step_index + 1,
        service_mass=params.field.service_decay * carry.service_mass
        + responsibilities(next_state[:2], params.gmm),
    )
    return next_carry


def ideal_walk(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
    steps: int,
    preflight_steps: int = 0,
) -> jax.Array:
    """Run the ideal kernel and return its executed path, shape ``(steps, 6)``."""
    from ergodic_control_mppi.mppi.single import initialize_single, stationary_step

    carry = initialize_single(params, initial_state, initial_controls, key)
    carry, _ = jax.lax.scan(
        lambda held, _: (stationary_step(params, held, initial_state)[0], None),
        carry, xs=None, length=preflight_steps,
    )
    _, states = jax.lax.scan(
        lambda held, _: (lambda nxt: (nxt, nxt.state))(ideal_step(params, held)),
        carry, xs=None, length=steps,
    )
    return states


def ideal_batch(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    keys: jax.Array,
    steps: int,
    preflight_steps: int = 0,
) -> jax.Array:
    """Vmap :func:`ideal_walk` over lanes, matching :func:`residual_batch`'s width rules."""
    states = jnp.broadcast_to(
        jnp.atleast_2d(initial_state), (keys.shape[0], jnp.atleast_2d(initial_state).shape[-1])
    )
    return jax.vmap(
        lambda lane_params, lane_state, lane_key: ideal_walk(
            lane_params, lane_state, initial_controls, lane_key,
            steps=steps, preflight_steps=preflight_steps,
        )
    )(params, states, keys)


def residual_batch(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    keys: jax.Array,
    steps: int,
    stride: int,
    preflight_steps: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """
    Vmap :func:`residual_walk` over lanes, as ``mppi.single.run_batch`` does for runs.
    """
    states = jnp.broadcast_to(
        jnp.atleast_2d(initial_state), (keys.shape[0], jnp.atleast_2d(initial_state).shape[-1])
    )
    return jax.vmap(
        lambda lane_params, lane_state, lane_key: residual_walk(
            lane_params, lane_state, initial_controls, lane_key,
            steps=steps, stride=stride, preflight_steps=preflight_steps,
        )
    )(params, states, keys)

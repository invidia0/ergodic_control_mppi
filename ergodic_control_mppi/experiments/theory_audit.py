r"""Measure every term the closed-loop analysis names, on the loop it analyses.

Sec. "guarantees" states five assumptions, two theorems, four propositions and a corollary,
and none of them currently carries a number. Each statement there is an inequality whose
*both* sides are computable from a recorded planning step, so this module computes them and
reports the slack. Nothing here proves an assumption -- Assumptions 1-5 are conditions, and
what is reported is whether the deployed loop satisfies them and by what margin.

The per-step quantities follow Prop. "executed_flow_tracking" exactly:

    ||v_exec - h(z_t)||^2  <=  2||v_exec - v_bar||^2  +  2 sum_b w_b S_FM_b
    \_______ eps_track ___/     \___ eps_avg ____/       \___ eps_FM ____/

with S_FM the *squared* Euler residual of eq. (S_FM_def), not the first-order surrogate the
MPPI weights are actually built from -- the proposition holds for any nonnegative weights
summing to one, which is what lets the two differ.

Time averages are taken by striding recorded snapshots rather than by instrumenting
``mppi_step``: the averages are Cesaro limits, so a strided subsample is unbiased, and the
hot loop keeps its signature and its numerical branch. ``mppi.replay.replay_step`` already
expands a snapshot back into the exact cloud its step used.
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
    """One replanning step's error-budget terms, all in m^2/s^2 unless noted.

    Attributes:
        eps_track: ``||v_exec - h(z_t)||^2``, the executed tracking error.
        eps_avg: ``||v_exec - v_bar||^2``, the control-to-motion averaging gap.
        eps_fm_k0: ``sum_b w_b`` times the ``k=0`` term of the squared Euler residual --
            the only term the proof of Prop. "executed_flow_tracking" uses.
        eps_fm_full: the same weighted average over the complete ``T``-step residual, which
            is what the paper's ``eps_FM`` denotes.
        rhs_k0: ``(sqrt(eps_avg) + sqrt(eps_fm_k0))^2``, the bound as the proof builds it.
            The sharp L^2 triangle inequality -- Young at its optimal parameter -- rather
            than the ``||a+b||^2 <= 2||a||^2 + 2||b||^2`` split, whose factor 2 the audit
            measured at 2.001 slack and which was therefore the entire looseness. Tight when
            the two errors are parallel.
        rhs_full: the same with ``eps_fm_full``, the bound as stated.
        jensen_slack: ``sum_b w_b ||v_b - v_bar||^2``, the weighted spread of the first-slot
            rollout velocities. With ``eps_avg = 0`` the sharp bound collapses to
            ``eps_fm_k0``, and this is exactly ``rhs_k0 - eps_track`` -- so it *is* the
            conservatism of the k=0 bound, not a proxy for it.
        flow_speed: ``||h(z_t)||`` in m/s.
        gauge_regularized: 1.0 when the speed gauge sat in its regularized branch. Detected
            from the output alone: the gauge returns exactly ``reference_speed`` whenever the
            raw field norm clears the ``1e-3`` floor, and strictly less when it does not.
        saturated_fraction: fraction of the executed control's three channels at their bound.
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
    """``(sqrt(a) + sqrt(b))^2``: the sharp L^2 triangle bound on ``||u + w||^2``.

    Young's inequality at the optimal parameter, rather than at the parameter 1 that gives
    the textbook ``2a + 2b``. The two agree only when ``a = b``; at the measured
    ``eps_avg ~ 4e-10`` against ``eps_fm ~ 0.6`` the cross term is ``~3e-5``, so this is
    ``eps_fm`` to five decimal places and the factor 2 the audit measured as 2.001 slack is
    recovered outright.
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

    origin = carry.state[:2]
    initial = jnp.broadcast_to(origin, (params.mppi.samples, 1, 2))
    evaluation = jnp.concatenate((initial, sampled_positions[:, :-1]), axis=1)
    displacements = sampled_positions - evaluation
    flow = reference_flow(params, evaluation, carry.memory, carry.service_mass)

    delta_t = params.model.delta_t
    weights = result.weights
    # Every rollout starts at z_t, so the field's first source particle -- a median over
    # identical values -- *is* z_t, and flow[0] is h(z_t) with no interpolation.
    reference = flow[0]
    rollout_velocity = displacements[:, 0] / delta_t
    executed_velocity = (step(carry.state, result.control, params.model)[:2] - origin) / delta_t
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
        [params.model.max_accel_lin_abs] * 2 + [params.model.max_accel_ang_abs],
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
    """Compute the Prop. "executed_flow_tracking" budget for one recorded planning step.

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
    """Jacobian of the ``n``-step endpoint map of As. "endpoint", shape ``(6, 3n)``.

    As. "endpoint" asks for full row rank at *some interior* admissible sequence. Passing a
    saturated ``controls`` is therefore not a counterexample to the assumption -- ``clamp``
    zeroes the derivative there, which is exactly why the assumption is stated on the
    interior. Report the interior witness and the saturated case side by side.

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
    """Run one closed loop, returning the executed path and a strided residual history.

    Nested scan rather than a residual at every step: the outer scan evaluates the budget
    once and the inner scan advances ``stride`` steps plainly, so measuring costs one extra
    rollout per ``stride`` instead of one per step. The time averages the analysis defines
    are Cesaro limits, so the strided sample is an unbiased estimator of them.

    **This is a different numerical branch than ``run_single``.** The nested scan lowers
    differently, and the closed loop amplifies a one-ULP difference into metres, so the path
    here is not the path that call produces from the same key -- measured, not assumed. It
    does not matter for what this function is for: the residuals and the trajectory they are
    reported against come from the *same* run, so every inequality check is internally exact,
    and the audit's conclusions are distributional over seeds like every other claim in this
    project. It does mean audit rows must never be pooled with campaign rows.

    Args:
        params: Controller parameters.
        initial_state: State with shape ``(6,)``.
        initial_controls: Warm-start controls with shape ``(T, 3)``.
        key: JAX PRNG key.
        steps: Total control steps; must be divisible by ``stride``. Static under JIT.
        stride: Steps between residual evaluations. Static under JIT.
        preflight_steps: Stationary planning iterations retained before motion starts.

    Returns:
        ``(path, residuals)`` with shapes ``(steps, 6)`` and ``(steps // stride, 10)``, the
        latter's columns ordered as :data:`RESIDUAL_FIELDS`.
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
    """Nearest admissible point: inside the workspace box, and outside any circular pillar.

    As. "ideal_kernel_stability" describes P_0 as realizing the reference flow exactly, but
    read literally that kernel is not confined: the field carries no boundary term (boundary
    and obstacle costs live in the rollout cost and act on the *executed* motion, which P_0
    overrides). An unprojected ideal walk leaves the workspace by a median 40 m, so its
    invariant law is not supported on Omega_free at all and the TV it reports is measuring
    divergence rather than coverage. Projecting restores As. 1 for the comparison kernel, at
    the price of a tracking residual wherever the flow points out -- measured, not assumed.

    The campaign maps carry their obstacles as a rasterized occupancy grid and leave
    ``obstacles`` empty, so only the box clip binds there. That is enough for the question
    being asked: ``coverage_terms`` masks the occupancy by the reachable set and renormalizes,
    so time spent inside a pillar is already excluded from TV, while time spent 40 m outside
    the box was not. The residual exposure -- how much of the ideal law sits inside obstacles
    -- is reported separately as ``inside_obstacle_fraction`` rather than projected away.
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
    """One step of the ideal kernel of As. "ideal_kernel_stability", tracking the flow exactly.

    As. 7 posits a comparison kernel P_0 "sharing the same augmented-state description and
    sampling mechanism but realizing the reference flow exactly". That wording is load-bearing
    and easy to misread: the reference field is *not* a standalone vector field, because its
    Stein term is built from the rollout occupancy. Integrating some position-only field would
    be a different object entirely. So this keeps the whole controller -- rollouts, weights,
    warm start, temperature, memory -- and overrides only the executed motion, which is the
    single thing As. 7 idealizes.

    The resulting motion is deliberately *not* dynamically feasible: velocity snaps to the
    reference rather than accelerating toward it, so the acceleration limit is ignored. That
    is the idealization, not a bug -- it is what makes eps_track identically zero, which is
    the premise Cor. "flow_matching_consistency" needs and the thing being tested here.
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
    """Vmap :func:`residual_walk` over lanes, as ``mppi.single.run_batch`` does for runs.

    The same branch warning applies: a lane count is part of a result's identity, so every
    cell whose residuals are compared must come from one width.
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

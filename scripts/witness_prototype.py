"""Phase 0.5 feasibility gate: is a witness-scored rollout worth a production arm?

Disposable. This is the smallest change that answers one question -- does scoring MPPI
rollouts by the RKHS witness reduce the discrepancy the Phase 0 certificate measures,
without wrecking feasibility, the shipped coverage metrics or the step budget -- so that
the answer, not a hunch, decides whether the witness arm gets built for real. Nothing here
is imported by the package: it monkeypatches :func:`ergodic_control_mppi.mppi.single.mppi_step`
for the duration of one process and reuses the existing KDE and target machinery verbatim.
No config schema, no accumulator, no campaign integration. If the gate passes, this file is
thrown away and the arm is built properly in ``mppi/core.py``; if it fails, this file is the
record of a rejected design.

**The empirical measure is a running mean-embedding accumulator**, which is the object the
herding argument is actually stated for and the fix for the first attempt's failure. The
first attempt (commit ``b3df81e``, rejected -- see
``configs/experiments/witness_arm_decision.md``) used the controller's *fading* 825-sample
ring buffer, so it descended a witness scoped to the last 16.5 s while ``E_N`` scored it
against the whole 400 s path: it improved its own gap and covered worse, the signature of
optimizing the wrong measure.

The accumulator keeps ``A_n(c) = (1/n) sum_i k(c, z_i)`` on a fixed grid and updates it in
place, ``A_{n+1} = (n A_n + k(., z_{n+1})) / (n + 1)``. That is the exact uniform average
over the whole executed history in **bounded** state -- one grid and a counter, independent
of ``n`` -- which is what reconciles herding with main.tex:103's explicit position against
methods needing the entire history. Querying it is a bilinear lookup, so the witness costs
``O(G + K T)`` a step against the buffer's ``O(K T P) = 31M`` kernel evaluations.

**The one remaining shortcut**: ``m_pi`` is the *continuous* smoothed mixture (``smoothed``
inflates every covariance by ``h/2``, which is ``m_pi`` up to the ``pi h`` gauge), not the
reachable-restricted discrete target the certificate uses. The two differ by the obstacle
mass, under 8% on these maps. It is evaluated analytically per query, so only the empirical
half is ever interpolated.

Usage::

    uv run --extra cuda13 python scripts/witness_prototype.py --calibrate
    uv run --extra cuda13 python scripts/witness_prototype.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.uav_pillar_tuning import PREFLIGHT_STEPS, _grid_config
from ergodic_control_mppi.metrics.discrepancy import grid_target, walk
from ergodic_control_mppi.metrics.ergodicity import (
    compute_ball_ergodic_metric,
    compute_fourier_ergodic_metric,
    compute_team_ergodic_error,
    compute_team_occupancy_grid,
)
from ergodic_control_mppi.models.double_integrator import step
from ergodic_control_mppi.mppi import single
from ergodic_control_mppi.mppi.core import (
    MPPIStepResult,
    _flow_tracking_cost,
    _rollouts,
    _smooth,
    adapt_temperature,
    reference_flow,
    sample_epsilon,
)
from ergodic_control_mppi.mppi.field import pdf, responsibilities, smoothed
from ergodic_control_mppi.mppi.single import stack_params
from ergodic_control_mppi.simulation import controller_key, select_device

# --------------------------------------------------------------------------- predeclared

#: Scale of the witness term against the rest of the rollout cost. Set once by
#: ``--calibrate``, which matches the witness term's spread across rollouts to the spread of
#: the field arm's `track_weight * _flow_tracking_cost` at the same states -- so the two arms
#: give their coverage term the same authority against the shared obstacle, boundary and
#: control costs, and a verdict here is not a verdict on an arbitrary gain. Frozen before the
#: comparison runs. Measured at step 2000 of a shipped-arm run on the first lane: the field
#: arm's term has spread 354 across rollouts there against the accumulated witness term's
#: 0.0465, so 7610 is the ratio. One state, so it is a scale, not a tuned gain.
#:
#: It is 330x the first attempt's weight because the accumulated witness is a genuinely
#: flatter object: an average over 2000 executed points against a fading window's local
#: bump. The scale keeps drifting down as coverage improves, and only the obstacle and
#: boundary trade-off feels it -- `alpha = 1.0` zeroes the control cross term, so in free
#: space the witness is the whole cost spread and the ESS-adaptive temperature absorbs its
#: overall size.
WITNESS_WEIGHT = 7610.0

#: The go/no-go rule, written before any prototype result was looked at. `plan.md` defaults:
#: proceed to Phase 1 only if the witness materially reduces the discrepancy without
#: materially degrading feasibility, the shipped coverage metrics or real time.
GATE = {
    # Hard feasibility: no new invalid or colliding run. Failing this is an immediate stop.
    "max_new_obstacle_fraction": 0.0,
    "max_new_outside_fraction": 0.0,
    # Discrepancy: the median final E_N must improve by at least 10%...
    "min_error_improvement": 0.10,
    # ...with the same sign on at least two thirds of paired cells.
    "min_sign_agreement": 2.0 / 3.0,
    # No primary shipped coverage metric may worsen by more than 5% at the median.
    "max_coverage_regression": 0.05,
    # Real time: the median step must stay inside the 50 Hz budget, with at most a 10%
    # regression. Failing either is an immediate stop.
    "step_budget_ms": 20.0,
    "max_time_regression": 0.10,
}

#: The primary shipped coverage metrics the gate protects, and the discrepancy quantities.
COVERAGE = ("occupancy_mse", "fourier_ergodic", "ball_ergodic", "tv")
DISCREPANCY = ("error_final", "weighted_gap", "looseness")


# --------------------------------------------------------------------------- the arm

#: Accumulator resolution, ``(rows, columns)`` of grid *nodes* spanning the workspace. At
#: this profile's 40 x 20 m that is 0.2 m cells, roughly a third of the kernel's own
#: lengthscale ``sqrt(h/2) = 0.69`` m, so bilinear interpolation of a field that smooth is
#: well inside the noise the controller already lives with. The update touches every node
#: once a step, which is 20k kernel evaluations against the rollout cost's millions.
ACCUMULATOR_BINS = (101, 201)


class WitnessControllerState(NamedTuple):
    """:class:`~ergodic_control_mppi.mppi.single.SingleControllerState` plus the accumulator.

    A strict extension: the first seven fields are the shipped carry in the shipped order,
    so the field arm's own update can be reused verbatim on them and ``run_single`` -- which
    only ever reads ``.state`` and ``.temperature`` off the carry -- needs no change.

    Attributes:
        mean_grid: ``A_n``, the empirical mean embedding on the accumulator grid.
        count: ``n``, the number of executed positions folded into ``mean_grid``.
    """

    state: jax.Array
    controls: jax.Array
    key: jax.Array
    temperature: jax.Array
    memory: jax.Array
    step_index: jax.Array
    service_mass: jax.Array
    mean_grid: jax.Array
    count: jax.Array


def accumulator_grid(params) -> jax.Array:
    """Node coordinates of the accumulator, shape ``(rows, columns, 2)``."""
    rows, columns = ACCUMULATOR_BINS
    mesh_x, mesh_y = jnp.meshgrid(
        jnp.linspace(params.workspace.x_limits[0], params.workspace.x_limits[1], columns),
        jnp.linspace(params.workspace.y_limits[0], params.workspace.y_limits[1], rows),
    )
    return jnp.stack((mesh_x, mesh_y), axis=-1)


def kernel_field(grid: jax.Array, position: jax.Array, bandwidth) -> jax.Array:
    """``k(c, z)`` at every accumulator node, shape ``(rows, columns)``."""
    return jnp.exp(-jnp.sum((grid - position) ** 2, axis=-1) / bandwidth)


def interpolate(field: jax.Array, positions: jax.Array, x_limits, y_limits) -> jax.Array:
    """Bilinear lookup of ``field`` at ``(..., 2)`` positions, clamped to the workspace.

    Nearest-neighbour would quantize the witness at the cell size, which is the same order
    as the kernel's own lengthscale -- and the witness exists to *rank* rollouts, so a
    piecewise-constant one would throw away most of the ranking.
    """
    rows, columns = field.shape
    fx = ((positions[..., 0] - x_limits[0]) / (x_limits[1] - x_limits[0])) * (columns - 1)
    fy = ((positions[..., 1] - y_limits[0]) / (y_limits[1] - y_limits[0])) * (rows - 1)
    fx = jnp.clip(fx, 0.0, columns - 1.0)
    fy = jnp.clip(fy, 0.0, rows - 1.0)
    x0 = jnp.floor(fx).astype(jnp.int32)
    y0 = jnp.floor(fy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, columns - 1)
    y1 = jnp.minimum(y0 + 1, rows - 1)
    tx, ty = fx - x0, fy - y0
    lower = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    upper = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return lower * (1.0 - ty) + upper * ty


def accumulate(mean_grid, count, position, grid, bandwidth):
    """Fold one executed position into the running mean, in place and in bounded state."""
    updated = (count * mean_grid + kernel_field(grid, position, bandwidth)) / (count + 1.0)
    return updated, count + 1.0


def witness_cost(params, positions: jax.Array, mean_grid: jax.Array) -> jax.Array:
    """Sum the witness ``w_n`` along each rollout, shape ``(K, T) -> (K,)``.

    ``w_n(z) = (1/n) sum_i k(z, z_i) - m_pi(z)``. The first term is the accumulator, read
    by interpolation; the second is closed form, since convolving the mixture with
    ``k / (pi h) = N(0, (h/2) I)`` inflates every covariance by ``(h/2) I`` -- the identity
    :func:`ergodic_control_mppi.mppi.field.smoothed` already implements for the control path.
    """
    bandwidth = params.field.fine_bandwidth
    empirical = interpolate(mean_grid, positions,
                            params.workspace.x_limits, params.workspace.y_limits)
    target = jnp.pi * bandwidth * pdf(positions, smoothed(params.gmm, bandwidth))
    return jnp.sum(empirical - target, axis=1)


def witness_mppi_step(params, previous_controls, state, key, temperature,
                      mean_grid, count) -> MPPIStepResult:
    """:func:`ergodic_control_mppi.mppi.core.mppi_step` with the witness cost in place of
    the field-tracking cost. Everything else -- sampling, clamping, the control cross term,
    the adaptive temperature, the smoothing -- is the shipped path untouched, so the
    comparison isolates the objective."""
    epsilon, key = sample_epsilon(key, params)
    costs, sampled_controls, sampled_positions = _rollouts(
        params, state, previous_controls, epsilon, temperature
    )
    surrogate = jnp.median(sampled_positions, axis=0)
    costs += WITNESS_WEIGHT * witness_cost(params, sampled_positions, mean_grid)
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


#: The shipped closed-loop entry points, saved when :func:`witness_arm` swaps them out.
_SHIPPED: dict = {}


def witness_initialize(params, initial_state, initial_controls, key):
    """The shipped carry plus a one-sample accumulator at the arming position."""
    base = _SHIPPED["initialize_single"](params, initial_state, initial_controls, key)
    return WitnessControllerState(
        *base,
        mean_grid=kernel_field(accumulator_grid(params), initial_state[:2],
                               params.field.fine_bandwidth),
        count=jnp.asarray(1.0, dtype=jnp.float32),
    )


def witness_single_step(params, carry):
    """One closed-loop step: plan against the accumulator, then fold in what was executed.

    The memory buffer and the service mass are still carried and still updated, so the two
    arms differ in exactly one thing -- which term ranks the rollouts.
    """
    result = witness_mppi_step(params, carry.controls, carry.state, carry.key,
                               carry.temperature, carry.mean_grid, carry.count)
    next_state = step(carry.state, result.control, params.model)
    mean_grid, count = accumulate(carry.mean_grid, carry.count, next_state[:2],
                                  accumulator_grid(params), params.field.fine_bandwidth)
    return WitnessControllerState(
        state=next_state,
        controls=result.controls,
        key=result.key,
        temperature=adapt_temperature(carry.temperature, result.weights, params),
        memory=jnp.concatenate((carry.memory[1:], next_state[None, :2]), axis=0),
        step_index=carry.step_index + 1,
        service_mass=params.field.service_decay * carry.service_mass
        + responsibilities(next_state[:2], params.gmm),
        mean_grid=mean_grid,
        count=count,
    ), result


def witness_stationary_step(params, carry, state):
    """Plan once while holding the executed state, the memory and the accumulator still.

    Preflight must not fold two hundred copies of the arming position into the running
    mean; the shipped arm resets the trail for the same reason.
    """
    next_carry, result = witness_single_step(params, carry)
    held = next_carry._replace(
        state=state,
        memory=jnp.broadcast_to(state[:2], next_carry.memory.shape),
        step_index=jnp.asarray(0, dtype=jnp.int32),
        service_mass=responsibilities(state[:2], params.gmm)
        / jnp.maximum(1.0 - params.field.service_decay, 1e-12),
        mean_grid=kernel_field(accumulator_grid(params), state[:2],
                               params.field.fine_bandwidth),
        count=jnp.asarray(1.0, dtype=jnp.float32),
    )
    return held, result


@contextlib.contextmanager
def witness_arm():
    """Swap the closed loop's three entry points for the witness versions.

    ``run_single`` and ``run_batch`` resolve these as module globals and never touch the
    carry beyond ``.state`` and ``.temperature``, so the extended carry rides through the
    shipped scan without the package moving.
    """
    saved = {name: getattr(single, name)
             for name in ("initialize_single", "single_step", "stationary_step")}
    _SHIPPED.update(saved)
    single.initialize_single = witness_initialize
    single.single_step = witness_single_step
    single.stationary_step = witness_stationary_step
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(single, name, function)


# --------------------------------------------------------------------------- scoring


def lane_metrics(positions, velocities, arrays, config, stride, delta_t) -> dict:
    """Score one executed path on both metric families plus feasibility and smoothness."""
    limits_x = tuple(float(v) for v in config.controller.workspace.x_limits)
    limits_y = tuple(float(v) for v in config.controller.workspace.y_limits)
    density = np.asarray(arrays["target_grid"], dtype=np.float64)
    mask = np.asarray(arrays["reachable_mask"], dtype=bool)
    bins = (density.shape[1], density.shape[0])

    support, weights = grid_target(density, mask, limits_x, limits_y)
    audited = np.asarray(positions[::stride], dtype=np.float64)
    result = walk(audited, support, weights,
                  float(config.controller.field.fine_bandwidth))

    occupancy = compute_team_occupancy_grid(positions, limits_x, limits_y, bins)
    visited = occupancy * mask
    visited = visited / visited.sum() if visited.sum() > 0 else visited
    desired = density * mask
    desired = desired / desired.sum()

    # Feasibility: samples landing in cells the reachable mask excludes, and samples that
    # left the workspace box at all.
    column = np.clip(((positions[:, 0] - limits_x[0]) / (limits_x[1] - limits_x[0])
                      * bins[0]).astype(int), 0, bins[0] - 1)
    row = np.clip(((positions[:, 1] - limits_y[0]) / (limits_y[1] - limits_y[0])
                   * bins[1]).astype(int), 0, bins[1] - 1)
    outside = ((positions[:, 0] < limits_x[0]) | (positions[:, 0] > limits_x[1])
               | (positions[:, 1] < limits_y[0]) | (positions[:, 1] > limits_y[1]))

    # `double_integrator.step` lays the state out as [x, y, vx, vy, yaw, yaw_rate].
    speed = np.linalg.norm(velocities, axis=1)
    jerk = np.diff(velocities, n=2, axis=0) / delta_t ** 2
    return {
        "error_final": float(result["error"][-1]),
        "weighted_gap": float(result["weighted_gap"]),
        "looseness": float(result["bound"][-1] / result["error"][-1]),
        "prefix_holds": bool(np.all(result["error"] <= result["bound"] + 1e-9)),
        "occupancy_mse": compute_team_ergodic_error(
            positions, density, limits_x, limits_y, reachable_mask=mask),
        "fourier_ergodic": compute_fourier_ergodic_metric(
            positions, density, limits_x, limits_y, reachable_mask=mask),
        "ball_ergodic": compute_ball_ergodic_metric(
            occupancy, density, limits_x, limits_y, max_radius=5.0, radii=32,
            reachable_mask=mask),
        "tv": 0.5 * float(np.abs(visited - desired).sum()),
        "obstacle_fraction": float(np.mean(~mask[row, column])),
        "outside_fraction": float(np.mean(outside)),
        "max_speed": float(speed.max()),
        "jerk_rms": float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1)))),
    }


def fly(params, initial, controls, keys, steps, patched: bool) -> tuple[np.ndarray, float]:
    """Run every lane of one arm and return paths and the wall time per control step.

    Timed twice: the first call pays compilation, the second is the measurement. The
    reported number is one control step of the *batched* loop, which advances all lanes at
    once and so amortizes launch latency across them -- it is the right quantity for the
    arm-to-arm ratio and the wrong one for the 50 Hz budget, which :func:`time_single`
    measures at the deployed shape of one lane.
    """
    # `jax.jit` memoizes on the traced function object, not on what its globals resolve to,
    # so without this the second arm silently reuses the first arm's compiled objective --
    # and every metric comes out identical, which reads as "no effect" rather than as a bug.
    jax.clear_caches()
    with witness_arm() if patched else contextlib.nullcontext():
        runner = jax.jit(single.run_batch, static_argnames=("steps", "preflight_steps"))
        result = runner(params, initial, controls, keys,
                        steps=steps, preflight_steps=PREFLIGHT_STEPS)
        jax.block_until_ready(result.path)
        started = time.perf_counter()
        result = runner(params, initial, controls, keys,
                        steps=steps, preflight_steps=PREFLIGHT_STEPS)
        jax.block_until_ready(result.path)
        wall = time.perf_counter() - started
    return np.asarray(result.path), wall / steps


def time_single(params, initial, controls, key, steps: int, patched: bool) -> float:
    """Seconds per control step for one lane -- the shape the 50 Hz budget is stated at.

    The batched loop in :func:`fly` shares one launch across lanes, so its per-step number
    is not what a deployed controller pays. Same two-call protocol: compile, then measure.
    """
    jax.clear_caches()
    with witness_arm() if patched else contextlib.nullcontext():
        runner = jax.jit(single.run_single, static_argnames=("steps", "preflight_steps"))
        for _ in range(2):
            started = time.perf_counter()
            result = runner(params, initial, controls, key,
                            steps=steps, preflight_steps=PREFLIGHT_STEPS)
            jax.block_until_ready(result.path)
            wall = time.perf_counter() - started
    return wall / steps


def _field_arm_to_carry(params, initial_state, initial_controls, key, steps):
    """Fly the shipped arm and return its final carry, which ``run_single`` does not expose.

    Calibration needs a *state*, not a path: both terms have to be compared on the same
    rollout ensemble at the same moment of the same run.
    """
    carry = single.initialize_single(params, initial_state, initial_controls, key)

    def preflight(current, _):
        held, _ = single.stationary_step(params, current, initial_state)
        return held, None

    carry, _ = jax.lax.scan(preflight, carry, xs=None, length=PREFLIGHT_STEPS)

    def advance(current, _):
        following, _ = single.single_step(params, current)
        return following, following.state[:2]

    return jax.lax.scan(advance, carry, xs=None, length=steps)


def calibrate(params, initial, controls, keys, steps: int) -> None:
    """Print the witness weight that matches the field arm's coverage-term authority.

    Both terms are compared by their *spread across rollouts* at the same state, since a
    term's rank authority in the softmax is its spread, not its mean: a constant offset
    cancels in ``costs - min(costs)``.

    Measured after ``steps`` of the shipped arm rather than at the arming position. The
    accumulator's witness shrinks as coverage improves -- that is the objective working --
    so the post-preflight state, where ``A_1`` is a single unit-height bump, would set a
    weight an order of magnitude too small for the regime the run actually spends its time
    in. The first attempt calibrated there because its fading buffer had no such drift.
    """
    lane = jax.tree.map(lambda leaf: leaf[0], params)
    carry, path = jax.jit(_field_arm_to_carry, static_argnames=("steps",))(
        lane, initial, controls, keys[0], steps=steps
    )
    grid = accumulator_grid(lane)
    bandwidth = lane.field.fine_bandwidth
    mean_grid = jnp.mean(
        jax.vmap(lambda z: kernel_field(grid, z, bandwidth))(path), axis=0
    )
    epsilon, _ = sample_epsilon(carry.key, lane)
    _, _, positions = _rollouts(lane, carry.state, carry.controls, epsilon, carry.temperature)
    evaluation = jnp.concatenate(
        (jnp.broadcast_to(carry.state[:2], (lane.mppi.samples, 1, 2)), positions[:, :-1]),
        axis=1,
    )
    flow = reference_flow(lane, evaluation, carry.memory, carry.service_mass)
    tracking = lane.field.track_weight * _flow_tracking_cost(
        flow[None], positions - evaluation, lane.model.delta_t
    )
    raw = witness_cost(lane, positions, mean_grid)
    print(f"calibrated at step {steps} of the field arm, n = {len(path)}")
    print(f"field-arm coverage term spread : {float(jnp.std(tracking)):.4g}")
    print(f"raw witness term spread        : {float(jnp.std(raw)):.4g}")
    print(f"WITNESS_WEIGHT = {float(jnp.std(tracking) / jnp.std(raw)):.3g}")


# --------------------------------------------------------------------------- gate


def report(field: list[dict], witness: list[dict], lanes, times: dict) -> bool:
    """Print the paired table and apply the predeclared rule. Returns whether it passed."""
    def median(rows, name):
        return float(np.median([r[name] for r in rows]))

    print(f"\n{'metric':>18} {'field':>12} {'witness':>12} {'change':>9}  cells better")
    for name in DISCREPANCY + COVERAGE + ("jerk_rms",):
        better = sum(w[name] < f[name] for f, w in zip(field, witness))
        base, new = median(field, name), median(witness, name)
        print(f"{name:>18} {base:12.5g} {new:12.5g} {(new / base - 1) * 100:8.1f}%"
              f"  {better}/{len(field)}")
    for name in ("obstacle_fraction", "outside_fraction", "max_speed"):
        print(f"{name:>18} {median(field, name):12.5g} {median(witness, name):12.5g}")
    print(f"{'step_ms_batched':>18} {times['field'] * 1e3:12.5g} {times['witness'] * 1e3:12.5g}"
          f" {(times['witness'] / times['field'] - 1) * 100:8.1f}%")
    print(f"{'step_ms_1_lane':>18} {times['field_single'] * 1e3:12.5g}"
          f" {times['witness_single'] * 1e3:12.5g}"
          f" {(times['witness_single'] / times['field_single'] - 1) * 100:8.1f}%")

    improvement = 1.0 - median(witness, "error_final") / median(field, "error_final")
    agreement = sum(w["error_final"] < f["error_final"]
                    for f, w in zip(field, witness)) / len(field)
    checks = {
        "no new obstacle contact": (
            median(witness, "obstacle_fraction")
            <= median(field, "obstacle_fraction") + GATE["max_new_obstacle_fraction"]),
        "no new workspace exit": (
            median(witness, "outside_fraction")
            <= median(field, "outside_fraction") + GATE["max_new_outside_fraction"]),
        "prefix inequality holds": all(r["prefix_holds"] for r in field + witness),
        f"median E_N improves >= {GATE['min_error_improvement']:.0%}":
            improvement >= GATE["min_error_improvement"],
        f"sign agreement >= {GATE['min_sign_agreement']:.0%}":
            agreement >= GATE["min_sign_agreement"],
        f"no coverage metric worse by > {GATE['max_coverage_regression']:.0%}": all(
            median(witness, name) <= median(field, name) * (1 + GATE["max_coverage_regression"])
            for name in COVERAGE),
        f"1-lane step <= {GATE['step_budget_ms']} ms":
            times["witness_single"] * 1e3 <= GATE["step_budget_ms"],
        f"step time regression <= {GATE['max_time_regression']:.0%}":
            times["witness_single"] <= times["field_single"] * (1 + GATE["max_time_regression"]),
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print(f"\nGATE: {'PASS -- proceed to Phase 1' if passed else 'FAIL -- stop, keep the field arm'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("results/uav/T150/clutter/maps.json"))
    parser.add_argument("--config", default="results/uav/T150/config.yaml")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--cells", type=int, default=3, help="maps, taken in manifest order")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/uav/witness_prototype.json"))
    parser.add_argument("--timing-steps", type=int, default=2000,
                        help="single-lane steps used for the real-time measurement")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate-steps", type=int, default=2000,
                        help="field-arm steps to build the accumulator the weight is set at")
    arguments = parser.parse_args()

    entries = json.loads(arguments.maps.read_text())["maps"][: arguments.cells]
    loaded = [_grid_config(Path(e["run_dir"]), arguments.config) for e in entries]
    starts = {tuple(np.asarray(a["initial_state"], dtype=np.float64)) for _, _, a in loaded}
    if len(starts) != 1:
        raise SystemExit("run_batch shares one start across lanes; these maps do not agree")

    device = select_device("auto")
    lanes = [(index, seed) for index in range(len(entries)) for seed in range(arguments.seeds)]
    stacked = stack_params(
        [jax.device_put(loaded[index][0].controller, device) for index, _ in lanes]
    )
    keys = jnp.stack([controller_key(seed) for _, seed in lanes])
    initial = jnp.asarray(np.asarray(loaded[0][2]["initial_state"]), dtype=jnp.float32)
    controls = jnp.zeros((loaded[0][0].controller.mppi.horizon, 3), dtype=jnp.float32)

    if arguments.calibrate:
        calibrate(stacked, initial, controls, keys, arguments.calibrate_steps)
        return

    print(f"[prototype] {len(lanes)} lanes x {arguments.steps} steps on {device}; "
          f"WITNESS_WEIGHT = {WITNESS_WEIGHT:.3g}", flush=True)
    paths, times, scores = {}, {}, {}
    for arm, patched in (("field", False), ("witness", True)):
        paths[arm], times[arm] = fly(stacked, initial, controls, keys, arguments.steps, patched)
        if not np.isfinite(paths[arm]).all():
            raise SystemExit(f"nonfinite trajectory on the {arm} arm")
        if arm == "witness" and np.array_equal(paths["field"], paths["witness"]):
            raise SystemExit("both arms flew the same path; the objective swap did not take")
        times[f"{arm}_single"] = time_single(
            jax.tree.map(lambda leaf: leaf[0], stacked), initial, controls, keys[0],
            arguments.timing_steps, patched,
        )
        print(f"[prototype] {arm}: {times[arm] * 1e3:.2f} ms/batched step, "
              f"{times[f'{arm}_single'] * 1e3:.2f} ms/step at one lane", flush=True)
        scores[arm] = [
            lane_metrics(np.asarray(paths[arm][row][:, :2], dtype=np.float64),
                         np.asarray(paths[arm][row][:, 2:4], dtype=np.float64),
                         loaded[index][2], loaded[index][0], arguments.stride,
                         float(loaded[index][0].controller.model.delta_t))
            for row, (index, _) in enumerate(lanes)
        ]

    passed = report(scores["field"], scores["witness"], lanes, times)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({
        "gate": GATE, "witness_weight": WITNESS_WEIGHT, "passed": passed,
        "steps": arguments.steps, "stride": arguments.stride,
        "lanes": [{"map_seed": entries[i]["map_seed"], "obs_num": entries[i]["obs_num"],
                   "seed": s} for i, s in lanes],
        "step_seconds": times, "field": scores["field"], "witness": scores["witness"],
    }, indent=2) + "\n")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

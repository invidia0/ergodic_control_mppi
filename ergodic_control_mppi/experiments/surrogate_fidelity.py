r"""Measure what the median-source surrogate costs, instead of assuming it costs something.

:func:`ergodic_control_mppi.mppi.core.reference_flow` does not build eq. (25) at each rollout's own
states. It evaluates it once on one representative path -- the per-horizon-step median over
rollouts -- and broadcasts the resulting ``(T, 2)`` field to every rollout. That median path
is also the plan the self-repulsion term repels from. Both are a compression: the paper's
``eps_comp``.

The compression is only visible to MPPI through the *ranking* it induces over rollouts,
since the weights are a softmax of the costs and a monotone re-labelling of every cost leaves
the update unchanged. So the quantity that matters is the rank correlation between

    surrogate:  S_track(v* on the median path, broadcast to all rollouts)   shape (K,)
    faithful:   S_track(v* queried at each rollout's own states)            shape (K,)

not the pointwise field error. This module computes both on real planning steps of the
deployed profile and reports Spearman rho between them.

Why the rollout count is swept rather than fixed at the deployed ``N=250``: the faithful
field is ``K`` evaluations of a field whose plan term is ``O(T^2)``, against the surrogate's
one, which is the reason the surrogate exists. The sweep covers the range where the faithful
field is affordable, and any claim made from it must state that range.

    uv run python -m ergodic_control_mppi.experiments.surrogate_fidelity
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.mppi.core import (
    _flow_tracking_cost,
    _rollouts,
    _smooth,
    effective_sample_fraction,
    field_at,
    reference_flow,
    sample_epsilon,
)
from ergodic_control_mppi.mppi.single import SingleControllerState
from ergodic_control_mppi.parameters import ControllerParams


class StepFidelity(NamedTuple):
    """One planning step's surrogate-vs-faithful comparison.

    Attributes:
        spearman: Rank correlation between the two per-rollout *flow* cost vectors. This
            isolates the compression: it is the surrogate's effect on the only term it
            touches, before the task costs and ``track_weight`` are applied.
        pearson: Linear correlation of the same two vectors, on the raw cost scale.
        weight_tv: Total-variation distance between the two weight simplices MPPI would
            actually form -- the *total* rollout cost at the step's own temperature. This is
            the operational number: it is how differently the two fields steer the update.
        field_rmse: RMS difference between the two flow fields at the rollout states, m/s.
        cost_scale: RMS of the faithful cost vector, for reading ``field_rmse`` against.
        ess_fraction: effective sample fraction of the faithful weights. Reported because
            ``weight_tv`` is only informative when the weights are actually concentrated --
            two near-uniform simplices agree trivially. The profile targets 0.3.
        control_gap: ``||u_surrogate - u_faithful||`` for the control the step would execute,
            as a fraction of the linear acceleration bound. This is the end of the chain and
            the only quantity the vehicle sees: when the weights are near one-hot, a
            ``weight_tv`` of 1 means only that a different rollout won, which two rollouts
            proposing the same command render irrelevant.
    """

    spearman: float
    pearson: float
    weight_tv: float
    field_rmse: float
    cost_scale: float
    ess_fraction: float
    control_gap: float


def faithful_reference_flow(
    params: ControllerParams,
    evaluation_positions: jax.Array,
    memory: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """Return the reference field queried at each rollout's own states, shape ``(K, T, 2)``.

    :func:`field_at` verbatim -- the same call the controller makes, with the median path
    replaced by each rollout's own states in *both* roles it plays. Two things change:

    * the query set, so the score, the memory repulsion and the speed schedule are all read
      where that rollout actually goes rather than where the median path goes;
    * the plan the plan-repulsion term repels from, which for a given rollout is its own
      horizon rather than the shared median.

    Nothing is transcribed, so this cannot drift from the control path. It is
    ``O(K)`` calls of the deployed field rather than one, which is the whole reason the
    surrogate exists.
    """
    return jax.lax.map(
        lambda states: field_at(params, states, states, memory, service_mass),
        evaluation_positions,
    )


def step_fidelity(
    params: ControllerParams, carry: SingleControllerState
) -> StepFidelity:
    """Compare the two flow-matching cost vectors on the cloud one step actually sampled."""
    epsilon, _ = sample_epsilon(carry.key, params)
    task_costs, sampled_controls, sampled_positions = _rollouts(
        params, carry.state, carry.controls, epsilon, carry.temperature
    )
    origin = carry.state[:2]
    initial = jnp.broadcast_to(origin, (params.mppi.samples, 1, 2))
    evaluation = jnp.concatenate((initial, sampled_positions[:, :-1]), axis=1)
    displacements = sampled_positions - evaluation

    surrogate_field = jnp.broadcast_to(
        reference_flow(params, evaluation, carry.memory, carry.service_mass)[None],
        evaluation.shape,
    )
    faithful_field = faithful_reference_flow(params, evaluation, carry.memory, carry.service_mass)

    delta_t = params.model.delta_t
    surrogate = _flow_tracking_cost(surrogate_field, displacements, delta_t)
    faithful = _flow_tracking_cost(faithful_field, displacements, delta_t)

    difference = surrogate_field - faithful_field
    field_rmse = jnp.sqrt(jnp.mean(jnp.sum(difference * difference, axis=-1)))

    # The weights MPPI would actually form: the *total* rollout cost at the step's own
    # temperature, exactly as ``mppi_step`` builds it. Softmaxing the bare flow cost instead
    # would compare the two fields at a scale the controller never sees -- the flow term
    # enters multiplied by ``track_weight``, and against the task costs it competes with.
    def weights(flow_cost: jax.Array) -> jax.Array:
        cost = task_costs + params.field.track_weight * flow_cost
        shifted = -(cost - jnp.min(cost)) / carry.temperature
        exponentiated = jnp.exp(shifted)
        return exponentiated / jnp.sum(exponentiated)

    faithful_weights = weights(faithful)
    surrogate_weights = weights(surrogate)
    weight_tv = 0.5 * jnp.sum(jnp.abs(surrogate_weights - faithful_weights))

    # The update eq. (mppi_update) builds, smoothed as ``mppi_step`` smooths it, down to the
    # one control the step would execute.
    def executed(weight: jax.Array) -> jax.Array:
        controls = carry.controls + jnp.einsum(
            "k,kti->ti", weight, sampled_controls - carry.controls
        )
        return _smooth(controls, params.mppi.smooth_window)[0]

    control_gap = jnp.linalg.norm(
        executed(surrogate_weights) - executed(faithful_weights)
    ) / params.model.max_accel_lin_abs

    return StepFidelity(
        spearman=_spearman(surrogate, faithful),
        pearson=_pearson(surrogate, faithful),
        weight_tv=weight_tv,
        field_rmse=field_rmse,
        cost_scale=jnp.sqrt(jnp.mean(faithful * faithful)),
        ess_fraction=effective_sample_fraction(faithful_weights, params.mppi.samples),
        control_gap=control_gap,
    )


def _ranks(values: jax.Array) -> jax.Array:
    """Ordinal ranks. The costs are continuous, so ties are a measure-zero concern."""
    order = jnp.argsort(values)
    return jnp.argsort(order).astype(values.dtype)


def _pearson(first: jax.Array, second: jax.Array) -> jax.Array:
    """Pearson correlation of two vectors."""
    first = first - jnp.mean(first)
    second = second - jnp.mean(second)
    denominator = jnp.linalg.norm(first) * jnp.linalg.norm(second)
    return jnp.sum(first * second) / jnp.maximum(denominator, 1e-30)


def _spearman(first: jax.Array, second: jax.Array) -> jax.Array:
    """Spearman rank correlation: Pearson on the ranks."""
    return _pearson(_ranks(first), _ranks(second))


def fidelity_walk(
    params: ControllerParams,
    initial_state: jax.Array,
    initial_controls: jax.Array,
    key: jax.Array,
    steps: int,
    stride: int,
    preflight_steps: int = 0,
) -> jax.Array:
    """Fly one closed loop, comparing the two costs every ``stride`` steps.

    Same nested-scan shape as
    :func:`ergodic_control_mppi.experiments.theory_audit.residual_walk`, and the same caveat:
    this is a different numerical branch than ``run_single``, so the path here is not that
    call's path from the same key. It does not matter -- both cost vectors are read off the
    *same* cloud at every measured step, so each comparison is internally exact.

    Returns:
        Array of shape ``(steps // stride, 5)``, columns ordered as :class:`StepFidelity`.
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
        return next_carry, None

    def measure(carry, _):
        fidelity = jnp.stack(list(step_fidelity(params, carry)))
        carry, _ = jax.lax.scan(advance, carry, xs=None, length=stride)
        return carry, fidelity

    carry = initialize_single(params, initial_state, initial_controls, key)
    carry, _ = jax.lax.scan(
        lambda held, _: (stationary_step(params, held, initial_state)[0], None),
        carry,
        xs=None,
        length=preflight_steps,
    )
    _, fidelities = jax.lax.scan(measure, carry, xs=None, length=steps // stride)
    return fidelities


def summarize(fidelities: np.ndarray) -> dict[str, float]:
    """Median and worst case over measured steps, as the driver reports them."""
    fidelities = np.asarray(fidelities)
    return {
        "steps": int(fidelities.shape[0]),
        "spearman_median": float(np.median(fidelities[:, 0])),
        "spearman_min": float(np.min(fidelities[:, 0])),
        "pearson_median": float(np.median(fidelities[:, 1])),
        "weight_tv_median": float(np.median(fidelities[:, 2])),
        "weight_tv_max": float(np.max(fidelities[:, 2])),
        "field_rmse_median": float(np.median(fidelities[:, 3])),
        "cost_scale_median": float(np.median(fidelities[:, 4])),
        "ess_fraction_median": float(np.median(fidelities[:, 5])),
        "control_gap_median": float(np.median(fidelities[:, 6])),
        "control_gap_p90": float(np.quantile(fidelities[:, 6], 0.90)),
        "control_gap_max": float(np.max(fidelities[:, 6])),
    }


def _main() -> None:
    """Sweep the rollout count on the campaign maps and print one row per (K, map, seed)."""
    import argparse
    import json
    from dataclasses import replace
    from pathlib import Path

    from ergodic_control_mppi.experiments.uav_pillar_tuning import (
        PREFLIGHT_STEPS,
        _grid_config,
    )
    from ergodic_control_mppi.simulation import controller_key, select_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uav_profile.yaml")
    parser.add_argument("--maps", type=Path, default=Path("results/uav/campaign_maps.json"))
    # The faithful field is O((KT)^2) in its source set; past the low tens the exact median
    # heuristic over pi_hat_t no longer fits. Any claim made from this sweep says so.
    parser.add_argument("--samples", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--map-count", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    device = select_device(arguments.device)
    manifest = json.loads(arguments.maps.read_text(encoding="utf-8"))
    entries = manifest["maps"][: arguments.map_count]

    header = ("samples", "map_seed", "seed", "spearman_median", "spearman_min",
              "weight_tv_median", "weight_tv_max", "field_rmse_median", "cost_scale_median",
              "ess_fraction_median", "control_gap_median", "control_gap_p90",
              "control_gap_max")
    print("\t".join(header), flush=True)

    pooled: dict[int, list[np.ndarray]] = {}
    for samples in arguments.samples:
        for entry in entries:
            config, _, _ = _grid_config(Path(entry["run_dir"]), arguments.config)
            controller = replace(
                config.controller,
                mppi=replace(config.controller.mppi, samples=samples),
            )
            params = jax.device_put(controller, device)
            state = jnp.asarray(entry["initial_state"], dtype=jnp.float32)
            controls = jnp.zeros((controller.mppi.horizon, 3), dtype=jnp.float32)
            for seed in arguments.seeds:
                fidelities = jax.jit(
                    fidelity_walk, static_argnames=("steps", "stride", "preflight_steps")
                )(
                    params, state, controls, controller_key(seed),
                    steps=arguments.steps, stride=arguments.stride,
                    preflight_steps=PREFLIGHT_STEPS,
                )
                fidelities = np.asarray(jax.block_until_ready(fidelities))
                pooled.setdefault(samples, []).append(fidelities)
                summary = summarize(fidelities)
                print("\t".join([
                    str(samples), str(entry["map_seed"]), str(seed),
                    f"{summary['spearman_median']:.4f}", f"{summary['spearman_min']:.4f}",
                    f"{summary['weight_tv_median']:.4f}", f"{summary['weight_tv_max']:.4f}",
                    f"{summary['field_rmse_median']:.4f}",
                    f"{summary['cost_scale_median']:.4g}",
                    f"{summary['ess_fraction_median']:.4f}",
                    f"{summary['control_gap_median']:.5f}",
                    f"{summary['control_gap_p90']:.5f}",
                    f"{summary['control_gap_max']:.5f}",
                ]), flush=True)

    print("\n== pooled over maps and seeds ==", flush=True)
    for samples in arguments.samples:
        summary = summarize(np.concatenate(pooled[samples], axis=0))
        print(f"K={samples:4d}  n={summary['steps']:4d}  "
              f"spearman median {summary['spearman_median']:.4f} "
              f"min {summary['spearman_min']:.4f}  |  "
              f"weight TV median {summary['weight_tv_median']:.2e} "
              f"max {summary['weight_tv_max']:.2e}  |  "
              f"ESS {summary['ess_fraction_median']:.3f}  |  "
              f"control gap median {summary['control_gap_median']:.2e} "
              f"p90 {summary['control_gap_p90']:.2e} "
              f"max {summary['control_gap_max']:.2e}", flush=True)


if __name__ == "__main__":
    _main()

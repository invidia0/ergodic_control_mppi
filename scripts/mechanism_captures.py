"""Fly one seed per arm in the empty workspace and record what Sec. III-D/E need to draw.

Promoted from the scratch capture harness. It re-uses the baselines scenario builder, so a
drawn path is the same measurement the CSVs score -- nothing here feeds a number, it only
draws, but it must draw the controller that the numbers describe.

Beyond the executed path it records two things ``run_single`` does not return:

* the **plan** at one frozen step, together with the memory buffer and service mass at that
  step, which is everything ``field.potential`` needs to evaluate ``Phi`` on a grid. That is
  what lets ``fig_plan_gain`` put a potential contour under the trajectory -- only possible
  because the field is a gradient.
* the **service mass** every ``--stride`` steps. ``sigma_j`` and the bent ``log w_j`` are
  pure functions of it, so the time series in ``fig_service_gate`` is recomputed offline
  rather than instrumented into the control loop.

The open tier only: both figures exist to show mechanism, and in clutter any shape is
partly attributable to obstacle avoidance.

    uv run python scripts/mechanism_captures.py --out results/report/captures \\
        --axis plan_gain --levels 0,3,6,10
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = "configs/uav_profile.yaml"
#: Step the plan and buffer are frozen at. Late enough that the memory buffer is full and
#: the vehicle has settled into its working regime, early enough to be well inside a
#: 20 000-step run at every arm.
DEFAULT_FREEZE = 12000


def capture(config_path: str, overrides: dict, seed: int, steps: int, stride: int,
            freeze: int) -> dict:
    """Fly one arm and return the arrays the mechanism figures read."""
    import jax
    import jax.numpy as jnp

    from ergodic_control_mppi.config import load_config
    from ergodic_control_mppi.experiments.baselines import _open_arrays, _open_scenario
    from ergodic_control_mppi.experiments.uav_ablation import _apply
    from ergodic_control_mppi.mppi.single import initialize_single, single_step
    from ergodic_control_mppi.simulation import controller_key

    config = _apply(load_config(config_path), overrides)
    config = replace(config, run=replace(config.run, steps=steps, seed=seed))
    scenario = _open_scenario(config)
    arrays = _open_arrays(scenario)
    params = scenario.params

    state = jnp.asarray(np.asarray(arrays["initial_state"]), dtype=jnp.float32)
    controls = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
    carry = initialize_single(params, state, controls, controller_key(seed))

    # One scan per segment rather than one over the whole run: the frozen step needs the
    # *carry*, which a single scan would not hand back, and splitting at `freeze` costs one
    # extra compile against threading a conditional through 20 000 steps.
    def segment(carry, length):
        def body(held, _):
            nxt, result = single_step(params, held)
            return nxt, (nxt.state[:2], nxt.service_mass)

        return jax.lax.scan(body, carry, None, length=length)

    scanned = jax.jit(segment, static_argnames=("length",))
    frozen_at = min(max(freeze, 1), steps)
    carry, first = scanned(carry, frozen_at)
    # The plan the controller would repel from at this step, built exactly as
    # `reference_flow` builds it.
    _, result = jax.jit(single_step)(params, carry)
    plan = np.asarray(result.surrogate)
    frozen = {
        "plan": plan,
        "memory": np.asarray(carry.memory),
        "service_mass": np.asarray(carry.service_mass),
        "freeze_step": np.asarray(frozen_at),
    }
    positions, masses = first
    if steps > frozen_at:
        _, rest = scanned(carry, steps - frozen_at)
        positions = jnp.concatenate((positions, rest[0]), axis=0)
        masses = jnp.concatenate((masses, rest[1]), axis=0)

    gmm = params.gmm
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "service_mass_history": np.asarray(masses[::stride], dtype=np.float64),
        "stride": np.asarray(stride),
        "delta_t": np.asarray(float(params.model.delta_t)),
        "means": np.asarray(gmm.means),
        # The panel draws 2-sigma Mahalanobis rings, which need the covariances themselves;
        # GMMParams stores the field as `covariance`, singular.
        "covariances": np.asarray(gmm.covariance),
        "log_weights": np.asarray(gmm.log_weights),
        "recency": np.asarray(
            float(params.field.memory_decay) ** np.arange(params.mppi.memory_length)[::-1]
        ),
        "fine_bandwidth": np.asarray(float(params.field.fine_bandwidth)),
        "memory_gain": np.asarray(float(params.field.memory_gain)),
        "memory_balance": np.asarray(float(params.field.memory_balance)),
        "plan_gain": np.asarray(float(params.field.plan_gain)),
        "release_ratio": np.asarray(float(params.field.release_ratio)),
        "deficit_ceiling": np.asarray(float(params.field.deficit_ceiling)),
        "limits": np.asarray(list(scenario.map_x_limits) + list(scenario.map_y_limits)),
        **frozen,
    }


def _level(text: str):
    """Parse one axis level: ``off`` means the knob's disabling value, 0."""
    return 0.0 if text.lower() in ("off", "none") else float(text)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/report/captures"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--axis", required=True,
                        help="FieldParams field to sweep, e.g. plan_gain or release_ratio")
    parser.add_argument("--levels", required=True,
                        help="Comma-separated levels; 'off' means 0")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=20,
                        help="Steps between recorded service-mass samples")
    parser.add_argument("--freeze", type=int, default=DEFAULT_FREEZE,
                        help="Step the plan and memory buffer are frozen at")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    written = []
    for text in arguments.levels.split(","):
        level = _level(text.strip())
        data = capture(arguments.config, {arguments.axis: level}, arguments.seed,
                       arguments.steps, arguments.stride, arguments.freeze)
        path = arguments.out / f"{arguments.axis}_{text.strip()}_s{arguments.seed}.npz"
        np.savez_compressed(path, **data)
        written.append(str(path))
        print(f"wrote {path}: {len(data['positions'])} steps", flush=True)
    index = arguments.out / f"{arguments.axis}_index.json"
    index.write_text(json.dumps({"axis": arguments.axis, "captures": written}, indent=2)
                     + "\n", encoding="utf-8")
    print(f"wrote {index}")


if __name__ == "__main__":
    main()

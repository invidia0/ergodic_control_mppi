"""Choose a baseline's own hyperparameter on the open field, by the fidelity gate's criterion.

Making the solver cells square (``_solver_shape``) turned two settings from counts of cells
on a 0.5 x 0.25 m grid into lengths in metres. That silently retuned both baselines:
``fmec_bandwidth`` widened three- to fourfold and dropped FMEC below the gate at 0 of 3
modes, the over-wide-kernel under-exploration that Fig. 2 of Sun et al. describes;
``hedac_sensor`` changed by the same factor but stayed above the gate, which is exactly why
it needs checking rather than trusting -- passing is not evidence that a number is right.

Choosing these values *for* the baselines is required for a fair comparison rather than a
favour to them. The gate admits a reimplementation only once it reproduces its published
behaviour, and a baseline left at a number that is an artefact of a grid bug would lose for
our reasons instead of its own. The selection rule is therefore: among the settings that
reach every mode, take the one where the baseline scored best. Picking a competitor's weaker
setting is the failure mode worth guarding against here, so the rule is deliberately biased
against us.

Where the passing range is wide and the scores within it are non-monotonic, that is the
chaotic closed loop rather than a real optimum -- three seeds cannot resolve a factor under
about two -- and the choice should be read as "the baseline's strongest admissible setting",
not as a tuned optimum.

    JAX_PLATFORMS=cpu uv run python scripts/baseline_param_sweep.py \
        --method fmec --parameter fmec_bandwidth --values 0.35,0.5,0.7,1.0,2.0
"""

import argparse

from ergodic_control_mppi.experiments import baselines
from ergodic_control_mppi.experiments.baselines import BaselineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=[m for m in baselines.METHODS
                                                            if m != "ours"])
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    # Three seeds decide nothing on a loop this chaotic. The `hedac_sensor` sweep put one
    # setting at 3/3 modes and another at 2/3 with a six-fold better metric, which is a
    # one-seed difference deciding a six-fold one -- so a shortlist gets re-run wider before
    # anything is chosen. Leave the device alone too: the backend moves the target grid in
    # the last float32 bits and that alone flipped a mode count between CPU and GPU, so the
    # deciding run must be on whatever device the tier will fly.
    parser.add_argument("--seeds", default=",".join(str(s) for s in baselines.FIDELITY_SEEDS))
    arguments = parser.parse_args()
    seeds = tuple(int(s) for s in arguments.seeds.split(","))

    from ergodic_control_mppi.config import load_config

    if not hasattr(BaselineConfig(), arguments.parameter):
        raise SystemExit(f"BaselineConfig has no {arguments.parameter!r}")

    config = load_config("configs/uav_profile.yaml")
    scenario = baselines._open_scenario(config)
    state0 = baselines._open_arrays(scenario)["initial_state"]

    import jax

    # Recorded because it changes the answer, not as provenance decoration.
    print(f"{arguments.method} / {arguments.parameter} sweep on {jax.devices()[0]}, "
          f"{len(seeds)} seeds x {arguments.steps} steps\n", flush=True)
    for value in (float(v) for v in arguments.values.split(",")):
        cfg = BaselineConfig(steps=arguments.steps, **{arguments.parameter: value})
        check = baselines.fidelity_check(arguments.method, scenario, state0, cfg=cfg,
                                         steps=arguments.steps, seeds=seeds)
        print(f"  {arguments.parameter} {value:5.2f}  "
              f"{'PASS' if check['passed'] else 'FAIL'}  "
              f"modes {check['modes_reached']}  best {check['ergodic_best']:.2e}  "
              f"final {check['ergodic_final']:.2e}  {check['distance_m']:.0f} m", flush=True)


if __name__ == "__main__":
    main()

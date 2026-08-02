"""Why the MPPI weights look the way they do: ESS behaviour and cost composition.

Two diagnostics that answer questions the coverage metrics cannot. Both are cheap on
purpose -- they were written after an 18-seed x 20 000-step sweep was queued to answer a
question a 600-step single-seed run could settle in four minutes.

``ess`` sweeps candidate temperature caps and penalty scales and reports where the adaptive
loop actually converges. ESS and temperature settle within ~350 steps, so a short run is
sufficient, and it runs on CPU so it need not contend with a sweep on the GPU. The CPU and
GPU backends give different trajectories, but ESS is set by the cost *scale* rather than
the path, so it screens combinations faithfully.

``costs`` decomposes one planning step's per-rollout cost into its terms. Only cross-rollout
variance matters -- the softmax sees differences, so a term with a huge mean but no spread
is invisible to the weighting, and a term with modest mean but large spread dominates it.

    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics ess --run-dir ...
    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics costs --run-dir ...
"""

import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.common import append_csv
from ergodic_control_mppi.mppi.core import _grid_cost, _rollouts, sample_epsilon
from ergodic_control_mppi.mppi.replay import restore_snapshot
from ergodic_control_mppi.mppi.single import initialize_single, single_step

ESS_FIELDS = [
    "arm", "penalty_scale", "boundary_scale", "lam_max", "steps", "seed",
    "lambda_final", "lambda_pinned", "ess_percent_median", "ess_percent_settled",
    "ess_target_percent", "samples", "device",
]
COST_FIELDS = [
    "step", "term", "mean", "std", "spread_p99_min", "share_of_total_spread", "device",
]


def _grid_config(run_directory: Path, config_path: str):
    """Load the profile with the recorded run's grid folded in."""
    arrays = np.load(run_directory / "figure_data.npz", allow_pickle=False)
    config = load_config(config_path)
    workspace = replace(
        config.controller.workspace,
        grid=jnp.asarray(np.asarray(arrays["grid"], dtype=np.float32)),
        grid_origin=jnp.asarray(np.asarray(arrays["grid_origin"], dtype=np.float32)),
        grid_resolution=float(arrays["grid_resolution"]),
    )
    return replace(config, controller=replace(config.controller, workspace=workspace)), arrays


def _scaled(params, penalty_scale: float, boundary_scale: float, lam_max: float | None):
    """Apply a penalty rescale and temperature cap to controller params."""
    workspace = params.workspace
    if penalty_scale != 1.0:
        workspace = replace(
            workspace,
            obstacle_cost=workspace.obstacle_cost * penalty_scale,
            out_of_map_cost=workspace.out_of_map_cost * penalty_scale,
            boundary_weight=workspace.boundary_weight * penalty_scale,
        )
    if boundary_scale != 1.0:
        workspace = replace(
            workspace, boundary_weight=workspace.boundary_weight * boundary_scale
        )
    mppi = params.mppi
    if lam_max is not None:
        mppi = replace(mppi, temperature_max=lam_max)
    return replace(params, workspace=workspace, mppi=mppi)


def ess_sweep(run_directory: Path, config_path: str, output: Path, steps: int,
              seed: int, device: str) -> None:
    """Report where the adaptive temperature converges under each candidate setting."""
    config, arrays = _grid_config(run_directory, config_path)
    initial = jnp.asarray(arrays["snap_state"][0], dtype=jnp.float32)
    combinations = [
        ("shipped", 1.0, 1.0, None),
        ("lam_max_1e3", 1.0, 1.0, 1e3),
        ("lam_max_1e4", 1.0, 1.0, 1e4),
        ("lam_max_1e5", 1.0, 1.0, 1e5),
        ("penalty_0.1", 0.1, 1.0, None),
        ("penalty_0.01", 0.01, 1.0, None),
        ("penalty_0.1_lam1e3", 0.1, 1.0, 1e3),
        ("penalty_0.01_lam1e3", 0.01, 1.0, 1e3),
        ("boundary_0.1_lam1e3", 1.0, 0.1, 1e3),
        ("boundary_0.01_lam1e3", 1.0, 0.01, 1e3),
    ]
    step = jax.jit(single_step)
    for name, penalty, boundary, lam_max in combinations:
        params = _scaled(config.controller, penalty, boundary, lam_max)
        carry = initialize_single(
            params, initial,
            jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32),
            jax.random.key(seed),
        )
        fractions, temperature = [], 0.0
        for _ in range(steps):
            carry, result = step(params, carry)
            weights = np.asarray(result.weights)
            fractions.append(100.0 / (np.sum(weights * weights) * weights.size))
            temperature = float(carry.temperature)
        settled = float(np.mean(fractions[-min(200, steps // 3):]))
        cap = params.mppi.temperature_max
        row = {
            "arm": name, "penalty_scale": penalty, "boundary_scale": boundary,
            "lam_max": cap, "steps": steps, "seed": seed,
            "lambda_final": temperature,
            "lambda_pinned": int(abs(temperature - cap) / cap < 1e-6),
            "ess_percent_median": float(np.median(fractions)),
            "ess_percent_settled": settled,
            "ess_target_percent": 100.0 * params.mppi.ess_target,
            "samples": params.mppi.samples, "device": device,
        }
        append_csv(output, row, ESS_FIELDS)
        print(f"{name:<22} lambda={temperature:>9.1f} "
              f"{'PINNED' if row['lambda_pinned'] else '     '} "
              f"ESS_settled={settled:5.2f}%", flush=True)


def cost_decomposition(run_directory: Path, config_path: str, output: Path,
                       device: str) -> None:
    """Split one step's per-rollout cost into terms, by cross-rollout spread."""
    config, arrays = _grid_config(run_directory, config_path)
    params = config.controller
    for index, recorded in enumerate(arrays["snapshot_steps"]):
        carry = restore_snapshot(arrays, index)
        epsilon, _ = sample_epsilon(carry.key, params)
        costs, _, positions = _rollouts(
            params, carry.state, carry.controls, epsilon, carry.temperature
        )
        limits = params.workspace
        gap = jnp.minimum(
            jnp.minimum(positions[..., 0] - limits.x_limits[0],
                        limits.x_limits[1] - positions[..., 0]),
            jnp.minimum(positions[..., 1] - limits.y_limits[0],
                        limits.y_limits[1] - positions[..., 1]),
        )
        encroach = jnp.maximum(limits.boundary_margin - gap, 0.0)
        outside = (
            (positions[..., 0] < limits.x_limits[0]) | (positions[..., 0] > limits.x_limits[1])
            | (positions[..., 1] < limits.y_limits[0]) | (positions[..., 1] > limits.y_limits[1])
        )
        terms = {
            "grid_obstacle": np.asarray(_grid_cost(positions, params)).sum(1),
            "boundary_margin": np.asarray(limits.boundary_weight * encroach * encroach).sum(1),
            "out_of_map": np.asarray(outside * limits.out_of_map_cost).sum(1),
        }
        total = np.asarray(costs)
        terms["flow_and_control"] = total - sum(terms.values())
        total_spread = float(np.percentile(total, 99) - total.min()) or 1.0
        for term, values in terms.items():
            spread = float(np.percentile(values, 99) - values.min())
            append_csv(output, {
                "step": int(recorded), "term": term, "mean": float(values.mean()),
                "std": float(values.std()), "spread_p99_min": spread,
                "share_of_total_spread": spread / total_spread, "device": device,
            }, COST_FIELDS)
        print(f"step {int(recorded):>6}  " + "  ".join(
            f"{t}={float(np.percentile(v, 99) - v.min()):.3g}" for t, v in terms.items()
        ), flush=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ess", "costs"])
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Run directory holding figure_data.npz")
    parser.add_argument("--config", default="configs/uav_profile.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "gpu"])
    arguments = parser.parse_args()
    default = Path("results/uav") / f"diagnostic_{arguments.mode}.csv"
    output = arguments.output or default
    if arguments.mode == "ess":
        ess_sweep(arguments.run_dir, arguments.config, output, arguments.steps,
                  arguments.seed, arguments.device)
    else:
        cost_decomposition(arguments.run_dir, arguments.config, output, arguments.device)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

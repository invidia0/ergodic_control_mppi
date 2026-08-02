"""Offline ablation of the deployment profile on a recorded UAV map.

The vehicle is not in the loop here: every arm is the ideal offline controller flying the
same inflated grid, start state, and horizon that the UAV flew. That isolates controller
tuning from the deployment, and it is what makes the arms comparable at all -- a UAV run
answers a different question and costs 30x the wall time.

One row per (arm, seed), not per arm. Aggregates belong in the analysis, not the archive.

Determinism matters more here than anywhere else in the repo. Two arms differing only by a
kernel choice the XLA autotuner made under different machine load produced trajectories
16 m apart in this workspace; ``ergodic_control_mppi/__init__`` pins that off, and every
row records ``device`` because the CPU and GPU backends are individually deterministic but
disagree with each other.

    uv run python -m ergodic_control_mppi.experiments.uav_ablation \
        --run-dir results/uav/baseline --seeds 18 --steps 20000
"""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.common import append_csv
from ergodic_control_mppi.metrics.ergodicity import (
    compute_fourier_ergodic_metric,
    compute_team_ergodic_error,
)
from ergodic_control_mppi.metrics.modes import compute_mode_metrics
from ergodic_control_mppi.simulation import run_simulation

FIELDS = [
    "arm",
    "axis",
    "value",
    "seed",
    "steps",
    "map_seed",
    "occupancy_mse",
    "fourier_ergodic",
    "all_modes_reached",
    "first_all_modes_s",
    "mode_visits",
    "mode_cycles",
    "mode_dwell_median_s",
    "in_mode_fraction",
    "x_max",
    "path_length_m",
    "wall_seconds",
    "device",
    "jax_version",
]

# Each arm is one override of the shipped profile, so the profile itself is the control.
# `axis` groups them for the analysis; `baseline` is the profile unmodified.
ARMS: list[tuple[str, str, Any, dict]] = [
    ("baseline", "-", "-", {}),
    # The lengthscale that decides mode visitation. The repulsion kernel peaks at
    # sqrt(h/2); the working band is 0.55-0.80 of a mode radius (2.35-2.65 m here).
    ("h_0.94", "fine_bandwidth", 0.94, {"fine_bandwidth": 0.94}),
    ("h_2.35", "fine_bandwidth", 2.35, {"fine_bandwidth": 2.35}),
    ("h_4.0", "fine_bandwidth", 4.0, {"fine_bandwidth": 4.0}),
    ("h_6.0", "fine_bandwidth", 6.0, {"fine_bandwidth": 6.0}),
    ("h_6.6", "fine_bandwidth", 6.6, {"fine_bandwidth": 6.6}),
    ("h_8.5", "fine_bandwidth", 8.5, {"fine_bandwidth": 8.5}),
    ("h_11.0", "fine_bandwidth", 11.0, {"fine_bandwidth": 11.0}),
    # The bank averages scales as total/Q, so extra scales split weight rather than add
    # resolution. Included to show a bank cannot beat one well-placed scale.
    ("Q3_fine", "memory_scales", 3, {"memory_scales": 3, "fine_bandwidth": 0.08,
                                     "coarse_bandwidth": 6.6}),
    ("Q3_coarse", "memory_scales", 3, {"memory_scales": 3, "fine_bandwidth": 0.08,
                                       "coarse_bandwidth": 11.0}),
    ("Q2", "memory_scales", 2, {"memory_scales": 2, "fine_bandwidth": 0.94,
                                "coarse_bandwidth": 11.0}),
    # Temporal half of the memory. Long tau flattens the excess field once the spatial
    # scale is basin-sized, which removes the gradient that drives transits.
    ("tau_3", "memory_time", 3.0, {"memory_time": 3.0}),
    ("tau_11", "memory_time", 11.0, {"memory_time": 11.0}),
    ("tau_20", "memory_time", 20.0, {"memory_time": 20.0}),
    ("gain_8", "memory_gain", 8.0, {"memory_gain": 8.0}),
    ("gain_60", "memory_gain", 60.0, {"memory_gain": 60.0}),
    ("memory_off", "memory_gain", 0.0, {"memory_gain": 0.0}),
    # The curl. The campaign flagged theta for possible removal; this is the deployment's
    # own evidence rather than a transfer.
    ("theta_0", "theta", 0.0, {"theta": 0.0}),
    ("theta_15", "theta", 15.0, {"theta": 15.0}),
    ("theta_45", "theta", 45.0, {"theta": 45.0}),
    ("flow_6000", "flow_weight", 6000.0, {"flow_weight": 6000.0}),
    ("T_150", "T", 150, {"T": 150}),
    ("T_500", "T", 500, {"T": 500}),
    # Temperature saturation. Measured on the flown profile, the MPPI weights are one-hot
    # (ESS 1-7 of 250 against an ess_target of 0.3) and the temperature sits pinned at
    # lam_max from step ~250 onward, so the update is argmin over rollouts rather than a
    # weighted average. Two candidate causes, tested separately here because they pull in
    # opposite directions on constraint hardness.
    #
    # (a) The cap is simply too low for the cost scale: reaching ESS = 0.3 needs
    #     lambda ~ spread/ln(K) ~ 2e4.
    ("lam_max_1e3", "lam_max", 1e3, {"lam_max": 1e3}),
    ("lam_max_1e4", "lam_max", 1e4, {"lam_max": 1e4}),
    ("lam_max_1e5", "lam_max", 1e5, {"lam_max": 1e5}),
    # (b) The penalties are out of scale against the objective. Per rollout the flow term
    #     spreads by ~2.5e3 while out-of-map and boundary spread by 3e4-1.1e5, so the
    #     softmax ranks rollouts on constraint violation and barely sees coverage. Scaling
    #     the penalties down keeps them decisive (still 10-30x a typical flow step) while
    #     letting the weights discriminate among the rollouts that satisfy them.
    ("penalty_0.1", "penalty_scale", 0.1, {"penalty_scale": 0.1}),
    ("penalty_0.01", "penalty_scale", 0.01, {"penalty_scale": 0.01}),
    # Neither single axis can work on its own: rescaling leaves the cap binding, and
    # raising the cap leaves the penalties drowning the objective. Reaching ESS = 0.3
    # needs lambda ~ the cost std, which after a 0.01 rescale is still 500-900.
    ("penalty_0.01_lam1e3", "combined", "0.01+1e3", {"penalty_scale": 0.01, "lam_max": 1e3}),
    ("penalty_0.01_lam1e4", "combined", "0.01+1e4", {"penalty_scale": 0.01, "lam_max": 1e4}),
    ("penalty_0.1_lam1e3", "combined", "0.1+1e3", {"penalty_scale": 0.1, "lam_max": 1e3}),
    ("penalty_0.1_lam1e4", "combined", "0.1+1e4", {"penalty_scale": 0.1, "lam_max": 1e4}),
    # The structurally honest variant. `oom_cost` and the obstacle cost are genuine
    # constraints and *should* dominate -- a violating rollout deserves ~zero weight. Only
    # boundary_weight * encroach^2 is a shaping term wearing hard-constraint magnitude
    # (up to 225 per step against a flow term averaging 1.5). Shrinking the constraints
    # along with it is what risks a rollout cutting a corner through an obstacle for
    # better coverage: at scale 0.01 the obstacle cost falls to 10/step, ~40 over a few
    # cells, against a flow spread of ~2500. These arms shrink only the shaping term.
    ("boundary_0.1_lam1e3", "boundary_scale", "0.1+1e3",
     {"boundary_scale": 0.1, "lam_max": 1e3}),
    ("boundary_0.01_lam1e3", "boundary_scale", "0.01+1e3",
     {"boundary_scale": 0.01, "lam_max": 1e3}),
    ("boundary_0.01_lam1e4", "boundary_scale", "0.01+1e4",
     {"boundary_scale": 0.01, "lam_max": 1e4}),
]


def _apply(config, overrides: dict):
    """Return the config with one arm's overrides applied.

    ``memory_time`` and ``theta`` are stored derived (as a decay and a rotation matrix),
    so they are re-derived here exactly as ``config.py`` does rather than set directly.
    """
    delta_t = config.controller.model.delta_t
    mppi, stein = config.controller.mppi, config.controller.stein
    workspace = config.controller.workspace
    overrides = dict(overrides)
    if "penalty_scale" in overrides:
        # Scale the three constraint penalties together. Their *ratio* to the flow cost is
        # what sets how much of the softmax spread is objective rather than violation, so
        # they have to move as a group or the balance between them shifts too.
        scale = overrides.pop("penalty_scale")
        workspace = replace(
            workspace,
            obstacle_cost=workspace.obstacle_cost * scale,
            out_of_map_cost=workspace.out_of_map_cost * scale,
            boundary_weight=workspace.boundary_weight * scale,
        )
    if "boundary_scale" in overrides:
        workspace = replace(
            workspace,
            boundary_weight=workspace.boundary_weight * overrides.pop("boundary_scale"),
        )
    if "lam_max" in overrides:
        mppi = replace(mppi, temperature_max=overrides.pop("lam_max"))
    if "memory_time" in overrides:
        tau = overrides.pop("memory_time")
        stein = replace(stein, memory_decay=float(np.exp(-delta_t / tau)))
        mppi = replace(mppi, memory_length=int(np.ceil(3.0 * tau / delta_t)))
    if "theta" in overrides:
        angle = np.radians(overrides.pop("theta"))
        stein = replace(
            stein,
            rotation=jnp.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=jnp.float32,
            ),
        )
    if "T" in overrides:
        mppi = replace(mppi, horizon=overrides.pop("T"))
    if "K" in overrides:
        mppi = replace(mppi, samples=overrides.pop("K"))
    if overrides:
        stein = replace(stein, **overrides)
    return replace(
        config,
        controller=replace(
            config.controller, mppi=mppi, stein=stein, workspace=workspace
        ),
    )


def run_arm(config, arrays, seed: int, steps: int, overrides: dict, device: str) -> dict:
    """Run one (arm, seed) cell and return its row."""
    config = _apply(config, overrides)
    config = replace(config, run=replace(config.run, steps=steps, seed=seed))
    started = time.perf_counter()
    result = run_simulation(
        config, device=device, initial_state=np.asarray(arrays["initial_state"])
    )
    wall = time.perf_counter() - started
    positions = np.asarray(result.paths[:, 0, :2])
    workspace = config.controller.workspace
    limits = (
        tuple(float(v) for v in np.asarray(workspace.x_limits)),
        tuple(float(v) for v in np.asarray(workspace.y_limits)),
    )
    target = np.asarray(arrays["target_grid"])
    mask = np.asarray(arrays["reachable_mask"])
    modes = compute_mode_metrics(
        positions,
        np.asarray(config.controller.gmm.means),
        np.asarray(config.controller.gmm.covariance_inverse),
        config.controller.model.delta_t,
    )
    return {
        "seed": seed,
        "steps": steps,
        "occupancy_mse": compute_team_ergodic_error(
            positions[:, None, :], target, *limits, reachable_mask=mask
        ),
        "fourier_ergodic": compute_fourier_ergodic_metric(
            positions[:, None, :], target, *limits, reachable_mask=mask
        ),
        "all_modes_reached": int(np.isfinite(modes["first_all_modes_s"])),
        "first_all_modes_s": modes["first_all_modes_s"],
        "mode_visits": modes["mode_visits"],
        "mode_cycles": modes["mode_cycles"],
        "mode_dwell_median_s": modes["mode_dwell_median_s"],
        "in_mode_fraction": modes["in_mode_fraction"],
        "x_max": float(positions[:, 0].max()),
        "path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "wall_seconds": wall,
        "device": result.device,
        "jax_version": jax.__version__,
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Recorder output whose grid, start, and target the arms reuse")
    parser.add_argument("--output", type=Path, default=Path("results/uav/ablation.csv"))
    parser.add_argument("--seeds", type=int, default=18)
    parser.add_argument("--first-seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--arms", default="", help="Comma-separated arm names; default all")
    arguments = parser.parse_args()

    manifest = json.loads((arguments.run_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(arguments.run_dir / "arrays.npz", allow_pickle=False)
    base = load_config("configs/uav_profile.yaml")
    base = replace(
        base,
        controller=replace(
            base.controller,
            workspace=replace(
                base.controller.workspace,
                grid=jnp.asarray(np.asarray(arrays["grid"], dtype=np.float32)),
                grid_origin=jnp.asarray(
                    [float(v) for v in np.asarray(arrays["grid_origin"])], dtype=jnp.float32
                ),
                grid_resolution=float(arrays["grid_resolution"]),
            ),
        ),
    )
    wanted = set(filter(None, arguments.arms.split(",")))
    completed = set()
    if arguments.output.exists():
        with arguments.output.open(encoding="utf-8", newline="") as stream:
            completed = {
                (row["arm"], int(row["seed"]), int(row["map_seed"]), int(row["steps"]))
                for row in csv.DictReader(stream)
            }
    seeds = range(arguments.first_seed, arguments.first_seed + arguments.seeds)
    for name, axis, value, overrides in ARMS:
        if wanted and name not in wanted:
            continue
        for seed in seeds:
            identity = (name, seed, int(manifest["map_seed"]), arguments.steps)
            if identity in completed:
                print(f"SKIP arm={name} seed={seed} map={manifest['map_seed']}")
                continue
            row = run_arm(base, arrays, seed, arguments.steps, overrides, arguments.device)
            row.update({"arm": name, "axis": axis, "value": value,
                        "map_seed": manifest["map_seed"]})
            append_csv(arguments.output, {f: row.get(f, "") for f in FIELDS}, FIELDS)
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()

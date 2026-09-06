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

    # ------------------------------------------------------------------ mechanism axes
    # lambda: the fading-memory gain. `memory_off` is a necessity row -- it deletes one of
    # the three terms of Phi outright and carries the argument that the term is load-bearing.
    ("memory_off", "memory_gain", 0.0, {"memory_gain": 0.0}),
    ("gain_30", "memory_gain", 30.0, {"memory_gain": 30.0}),
    ("gain_120", "memory_gain", 120.0, {"memory_gain": 120.0}),
    # g: the plan self-repulsion gain. `plan_off` is the second necessity row, and the one
    # the mechanism claim rests on: the memory can repel from the past but cannot make the
    # *plan* space-filling, so at g = 0 a compact repeated circuit stays admissible.
    ("plan_off", "plan_gain", 0.0, {"plan_gain": 0.0}),
    ("plan_3", "plan_gain", 3.0, {"plan_gain": 3.0}),
    ("plan_10", "plan_gain", 10.0, {"plan_gain": 10.0}),
    # h: the one surviving lengthscale. The repulsion kernel peaks at sqrt(h/2), and with
    # the plan term supplying the filling this no longer has to be basin-sized to evict the
    # vehicle -- which is why the deployed value moved from 5.0 to 0.94. The levels bracket
    # both readings.
    ("h_0.47", "fine_bandwidth", 0.47, {"fine_bandwidth": 0.47}),
    ("h_2.35", "fine_bandwidth", 2.35, {"fine_bandwidth": 2.35}),
    ("h_5.0", "fine_bandwidth", 5.0, {"fine_bandwidth": 5.0}),
    # tau_M: the temporal half of the memory. tau is a trail *length*, tau * v.
    ("tau_3", "memory_time", 3.0, {"memory_time": 3.0}),
    ("tau_11", "memory_time", 11.0, {"memory_time": 11.0}),
    # a: trail avoidance against over-coverage correction. One level, because moving b at
    # fixed lambda walks a hyperbola in (lambda_t, lambda_e) -- the axis screens flat for a
    # reparameterization reason, not an empirical one.
    ("balance_0.5", "memory_balance", 0.5, {"memory_balance": 0.5}),
    # sigma*: the per-mode release. `release_off` is the third necessity row -- it leaves the
    # promotion-only bend, which is capped at log((c+1)/c) = 3.04 nats and provably cannot
    # overturn the 18.7/31.1-nat Delta_j margins. If the vehicle still escapes basins at this
    # level, the demotion term is not what escapes them and Sec. III-E is wrong.
    ("release_off", "release_ratio", 0.0, {"release_ratio": 0.0}),
    ("release_1.75", "release_ratio", 1.75, {"release_ratio": 1.75}),
    ("release_3.0", "release_ratio", 3.0, {"release_ratio": 3.0}),
    # c: the destination bend. A PRE-REGISTERED NULL. The theory says a 3.04-nat promotion
    # cannot overturn an 18.7-nat margin, so c should show no effect at any level. A knob
    # predicted flat and measured flat is evidence for the promotion/demotion split, not a
    # wasted arm; a knob predicted flat and measured live falsifies it.
    ("ceiling_0", "deficit_ceiling", 0.0, {"deficit_ceiling": 0.0}),
    ("ceiling_0.5", "deficit_ceiling", 0.5, {"deficit_ceiling": 0.5}),
    # tau_s: the service window. Deliberately not the trail's 5.5 s -- the trail is a length
    # of path and this is a history of visits; measured, sigma only discriminates served from
    # unserved at 40-60 s.
    ("service_20", "service_time", 20.0, {"service_time": 20.0}),
    ("service_90", "service_time", 90.0, {"service_time": 90.0}),
    # beta: transit speedup. 1 restores the flat constant-speed gauge exactly.
    ("transit_1", "transit_speedup", 1.0, {"transit_speedup": 1.0}),
    ("transit_8", "transit_speedup", 8.0, {"transit_speedup": 8.0}),
    # eps_s: the service floor. One level, because it is a *second* threshold on the same
    # sigma -- half-release at 1 + eps_s = 1.3 -- competing with the demotion's release at
    # sigma* = 2.24. Worth measuring once before deciding whether to pin it.
    ("floor_1.0", "service_floor", 1.0, {"service_floor": 1.0}),

    # ------------------------------------------------------------------ MPPI axes
    # Horizon reach is L = reference_speed * T * dt, and at the achieved speed the shipped
    # T = 350 plans 3.2 m ahead against a 15.6 m hop. T is swept as a proxy for L.
    ("T_150", "T", 150, {"T": 150}),
    ("T_500", "T", 500, {"T": 500}),
    ("T_750", "T", 750, {"T": 750}),
    # Rollout count. More samples average more divergent directions into one update, so K
    # trades decisiveness for smoothness.
    ("K_125", "K", 125, {"K": 125}),
    ("K_500", "K", 500, {"K": 500}),
    # The one arm that changes the cost model rather than the numbers: K x T x 2 floats
    # per lane is ~2.8 MB of positions alone, so this moves the loop from
    # launch-latency bound toward throughput bound. Runs on its own matched-width
    # branch -- see scripts/final_ablation.py.
    ("K_1000", "K", 1000, {"K": 1000}),
    # The control-cost coefficient is lambda*(1-alpha), so at alpha < 1 a large control cost
    # spreads the sample costs, the ESS controller raises lambda to compensate, and that
    # raises the control cost again. A 48-cell probe measured the runaway: at alpha <= 0.9
    # lambda is pinned at cap with ESS at the 1/K floor and the vehicle achieves 0.35-0.50
    # m/s against a commanded 1.8. alpha = 1.0 severs the coupling and is the deployed value,
    # so 0.9 is here as the low anchor.
    ("alpha_0.9", "alpha", 0.90, {"alpha": 0.90}),
    # Fraction of rollouts that ignore the warm start. Zero makes the plan fully committed
    # to its previous solution, which is the regime a long dwell lives in.
    ("explore_0", "exploration", 0.0, {"exploration": 0.0}),
    ("lam_max_1e4", "lam_max", 1e4, {"lam_max": 1e4}),
    # gamma_track, the weight on the cost that makes a rollout follow -grad Phi. Named for
    # the parameter and not for `flow_weight`, the config key it used to have: an arm called
    # `flow_*` in a campaign whose whole point is that the Stein flow was removed reads as a
    # survivor of it. Nothing Stein-era is swept here -- `config.py` raises on every
    # withdrawn key.
    ("gamma_1500", "track_weight", 1500.0, {"track_weight": 1500.0}),
    ("gamma_6000", "track_weight", 6000.0, {"track_weight": 6000.0}),
    # reference_speed normalises the field, so only its *direction* matters and this asks
    # for the speed alpha = 1.0 achieves by a different route.
    ("refspeed_2.5", "reference_speed", 2.5, {"reference_speed": 2.5}),
    ("refspeed_3.0", "reference_speed", 3.0, {"reference_speed": 3.0}),
    # `oom_cost` and the obstacle cost are genuine constraints and *should* dominate -- a
    # violating rollout deserves ~zero weight. Only boundary_weight * encroach^2 is a
    # shaping term wearing hard-constraint magnitude, so these two shrink the shaping term
    # and the constraint set separately rather than together.
    ("penalty_0.1", "penalty_scale", 0.1, {"penalty_scale": 0.1}),
    ("boundary_0.1", "boundary_scale", 0.1, {"boundary_scale": 0.1}),
]

# The campaign's arm set. Selected from ARMS rather than redeclared so an arm name means the
# same override in every archive. Here it is every arm: the Stein-era table carried
# diagnostics for hypotheses that have since closed, and the port dropped them rather than
# spending cells restating dead questions.
#
# 22 mechanism arms and 16 MPPI arms against one baseline. The three necessity rows are
# `memory_off`, `plan_off` and `release_off` -- one per term of Phi that the argument claims
# is load-bearing -- and the two `ceiling_*` arms are the pre-registered null.
FINAL_ARMS = tuple(name for name, *_ in ARMS)

_BY_NAME = {name: (axis, value, overrides) for name, axis, value, overrides in ARMS}
_unknown = [name for name in FINAL_ARMS if name not in _BY_NAME]
if _unknown:
    raise ValueError(f"FINAL_ARMS names absent from the ablation table: {_unknown}")
if len(set(FINAL_ARMS)) != len(FINAL_ARMS):
    raise ValueError("FINAL_ARMS contains duplicates")

# `K = 1000` quadruples the rollout tensor and may not fit the campaign's lane count, so the
# whole K axis runs as its own matched-width batch with its own baseline replicate rather
# than letting one arm silently sit on a different numerical branch than its comparators.
QUARANTINED_AXES = ("K",)


def _apply(config, overrides: dict):
    """Return the config with one arm's overrides applied.

    ``memory_time`` and ``service_time`` are stored derived (as decays), so they are
    re-derived here exactly as ``config.py`` does rather than set directly.
    """
    delta_t = config.controller.model.delta_t
    mppi, field = config.controller.mppi, config.controller.field
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
    if "smooth_window" in overrides:
        # Static, so each level is its own compile and levels cannot share a batched call.
        mppi = replace(mppi, smooth_window=int(overrides.pop("smooth_window")))
    if "alpha" in overrides:
        # The control-cost weight is lambda*(1-alpha) and lambda is adapted online, so this
        # axis sets how strongly control cost is coupled to the temperature loop rather
        # than its absolute size. Sweep it against lam_max, not alone.
        mppi = replace(mppi, alpha=overrides.pop("alpha"))
    if "memory_time" in overrides:
        tau = overrides.pop("memory_time")
        field = replace(field, memory_decay=float(np.exp(-delta_t / tau)))
        mppi = replace(mppi, memory_length=int(np.ceil(3.0 * tau / delta_t)))
    if "service_time" in overrides:
        # Stored derived, exactly as config.py does it. Unlike memory_time this sets no
        # buffer length -- the accumulator is a J-vector, so the window is free of the
        # O(P^2) occupancy term and tau_s can be tens of seconds at no cost.
        field = replace(
            field, service_decay=float(np.exp(-delta_t / overrides.pop("service_time")))
        )
    if "T" in overrides:
        mppi = replace(mppi, horizon=overrides.pop("T"))
    if "K" in overrides:
        mppi = replace(mppi, samples=overrides.pop("K"))
    if "exploration" in overrides:
        mppi = replace(mppi, exploration=overrides.pop("exploration"))
    # Whatever is left names a FieldParams field directly; an unknown key raises here
    # rather than being silently ignored, which is what keeps an arm table honest.
    if overrides:
        field = replace(field, **overrides)
    return replace(
        config,
        controller=replace(
            config.controller, mppi=mppi, field=field, workspace=workspace
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

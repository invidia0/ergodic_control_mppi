"""Resumable development/holdout tuning for the seven-cell pillar campaign."""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.summary import compute_row
from ergodic_control_mppi.experiments.common import append_csv
from ergodic_control_mppi.experiments.uav_ablation import ARMS
from ergodic_control_mppi.experiments.uav_ablation import FIELDS as ABLATION_FIELDS
from ergodic_control_mppi.experiments.uav_ablation import _apply
from ergodic_control_mppi.experiments.uav_pillars import (
    QUALIFICATION_FIELDS,
    geometry_sentence,
    inflation_text,
)
from ergodic_control_mppi.simulation import run_simulation

PREFLIGHT_STEPS = 200
SCREEN_ARMS = {
    "shipped": {},
    "common_cap": {"lam_max": 1e5},
    "T500": {"T": 500, "lam_max": 1e5},
    "K500": {"K": 500, "lam_max": 1e5},
    "T500_K500": {"T": 500, "K": 500, "lam_max": 1e5},
}
APPROACH_ARMS = {
    "base": {},
    "tau11": {"memory_time": 11.0},
    "h0.94": {"fine_bandwidth": 0.94},
    "h2.35": {"fine_bandwidth": 2.35},
    "tau11_h0.94": {"memory_time": 11.0, "fine_bandwidth": 0.94},
    "tau11_h2.35": {"memory_time": 11.0, "fine_bandwidth": 2.35},
}
# The broad one-factor sweep. Arms are *selected* from the ablation table rather than
# redeclared, so one arm definition serves both this campaign and the standalone ablation
# runner, and an arm name means the same override in both archives.
#
# The 10k cap layer this stage writes is complete and balanced -- one row per (arm, seed),
# whether or not the cell toured -- which is what lets it double as a second UAV ablation
# campaign under `scripts/report_figures.py`. The gated 20k layer cannot: survivors and
# non-survivors have different step counts, so they are not paired.
SWEEP_ARMS = (
    "baseline",
    "theta_0", "theta_15", "theta_45", "theta_60", "theta_75",
    "gain_8", "gain_30", "gain_60",
    "tau_3", "tau_11", "tau_20", "tau_30",
    "T_150", "T_500", "T_750",
    "h_0.94", "h_2.35", "h_8.5",
    "ell_self_0.25", "ell_self_4.0",
    "balance_0.5",
    "flow_1500", "flow_6000",
    "penalty_0.1", "boundary_0.1",
    "K_125", "K_500",
    "lam_max_1e5",
    "explore_0", "explore_0.3",
)
ABLATION_ARMS = {name: (axis, value, overrides) for name, axis, value, overrides in ARMS}
_missing = sorted(set(SWEEP_ARMS) - set(ABLATION_ARMS))
if _missing:
    raise ValueError(f"sweep names absent from the ablation table: {_missing}")

FIELDS = [
    "stage", "arm", "configuration", "map_seed", "seed", "steps",
    "preflight_steps", "horizon", "samples", "lam_max", "ess_target",
    "ess_settled_median", "ess_settled_mean", "temperature_settled_median",
    "temperature_cap_fraction", "occupancy_mse", "fourier_ergodic",
    "all_modes_reached", "first_all_modes_s", "mode_visits", "mode_cycles",
    "mode_dwell_median_s", "in_mode_fraction", "collisions", "min_clearance_m",
    "x_max", "path_length_m", "wall_seconds", "device", "jax_version",
    # Appended, never reordered: `axis` and `value` make a sweep row ablation-shaped so
    # `scripts/report_figures.py` can read this archive directly. Blank on the gated stages.
    "axis", "value",
]
CAP_FIELDS = FIELDS + ["accepted"]


def select_split(
    rows: list[dict[str, str]], blocked_mass: dict[int, float] | None = None
) -> dict[str, object]:
    """Select three development and three holdout maps from the qualifying seeds.

    ``blocked_mass`` maps a seed to its **worst mode's** out-of-reach target mass; when
    supplied, the six are the qualifying seeds carrying the least of it, and each
    representative is the least of its three. Without it the order is by seed, which is
    arbitrary.

    Why rank on it: qualification already requires every mode reachable and at least two
    of three mode-to-mode segments blocked, so every candidate is a genuine detour
    problem. What it does *not* look at is how much of the target lies inside inflation.

    Why the *worst mode* and not the average: ranking on the aggregate was tried and was
    anti-correlated with what it was meant to protect. Over fifteen generated maps, the two
    best by aggregate (10.1% and 10.4%) were among the three worst by single mode (22.5%
    and 25.4%) -- an average over three lobes buys a badly blocked one with two clean ones.
    Two flights on the 10.1% map each missed the 22.5% mode and no other. A tour needs
    every mode, so the binding constraint is the worst lobe, not the mean.

    This selects the environment, never the controller or the objective. The obstacles,
    the inflation and the target density are untouched, the detour requirement is
    unchanged, and the retained value is reported per map in the campaign report -- the
    benchmark is stated, not quietly made easier.
    """
    qualifying = [int(row["map_seed"]) for row in rows if int(row["qualifies"])]
    if blocked_mass:
        # Only measured candidates, and least-blocked first. A seed with no built map is
        # not a zero -- treating it as one would rank the unmeasured ahead of everything.
        qualifying = sorted(
            (seed for seed in qualifying if seed in blocked_mass),
            key=lambda seed: (blocked_mass[seed], seed),
        )
    else:
        qualifying.sort()
    if len(qualifying) < 6:
        raise ValueError(
            f"found {len(qualifying)} qualifying maps with a measurement; need 6"
        )
    selected = qualifying[:6]
    development, holdout = sorted(selected[:3]), sorted(selected[3:])

    by_seed = {int(row["map_seed"]): row for row in rows}

    def representative(seeds):
        if blocked_mass:
            return min(seeds, key=lambda seed: blocked_mass.get(seed, 0.0))
        return sorted(
            seeds, key=lambda seed: (float(by_seed[seed]["free_fraction"]), seed)
        )[1]

    return {
        "development": development,
        "holdout": holdout,
        "development_representative": representative(development),
        "holdout_representative": representative(holdout),
    }


def candidate_blocked_mass(root: Path) -> dict[int, float]:
    """Worst mode's out-of-reach target mass per candidate map, for ``select_split``."""
    return {
        int(row["map"]): float(row["worst_mode_unreachable"])
        for row in map_diagnostics(root)
    }


def write_split_selection(
    qualification: Path, output: Path, blocked_mass: dict[int, float] | None = None
) -> dict[str, object]:
    """Write the immutable six-map development/holdout selection."""
    with qualification.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selection = select_split(rows, blocked_mass)
    selected = set(selection["development"]) | set(selection["holdout"])
    representatives = {
        selection["development_representative"], selection["holdout_representative"]
    }
    for row in rows:
        seed = int(row["map_seed"])
        row["selected"] = int(seed in selected)
        row["representative"] = int(seed in representatives)
    with qualification.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUALIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def _merge(*overrides: dict) -> dict:
    merged = {}
    for override in overrides:
        merged.update(override)
    return merged


def stage_arms(stage: str, base_arm: str, winner: str) -> dict[str, dict]:
    """Return the preregistered arm overrides for one gated stage."""
    if base_arm not in SCREEN_ARMS:
        raise ValueError(f"unknown screen base arm: {base_arm}")
    base = SCREEN_ARMS[base_arm]
    if stage == "sweep":
        return {name: ABLATION_ARMS[name][2] for name in SWEEP_ARMS}
    if stage == "screen":
        return SCREEN_ARMS
    if stage == "approach":
        return {name: _merge(base, arm) for name, arm in APPROACH_ARMS.items()}
    if stage == "holdout":
        if winner not in APPROACH_ARMS:
            raise ValueError(f"unknown approach winner: {winner}")
        return {
            "baseline": SCREEN_ARMS["shipped"],
            "winner": _merge(base, APPROACH_ARMS[winner]),
        }
    raise ValueError(f"unknown stage: {stage}")


def _grid_config(run_directory: Path, config_path: str = "configs/uav_profile.yaml"):
    """Load a map's grid onto a controller config.

    ``config_path`` exists for the Cor. "flow_matching_consistency" sweep, which flies the
    same maps under several controller configurations; everything else takes the default and
    is unaffected.
    """
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(run_directory / "arrays.npz", allow_pickle=False)
    config = load_config(config_path)
    workspace = replace(
        config.controller.workspace,
        grid=jnp.asarray(np.asarray(arrays["grid"], dtype=np.float32)),
        grid_origin=jnp.asarray(arrays["grid_origin"], dtype=jnp.float32),
        grid_resolution=float(arrays["grid_resolution"]),
    )
    return replace(config, controller=replace(config.controller, workspace=workspace)), manifest, arrays


def score_run(config, arrays, manifest: dict, seed: int, steps: int, positions,
              velocities, ess_fractions, temperatures, wall: float,
              device: str) -> dict[str, object]:
    """Score one finished rollout into a sweep row.

    Split out of :func:`run_cell` so the sequential and batched drivers cannot drift apart
    on how a cell is scored -- the rollout differs between them, the scoring must not.
    """
    delta_t = config.controller.model.delta_t
    row = compute_row(
        identity={},
        positions=positions,
        target_grid=np.asarray(arrays["target_grid"]),
        x_limits=tuple(map(float, config.controller.workspace.x_limits)),
        y_limits=tuple(map(float, config.controller.workspace.y_limits)),
        reachable_mask=np.asarray(arrays["reachable_mask"]),
        gmm_means=np.asarray(config.controller.gmm.means),
        gmm_inverses=np.asarray(config.controller.gmm.covariance_inverse),
        delta_t=delta_t,
        occupancy=np.asarray(arrays["occupancy"]),
        grid_origin=tuple(map(float, np.asarray(arrays["grid_origin"]))),
        grid_resolution=float(arrays["grid_resolution"]),
        robot_radius=float(manifest.get("robot_radius", 0.30)),
        guard_states=np.full(steps, "pass"),
        guard_period=delta_t,
        speeds=np.linalg.norm(velocities, axis=1),
        actual_times=np.arange(steps) * delta_t,
        commanded_times=np.arange(steps) * delta_t,
        commanded=positions,
        step_ms=np.zeros(0),
        deadline_ms=float(manifest.get("deadline_ms", 16.0)),
        wall_seconds=wall,
        odometry_seconds=steps * delta_t,
        control_seconds=wall,
    )
    settled = slice(-min(5000, steps), None)
    ess = np.asarray(ess_fractions)[settled]
    temperature = np.asarray(temperatures)[settled]
    cap = config.controller.mppi.temperature_max
    return {
        "map_seed": int(manifest["map_seed"]), "seed": seed, "steps": steps,
        "preflight_steps": PREFLIGHT_STEPS,
        "horizon": config.controller.mppi.horizon,
        "samples": config.controller.mppi.samples,
        "lam_max": cap, "ess_target": config.controller.mppi.ess_target,
        "ess_settled_median": float(np.median(ess)),
        "ess_settled_mean": float(np.mean(ess)),
        "temperature_settled_median": float(np.median(temperature)),
        "temperature_cap_fraction": float(np.mean(temperature >= cap * (1.0 - 1e-6))),
        "occupancy_mse": row["occupancy_mse"],
        "fourier_ergodic": row["fourier_ergodic"],
        "all_modes_reached": int(np.isfinite(row["first_all_modes_s"])),
        "first_all_modes_s": row["first_all_modes_s"],
        "mode_visits": row["mode_visits"], "mode_cycles": row["mode_cycles"],
        "mode_dwell_median_s": row["mode_dwell_median_s"],
        "in_mode_fraction": row["in_mode_fraction"],
        "collisions": row["collisions"], "min_clearance_m": row["min_clearance_m"],
        "x_max": float(positions[:, 0].max()),
        "path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "wall_seconds": wall, "device": device, "jax_version": jax.__version__,
    }


def run_cell(config, arrays, manifest: dict, seed: int, steps: int, overrides: dict,
             device: str) -> dict[str, object]:
    """Run and score one ideal pillar-tuning cell with ESS diagnostics."""
    config = _apply(config, overrides)
    config = replace(config, run=replace(config.run, seed=seed, steps=steps))
    started = time.perf_counter()
    result = run_simulation(
        config,
        device=device,
        initial_state=np.asarray(arrays["initial_state"]),
        preflight_steps=PREFLIGHT_STEPS,
    )
    wall = time.perf_counter() - started
    return score_run(
        config, arrays, manifest, seed, steps,
        positions=np.asarray(result.paths[:, 0, :2]),
        velocities=np.asarray(result.paths[:, 0, 2:4]),
        ess_fractions=result.ess_fractions,
        temperatures=result.temperatures,
        wall=wall, device=result.device,
    )


def run_stage(run_directory: Path, output: Path, stage: str, base_arm: str,
              winner: str, first_seed: int, seeds: int, steps: int, device: str,
              wanted: set[str], cap_output: Path | None = None,
              cap_steps: int = 10000) -> None:
    """Resume one stage, stopping cells that miss the optional visitation cap."""
    if cap_output is not None and not 0 < cap_steps < steps:
        raise ValueError("cap_steps must be positive and smaller than steps")
    config, manifest, arrays = _grid_config(run_directory)
    completed = set()
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            completed = {
                (row["stage"], row["arm"], int(row["map_seed"]), int(row["seed"]), int(row["steps"]))
                for row in csv.DictReader(stream)
            }
    capped = {}
    if cap_output is not None and cap_output.exists():
        with cap_output.open(encoding="utf-8", newline="") as stream:
            capped = {
                (row["stage"], row["arm"], int(row["map_seed"]), int(row["seed"]), int(row["steps"])): row
                for row in csv.DictReader(stream)
            }
    for arm, overrides in stage_arms(stage, base_arm, winner).items():
        if wanted and arm not in wanted:
            continue
        axis, value, _ = ABLATION_ARMS.get(arm, ("", "", {}))
        for seed in range(first_seed, first_seed + seeds):
            identity = (stage, arm, int(manifest["map_seed"]), seed, steps)
            if cap_output is not None:
                cap_identity = (stage, arm, int(manifest["map_seed"]), seed, cap_steps)
                cap_row = capped.get(cap_identity)
                if cap_row is None:
                    cap_row = run_cell(
                        config, arrays, manifest, seed, cap_steps, overrides, device
                    )
                    cap_row.update({
                        "stage": stage, "arm": arm, "axis": axis, "value": value,
                        "configuration": json.dumps(
                            overrides, sort_keys=True, separators=(",", ":")
                        ),
                        "accepted": int(cap_row["all_modes_reached"]),
                    })
                    append_csv(
                        cap_output,
                        {field: cap_row.get(field, "") for field in CAP_FIELDS},
                        CAP_FIELDS,
                    )
                if not int(cap_row["accepted"]):
                    print(
                        f"DISCARD={stage}/{arm}/m{manifest['map_seed']}/s{seed} "
                        f"no full tour by {cap_steps} steps",
                        flush=True,
                    )
                    continue
            if identity in completed:
                print(f"SKIP stage={stage} arm={arm} map={manifest['map_seed']} seed={seed}")
                continue
            row = run_cell(config, arrays, manifest, seed, steps, overrides, device)
            row.update({
                "stage": stage,
                "arm": arm,
                "axis": axis,
                "value": value,
                "configuration": json.dumps(overrides, sort_keys=True, separators=(",", ":")),
            })
            append_csv(output, {field: row.get(field, "") for field in FIELDS}, FIELDS)
            print(f"ROW={stage}/{arm}/m{manifest['map_seed']}/s{seed} {row['wall_seconds']:.1f}s", flush=True)


def map_geometry(root: Path) -> tuple[dict, str]:
    """Read the obstacle geometry and inflation this campaign's maps were built with.

    Both are campaign variables now -- the same arms run at more than one obstacle density
    -- so the reports read them from the archive instead of restating a constant.
    """
    maps = sorted((root / "maps").glob("map_*/manifest.json")) if (root / "maps").exists() else []
    if not maps:
        return {}, "configured"
    geometry = json.loads(maps[0].read_text(encoding="utf-8")).get("map_parameters", {})
    return geometry, inflation_text(root / "maps", [path.parent.name for path in maps])


def measure_map(directory: Path) -> dict[str, object]:
    """Measure one archived map: free space, and how much target mass is out of reach.

    Ergodic coverage compares the time-averaged occupancy against the target, so target
    mass sitting inside inflated obstacles is error the vehicle is pulled toward and can
    never retire. At 45 pillars and a 1.04 m inflation it was 22.9%, which is what produced
    the observed local minima.

    Reports the aggregate *and* the per-mode worst case, because the aggregate hides the
    thing that actually breaks a run. On map 539 the total was 10.1% -- the best of fifteen
    candidates -- while one mode alone was 22.5% obstructed, and two flights on it missed
    that exact mode and no other. A mode the vehicle cannot cover is a failed tour whatever
    the average over the other two says.

    Needs the archived ``arrays.npz``, so it runs on a generated map rather than on the
    six-second qualification probe -- the probe never builds one.
    """
    from ergodic_control_mppi.deploy.grid import reachable_from

    gmm = load_config("configs/uav_profile.yaml").controller.gmm
    means = np.asarray(gmm.means)
    inverses = np.asarray(gmm.covariance_inverse)
    weights = np.exp(np.asarray(gmm.log_weights))
    normalizers = np.exp(np.asarray(gmm.log_normalizers))

    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        grid = np.asarray(arrays["grid"])
        origin = tuple(float(v) for v in np.asarray(arrays["grid_origin"]))
        resolution = float(arrays["grid_resolution"])
        start = tuple(float(v) for v in np.asarray(arrays["initial_state"])[:2])
    height, width = grid.shape
    rows_index, columns_index = np.mgrid[0:height, 0:width]
    x = origin[0] + (columns_index + 0.5) * resolution
    y = origin[1] + (rows_index + 0.5) * resolution
    blocked = grid > 1e-6
    visited = reachable_from(blocked.astype(np.float32), origin, resolution, start)

    density = np.zeros(grid.shape)
    per_mode = []
    for index in range(means.shape[0]):
        dx, dy = x - means[index, 0], y - means[index, 1]
        quadratic = (
            inverses[index, 0, 0] * dx * dx
            + 2 * inverses[index, 0, 1] * dx * dy
            + inverses[index, 1, 1] * dy * dy
        )
        lobe = weights[index] * normalizers[index] * np.exp(-0.5 * quadratic)
        density += lobe
        # Normalized within the lobe, so this is "how much of *this mode* is out of reach"
        # and does not shrink just because the mode carries little of the total weight.
        per_mode.append(float((lobe / lobe.sum())[~visited].sum()))
    density /= density.sum()
    return {
        "map": directory.name.removeprefix("map_"),
        "free_fraction": float(1.0 - blocked.mean()),
        "unreachable_target_mass": float(density[~visited].sum()),
        "per_mode_unreachable": per_mode,
        "worst_mode_unreachable": max(per_mode),
    }


def map_diagnostics(root: Path) -> list[dict[str, object]]:
    """Measure every archived map under ``root``, for the report's geometry table."""
    return [
        measure_map(path.parent)
        for path in sorted((root / "maps").glob("map_*/manifest.json"))
        if (path.parent / "arrays.npz").exists()
    ]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _paired_ratio(rows, arm: str, reference: str, field: str = "occupancy_mse") -> float:
    keyed = {(row["arm"], int(row["map_seed"]), int(row["seed"])): row for row in rows}
    ratios = []
    for _, map_seed, seed in sorted(key for key in keyed if key[0] == arm):
        other = keyed.get((reference, map_seed, seed))
        if other:
            ratios.append(float(keyed[(arm, map_seed, seed)][field]) / float(other[field]))
    return float(np.median(ratios)) if ratios else float("nan")


def _arm_summary(rows, arm: str) -> dict[str, object]:
    mine = [row for row in rows if row["arm"] == arm]
    reached = sum(int(row["all_modes_reached"]) for row in mine)
    repeats = sum(float(row["mode_cycles"]) >= 1 for row in mine)
    times = [float(row["first_all_modes_s"]) for row in mine if np.isfinite(float(row["first_all_modes_s"]))]
    return {
        "runs": len(mine), "reached": reached, "repeats": repeats,
        "first_median": float(np.median(times)) if times else float("inf"),
        "ess_median": float(np.median([float(row["ess_settled_median"]) for row in mine])) if mine else float("nan"),
        "cap_max": max((float(row["temperature_cap_fraction"]) for row in mine), default=float("nan")),
        "collisions": sum(int(row["collisions"]) for row in mine),
    }


def _capped_summary(cap_rows, full_rows, arm: str) -> dict[str, object]:
    """Summarize all cap attempts while taking repeat counts from surviving full runs."""
    summary = _arm_summary(cap_rows, arm)
    full = [row for row in full_rows if row["arm"] == arm]
    summary["full_runs"] = len(full)
    summary["repeats"] = sum(float(row["mode_cycles"]) >= 1 for row in full)
    return summary


def evaluate(root: Path) -> dict[str, object]:
    """Apply the preregistered screen and development gates."""
    rows = _read(root / "offline.csv")
    cap_rows = _read(root / "run_cap.csv") or rows
    screen = [row for row in rows if row["stage"] == "screen"]
    screen_cap = [row for row in cap_rows if row["stage"] == "screen"]
    reference = _capped_summary(screen_cap, screen, "common_cap")
    eligible = []
    screen_summary = {}
    for arm in SCREEN_ARMS:
        summary = _capped_summary(screen_cap, screen, arm)
        summary["mse_ratio"] = _paired_ratio(screen_cap, arm, "common_cap")
        summary["eligible"] = bool(
            summary["runs"] == 6
            and summary["full_runs"] == summary["reached"]
            and summary["collisions"] == 0
            and summary["cap_max"] < 0.01
            and 0.20 <= summary["ess_median"] <= 0.40
            and summary["mse_ratio"] <= 1.25
            and (
                summary["reached"] >= reference["reached"] + 2
                or summary["repeats"] >= reference["repeats"] + 2
            )
        )
        if summary["eligible"]:
            eligible.append(arm)
        screen_summary[arm] = summary
    screen_winner = (
        sorted(
            eligible,
            key=lambda arm: (
                -screen_summary[arm]["reached"], -screen_summary[arm]["repeats"],
                screen_summary[arm]["first_median"], screen_summary[arm]["mse_ratio"],
                len(SCREEN_ARMS[arm]),
            ),
        )[0]
        if eligible else "shipped"
    )

    approach = [row for row in rows if row["stage"] == "approach"]
    approach_cap = [row for row in cap_rows if row["stage"] == "approach"]
    maps = sorted({int(row["map_seed"]) for row in approach_cap})
    approach_summary = {}
    eligible = []
    for arm in APPROACH_ARMS:
        summary = _capped_summary(approach_cap, approach, arm)
        per_map = {
            map_seed: _capped_summary(
                [row for row in approach_cap if int(row["map_seed"]) == map_seed],
                [row for row in approach if int(row["map_seed"]) == map_seed], arm
            )
            for map_seed in maps
        }
        summary["per_map"] = per_map
        summary["mse_ratio"] = _paired_ratio(approach_cap, arm, "base")
        summary["eligible"] = bool(
            len(maps) == 3
            and all(
                value["runs"] == 6
                and value["reached"] >= 4
                and value["full_runs"] == value["reached"]
                for value in per_map.values()
            )
            and summary["collisions"] == 0
            and summary["cap_max"] < 0.01
            and 0.20 <= summary["ess_median"] <= 0.40
            and summary["mse_ratio"] <= 1.25
        )
        if summary["eligible"]:
            eligible.append(arm)
        approach_summary[arm] = summary
    approach_winner = (
        sorted(
            eligible,
            key=lambda arm: (
                -min(value["reached"] for value in approach_summary[arm]["per_map"].values()),
                -min(value["repeats"] for value in approach_summary[arm]["per_map"].values()),
                approach_summary[arm]["first_median"], len(APPROACH_ARMS[arm]),
            ),
        )[0]
        if eligible else ""
    )

    holdout = [row for row in rows if row["stage"] == "holdout" and row["arm"] == "winner"]
    holdout_cap = [
        row for row in cap_rows if row["stage"] == "holdout" and row["arm"] == "winner"
    ]
    holdout_maps = sorted({int(row["map_seed"]) for row in holdout_cap})
    holdout_per_map = {
        seed: _capped_summary(
            [row for row in holdout_cap if int(row["map_seed"]) == seed],
            [row for row in holdout if int(row["map_seed"]) == seed], "winner"
        )
        for seed in holdout_maps
    }
    holdout_pass = bool(
        len(holdout_maps) == 3
        and all(
            value["runs"] == 18
            and value["reached"] >= 15
            and value["full_runs"] == value["reached"]
            and value["collisions"] == 0
            and value["cap_max"] < 0.01
            and 0.20 <= value["ess_median"] <= 0.40
            for value in holdout_per_map.values()
        )
    )
    repeat_pass = holdout_pass and all(value["repeats"] >= 9 for value in holdout_per_map.values())
    return {
        "screen_winner": screen_winner,
        "screen": screen_summary,
        "approach_winner": approach_winner,
        "approach": approach_summary,
        "holdout_pass": holdout_pass,
        "repeat_pass": repeat_pass,
        "holdout": holdout_per_map,
    }


def build_report(root: Path) -> str:
    """Render a compact gate report from the dedicated tuning CSV."""
    result = evaluate(root)
    selection_path = root / "selection.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.exists() else {}
    )
    geometry, inflation = map_geometry(root)
    lines = [
        "# Pillar tuning", "",
        "Controller seeds are repeated trials within maps; maps are reported separately.", "",
        f"{geometry_sentence(geometry)} The safety inflation is {inflation}.",
        f"Development maps: **{selection.get('development', 'pending')}**; holdout maps: "
        f"**{selection.get('holdout', 'pending')}**. Selection is geometry-only.", "",
        "Every cell is evaluated at 10,000 steps. A cell without one dwell-qualified "
        "visit to each target mode stops there; survivors rerun deterministically to "
        "20,000 steps for repeat-loop evidence. Rejected 10k rows remain archived.", "",
        "## T/K/cap falsification screen", "",
        "| arm | 10k attempts | 10k tours | 20k runs | second tours | ESS median | cap max | MSE ratio | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm, summary in result["screen"].items():
        lines.append(
            f"| {arm} | {summary['runs']} | {summary['reached']} | {summary['full_runs']} | "
            f"{summary['repeats']} | "
            f"{summary['ess_median']:.3f} | {summary['cap_max']:.3f} | "
            f"{summary['mse_ratio']:.3f} | {'yes' if summary['eligible'] else 'no'} |"
        )
    lines.extend([
        "", f"Screen base: **{result['screen_winner']}**.", "",
        "## Development gate", "",
        "| arm | 10k attempts | 10k tours | 20k runs | second tours | ESS median | cap max | MSE ratio | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for arm, summary in result["approach"].items():
        lines.append(
            f"| {arm} | {summary['runs']} | {summary['reached']} | "
            f"{summary['full_runs']} | {summary['repeats']} | "
            f"{summary['ess_median']:.3f} | {summary['cap_max']:.3f} | "
            f"{summary['mse_ratio']:.3f} | {'yes' if summary['eligible'] else 'no'} |"
        )
    lines.extend([
        "", "| arm | map | 10k attempts | 10k tours | 20k runs | second tours |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm, summary in result["approach"].items():
        for map_seed, per_map in summary["per_map"].items():
            lines.append(
                f"| {arm} | {map_seed} | {per_map['runs']} | {per_map['reached']} | "
                f"{per_map['full_runs']} | {per_map['repeats']} |"
            )
    approach_attempts = sum(summary["runs"] for summary in result["approach"].values())
    holdout_attempts = sum(summary["runs"] for summary in result["holdout"].values())
    if approach_attempts and not result["approach_winner"]:
        approach_status = "none — development gate failed"
        holdout_status = repeat_status = "NOT RUN"
    else:
        approach_status = result["approach_winner"] or "pending"
        holdout_status = (
            "PASS" if result["holdout_pass"] else "FAIL" if holdout_attempts else "PENDING"
        )
        repeat_status = (
            "PASS" if result["repeat_pass"] else "FAIL" if holdout_attempts else "PENDING"
        )
    lines.extend([
        "", f"Approach winner: **{approach_status}**.",
        f"Holdout primary: **{holdout_status}**.",
        f"Holdout repeatability: **{repeat_status}**.",
    ])
    if approach_attempts and not result["approach_winner"]:
        lines.extend([
            "", "**Negative result:** no approach arm achieved at least four of six 10k "
            "tours on every development map. Holdout and online flights were not run.",
        ])
    if holdout_attempts:
        lines.extend([
            "", "| holdout map | 10k attempts | 10k tours | 20k runs | second tours | ESS median | cap max | collisions |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for map_seed, summary in result["holdout"].items():
            lines.append(
                f"| {map_seed} | {summary['runs']} | {summary['reached']} | "
                f"{summary['full_runs']} | {summary['repeats']} | "
                f"{summary['ess_median']:.3f} | {summary['cap_max']:.3f} | "
                f"{summary['collisions']} |"
            )
    lines.extend([
        "", "No waypoint guidance, replay model, or controller-performance map selection is used.",
        "The online-versus-offline visitation root cause remains unresolved; these gates only "
        "test whether the gap persists.",
    ])
    online = _read(root / "summary.csv")
    uav = [row for row in online if row["mode"] == "uav"]
    ideal = [row for row in online if row["mode"] == "ideal"]
    if uav or ideal:
        reached = sum(np.isfinite(float(row["first_all_modes_s"])) for row in uav)
        repeats = sum(float(row["mode_cycles"]) >= 1 for row in uav)
        altitude_p95 = []
        for row in uav:
            arrays_path = root / row["run_id"] / "arrays.npz"
            if arrays_path.exists():
                with np.load(arrays_path, allow_pickle=False) as arrays:
                    altitude_p95.append(
                        float(np.percentile(abs(np.asarray(arrays["odometry"])[:, 3] - 0.75), 95))
                    )
        safety = bool(uav) and all(
            int(row["collisions"]) == 0
            and float(row["step_p99_ms"]) < 16.0
            and float(row["deadline_miss_fraction"]) < 0.001
            and float(row["guard_fraction"]) < 0.01
            and 49.0 <= float(row["achieved_rate_hz"]) <= 51.0
            for row in uav
        )
        planar = bool(altitude_p95) and max(altitude_p95) <= 0.05
        lines.extend([
            "", "## Online SO3 gate", "",
            f"Flights/twins: **{len(uav)}/{len(ideal)}**; initial tours: **{reached}/5**; "
            f"second tours: **{repeats}/5**.",
            f"Safety/timing: **{'PASS' if safety else 'FAIL'}**; altitude planarity: "
            f"**{'PASS' if planar else 'FAIL'}**.",
            f"Operational visitation: **{'PASS' if reached >= 4 else 'FAIL'}**; strong "
            f"repeatability: **{'PASS' if repeats >= 3 else 'FAIL'}**.",
        ])
    return "\n".join(lines) + "\n"


def sweep_summary(root: Path) -> list[dict[str, object]]:
    """Score every sweep arm on continuous outcomes, ranked by tour rate.

    The 10k cap layer is the paired one: it holds a row for every (arm, seed) whether or
    not the cell toured, so ratios against ``baseline`` are computed there. Tour *rate*
    can only come from the gated 20k layer, and is NaN for an arm nothing survived.
    """
    delta_t = load_config("configs/uav_profile.yaml").controller.model.delta_t
    cap = [row for row in _read(root / "sweep_cap.csv") if row["stage"] == "sweep"]
    full = [row for row in _read(root / "sweep.csv") if row["stage"] == "sweep"]
    summaries = []
    for arm in sorted({row["arm"] for row in cap}):
        mine = [row for row in cap if row["arm"] == arm]
        survivors = [row for row in full if row["arm"] == arm]
        rates = [
            (1.0 + float(row["mode_cycles"])) / (int(row["steps"]) * delta_t / 1000.0)
            for row in survivors
        ]
        summaries.append({
            "arm": arm,
            "axis": mine[0]["axis"],
            "value": mine[0]["value"],
            "runs": len(mine),
            "tours": sum(int(row["accepted"]) for row in mine),
            "survivors": len(survivors),
            # Primary score: full sweeps of all three modes per 1000 simulated seconds.
            "tour_rate": float(np.median(rates)) if rates else float("nan"),
            "dwell": float(np.median([float(row["mode_dwell_median_s"]) for row in mine])),
            "in_mode": float(np.median([float(row["in_mode_fraction"]) for row in mine])),
            "speed": float(np.median([
                float(row["path_length_m"]) / (int(row["steps"]) * delta_t) for row in mine
            ])),
            "ess": float(np.median([float(row["ess_settled_median"]) for row in mine])),
            "cap_max": max(float(row["temperature_cap_fraction"]) for row in mine),
            "mse_ratio": _paired_ratio(cap, arm, "baseline"),
            "fourier_ratio": _paired_ratio(cap, arm, "baseline", "fourier_ergodic"),
            "collisions": sum(int(row["collisions"]) for row in mine),
            "clearance": min(float(row["min_clearance_m"]) for row in mine),
        })
    # Rank on the tour rate, then on the shortest dwell -- the exit is the failing verb, so
    # an arm that tours as often with briefer parks is the better one for the figure.
    return sorted(
        summaries,
        key=lambda s: (
            -(s["tour_rate"] if np.isfinite(s["tour_rate"]) else -1.0),
            -s["tours"],
            s["dwell"],
        ),
    )


def build_sweep_report(root: Path) -> str:
    """Render the one-factor sweep as a ranked effect table."""
    summaries = sweep_summary(root)
    baseline = next((s for s in summaries if s["arm"] == "baseline"), None)
    seeds = baseline["runs"] if baseline else 0
    geometry, inflation = map_geometry(root)
    lines = [
        "# One-factor sweep on the pillar map", "",
        # The same arm table runs at more than one obstacle density, so which one this is
        # has to be on the page: the two reports are only comparable once it is stated.
        f"{geometry_sentence(geometry)} The safety inflation is {inflation}.", "",
        "One axis moved at a time from the shipped profile; `baseline` is that profile "
        f"unmodified. {seeds} controller seeds per arm on one map, so these are repeated "
        "trials in one environment, not independent environments.", "",
        "Every cell runs 10,000 steps. That layer is complete and balanced -- one row per "
        "(arm, seed) whether or not the cell toured -- and is the layer the paired ratios "
        "and the second ablation campaign are computed on. Cells that reached all three "
        "modes rerun deterministically to 20,000 steps; only those carry a tour rate.", "",
        "Ranked by tour rate. ESS and cap fraction are reported, not enforced: in the "
        "preceding campaign a temperature-clipping veto removed the best-touring arm.", "",
        "| arm | axis | value | 10k tours | 20k runs | tours/1000 s | dwell s | in mode | "
        "speed m/s | ESS | cap | MSE ratio | Fourier ratio | collisions |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: |",
    ]
    for summary in summaries:
        rate = summary["tour_rate"]
        lines.append(
            f"| {summary['arm']} | {summary['axis']} | {summary['value']} | "
            f"{summary['tours']}/{summary['runs']} | {summary['survivors']} | "
            f"{'--' if not np.isfinite(rate) else f'{rate:.2f}'} | "
            f"{summary['dwell']:.1f} | {summary['in_mode']:.2f} | {summary['speed']:.3f} | "
            f"{summary['ess']:.3f} | {summary['cap_max']:.3f} | "
            f"{summary['mse_ratio']:.3f} | {summary['fourier_ratio']:.3f} | "
            f"{summary['collisions']} |"
        )
    if baseline:
        lines.extend([
            "", f"Baseline: {baseline['tours']}/{baseline['runs']} tours, dwell "
            f"{baseline['dwell']:.1f} s, {baseline['speed']:.3f} m/s, ESS "
            f"{baseline['ess']:.3f}, cap fraction {baseline['cap_max']:.3f}.",
        ])
    diagnostics = map_diagnostics(root)
    if diagnostics:
        lines.extend([
            "", "## Geometry", "",
            "Target mass inside inflated obstacles is error the vehicle is pulled toward "
            "and can never retire, so it is reported alongside free space rather than left "
            "implicit in the pillar count.", "",
            "| map | free fraction | target mass out of reach | worst single mode |",
            "| --- | ---: | ---: | ---: |",
        ])
        lines.extend(
            f"| {row['map']} | {row['free_fraction']:.3f} | "
            f"{100 * row['unreachable_target_mass']:.1f}% | "
            f"{100 * row['worst_mode_unreachable']:.1f}% |"
            for row in diagnostics
        )
    clearance = min((s["clearance"] for s in summaries), default=float("nan"))
    collisions = sum(s["collisions"] for s in summaries)
    lines.extend([
        "", f"Safety across the whole sweep: {collisions} collisions, minimum clearance "
        f"{clearance:.2f} m.", "",
        "Ratios are paired per seed against `baseline` on the 10k layer; below 1 is better. "
        "Significance is not claimed here -- `scripts/report_figures.py` runs the paired "
        "Wilcoxon with per-axis Holm correction over the same rows.",
    ])
    return "\n".join(lines) + "\n"


def write_ablation_copy(root: Path, output: Path) -> Path:
    """Export the balanced 10k sweep layer as a second UAV ablation campaign.

    Columns lead with the ablation schema so ``scripts/report_figures.py`` reads this the
    same way it reads ``results/uav/ablation.csv``; the ESS and temperature diagnostics
    this campaign also carries are appended and simply ignored there.

    Raises:
        ValueError: if the arms do not share one seed set. An unbalanced arm would
            silently shrink the paired Wilcoxon rather than fail, so it is caught here.
    """
    rows = [row for row in _read(root / "sweep_cap.csv") if row["stage"] == "sweep"]
    if not rows:
        raise ValueError(f"no sweep rows in {root / 'sweep_cap.csv'}")
    seeds = {arm: set() for arm in {row["arm"] for row in rows}}
    for row in rows:
        seeds[row["arm"]].add(int(row["seed"]))
    expected = seeds.get("baseline")
    if expected is None:
        raise ValueError("the sweep has no `baseline` arm to pair against")
    unbalanced = sorted(arm for arm, values in seeds.items() if values != expected)
    if unbalanced:
        raise ValueError(f"arms not paired over the baseline seeds {sorted(expected)}: {unbalanced}")
    extra = [field for field in CAP_FIELDS if field not in ABLATION_FIELDS]
    fields = ABLATION_FIELDS + extra
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return output


def dry_run() -> dict[str, int]:
    """Return the gated maximum row count without running a controller."""
    return {
        "screen_cap": len(SCREEN_ARMS) * 6,
        "screen_full_max": len(SCREEN_ARMS) * 6,
        "approach_cap": len(APPROACH_ARMS) * 3 * 6,
        "approach_full_max": len(APPROACH_ARMS) * 3 * 6,
        "holdout_cap": 2 * 3 * 18,
        "holdout_full_max": 2 * 3 * 18,
        "sweep_cap": len(SWEEP_ARMS) * 12,
        "sweep_full_max": len(SWEEP_ARMS) * 12,
    }


def _json_ready(value):
    """Replace non-finite report sentinels with JSON null recursively."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# Override name -> the YAML path it is written to. Kept beside ``_apply``'s coverage: an
# override the runner can honour but this cannot write would tune something the flight
# could never reproduce, so ``write_profile`` refuses rather than dropping it.
PROFILE_KEYS = {
    "T": ("mppi", "T"), "K": ("mppi", "K"),
    "lam_max": ("mppi", "lam_max"), "exploration": ("mppi", "exploration"),
    "alpha": ("mppi", "alpha"),
    "memory_time": ("stein", "memory_time"),
    "fine_bandwidth": ("stein", "fine_bandwidth"),
    "coarse_bandwidth": ("stein", "coarse_bandwidth"),
    "memory_scales": ("stein", "memory_scales"),
    "memory_gain": ("stein", "memory_gain"),
    "memory_balance": ("stein", "memory_balance"),
    "theta": ("stein", "theta"),
    "flow_weight": ("stein", "weight_stein"),
    "self_bandwidth": ("stein", "ell_self"),
    "reference_speed": ("stein", "reference_speed"),
}


def write_profile(overrides: dict, output: Path) -> Path:
    """Write ``configs/uav_profile.yaml`` with one arm's overrides applied.

    ``penalty_scale`` and ``boundary_scale`` are multipliers in ``_apply``, so they are
    applied here as multipliers too rather than written as literals.
    """
    data = yaml.safe_load(Path("configs/uav_profile.yaml").read_text(encoding="utf-8"))
    overrides = dict(overrides)
    for name, targets in (
        ("penalty_scale", (("map", "oom_cost"), ("map", "boundary_weight"))),
        ("boundary_scale", (("map", "boundary_weight"),)),
    ):
        if name in overrides:
            scale = overrides.pop(name)
            for section, key in targets:
                data[section][key] = float(data[section][key]) * scale
            if name == "penalty_scale":
                data["map"]["obstacles"]["weight"] = (
                    float(data["map"]["obstacles"]["weight"]) * scale
                )
    unknown = sorted(set(overrides) - set(PROFILE_KEYS))
    if unknown:
        raise ValueError(f"no profile key for overrides: {unknown}")
    for name, value in overrides.items():
        section, key = PROFILE_KEYS[name]
        data[section][key] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def write_winner_config(root: Path, output: Path) -> Path:
    """Materialize the holdout-accepted pillar profile for online flights."""
    result = evaluate(root)
    if not result["holdout_pass"] or not result["approach_winner"]:
        raise ValueError("holdout has not accepted a pillar profile")
    return write_profile(
        _merge(
            SCREEN_ARMS[result["screen_winner"]],
            APPROACH_ARMS[result["approach_winner"]],
        ),
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--qualification", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument(
        "--candidates", type=Path, default=None,
        help="Campaign root whose generated maps rank the qualifying seeds by "
             "out-of-reach target mass; without it the order is by seed",
    )
    run = sub.add_parser("run")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument(
        "--stage", required=True, choices=("screen", "approach", "holdout", "sweep")
    )
    run.add_argument("--base-arm", default="shipped", choices=tuple(SCREEN_ARMS))
    run.add_argument("--winner", default="base", choices=tuple(APPROACH_ARMS))
    run.add_argument("--first-seed", type=int, default=43)
    run.add_argument("--seeds", type=int, default=6)
    run.add_argument("--steps", type=int, default=20000)
    run.add_argument("--cap-output", type=Path)
    run.add_argument("--cap-steps", type=int, default=10000)
    run.add_argument("--device", default="gpu", choices=("auto", "cpu", "gpu"))
    run.add_argument("--arms", default="")
    report = sub.add_parser("report")
    report.add_argument("--root", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    sweep = sub.add_parser("sweep-report")
    sweep.add_argument("--root", required=True, type=Path)
    sweep.add_argument("--output", required=True, type=Path)
    # The ablation copy is the same 10k rows under the name the figure script expects.
    sweep.add_argument("--ablation", type=Path)
    value = sub.add_parser("value")
    value.add_argument("--path", required=True, type=Path)
    value.add_argument("--field", required=True)
    config = sub.add_parser("config")
    config.add_argument("--root", required=True, type=Path)
    config.add_argument("--output", required=True, type=Path)
    # A sweep arm is materialized by name and is not gated: the sweep is exploratory, and
    # the flight it feeds is a figure, not an accepted claim.
    config.add_argument("--arm", choices=SWEEP_ARMS)
    sub.add_parser("dry-run")
    arguments = parser.parse_args()
    if arguments.command == "select":
        print(json.dumps(write_split_selection(
            arguments.qualification, arguments.output,
            candidate_blocked_mass(arguments.candidates) if arguments.candidates else None,
        )))
    elif arguments.command == "run":
        run_stage(
            arguments.run_dir, arguments.output, arguments.stage, arguments.base_arm,
            arguments.winner, arguments.first_seed, arguments.seeds, arguments.steps,
            arguments.device, set(filter(None, arguments.arms.split(","))),
            arguments.cap_output, arguments.cap_steps,
        )
    elif arguments.command == "report":
        arguments.output.write_text(build_report(arguments.root), encoding="utf-8")
        (arguments.root / "gate.json").write_text(
            json.dumps(_json_ready(evaluate(arguments.root)), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.output}")
    elif arguments.command == "sweep-report":
        arguments.output.write_text(build_sweep_report(arguments.root), encoding="utf-8")
        print(f"wrote {arguments.output}")
        if arguments.ablation:
            written = write_ablation_copy(arguments.root, arguments.ablation)
            print(f"wrote {written}")
    elif arguments.command == "value":
        value = json.loads(arguments.path.read_text(encoding="utf-8"))[arguments.field]
        print(" ".join(map(str, value)) if isinstance(value, list) else value)
    elif arguments.command == "config":
        print(
            write_profile(ABLATION_ARMS[arguments.arm][2], arguments.output)
            if arguments.arm
            else write_winner_config(arguments.root, arguments.output)
        )
    else:
        print(json.dumps(dry_run(), indent=2))


if __name__ == "__main__":
    main()

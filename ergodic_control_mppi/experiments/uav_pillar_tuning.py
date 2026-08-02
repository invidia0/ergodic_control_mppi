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
from ergodic_control_mppi.experiments.uav_ablation import _apply
from ergodic_control_mppi.experiments.uav_pillars import QUALIFICATION_FIELDS, select_maps
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
FIELDS = [
    "stage", "arm", "configuration", "map_seed", "seed", "steps",
    "preflight_steps", "horizon", "samples", "lam_max", "ess_target",
    "ess_settled_median", "ess_settled_mean", "temperature_settled_median",
    "temperature_cap_fraction", "occupancy_mse", "fourier_ergodic",
    "all_modes_reached", "first_all_modes_s", "mode_visits", "mode_cycles",
    "mode_dwell_median_s", "in_mode_fraction", "collisions", "min_clearance_m",
    "x_max", "path_length_m", "wall_seconds", "device", "jax_version",
]
CAP_FIELDS = FIELDS + ["accepted"]


def select_split(rows: list[dict[str, str]]) -> dict[str, object]:
    """Select the first three development and next three holdout maps geometrically."""
    selected, _ = select_maps(rows, count=6)
    by_seed = {int(row["map_seed"]): row for row in rows}
    development, holdout = selected[:3], selected[3:]

    def representative(seeds):
        return sorted(
            seeds, key=lambda seed: (float(by_seed[seed]["free_fraction"]), seed)
        )[1]

    return {
        "development": development,
        "holdout": holdout,
        "development_representative": representative(development),
        "holdout_representative": representative(holdout),
    }


def write_split_selection(qualification: Path, output: Path) -> dict[str, object]:
    """Write the immutable six-map development/holdout selection."""
    with qualification.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selection = select_split(rows)
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


def _grid_config(run_directory: Path):
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(run_directory / "arrays.npz", allow_pickle=False)
    config = load_config("configs/uav_profile.yaml")
    workspace = replace(
        config.controller.workspace,
        grid=jnp.asarray(np.asarray(arrays["grid"], dtype=np.float32)),
        grid_origin=jnp.asarray(arrays["grid_origin"], dtype=jnp.float32),
        grid_resolution=float(arrays["grid_resolution"]),
    )
    return replace(config, controller=replace(config.controller, workspace=workspace)), manifest, arrays


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
    positions = np.asarray(result.paths[:, 0, :2])
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
        speeds=np.linalg.norm(result.paths[:, 0, 2:4], axis=1),
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
    ess = result.ess_fractions[settled]
    temperature = result.temperatures[settled]
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
        "wall_seconds": wall, "device": result.device, "jax_version": jax.__version__,
    }


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
                        "stage": stage, "arm": arm,
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
                "configuration": json.dumps(overrides, sort_keys=True, separators=(",", ":")),
            })
            append_csv(output, {field: row.get(field, "") for field in FIELDS}, FIELDS)
            print(f"ROW={stage}/{arm}/m{manifest['map_seed']}/s{seed} {row['wall_seconds']:.1f}s", flush=True)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _paired_ratio(rows, arm: str, reference: str) -> float:
    keyed = {(row["arm"], int(row["map_seed"]), int(row["seed"])): row for row in rows}
    ratios = []
    for _, map_seed, seed in sorted(key for key in keyed if key[0] == arm):
        other = keyed.get((reference, map_seed, seed))
        if other:
            ratios.append(float(keyed[(arm, map_seed, seed)]["occupancy_mse"]) / float(other["occupancy_mse"]))
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
    lines = [
        "# Seven-cell pillar tuning", "",
        "Controller seeds are repeated trials within maps; maps are reported separately.", "",
        "The field uses 45 vertical pillars (0.3–0.6 m radius, 2–3 m height, "
        "1.2 m minimum centre spacing, zero rings) and a 1.04 m seven-cell safety budget.",
        f"Development maps: **{selection.get('development', 'pending')}**; holdout maps: "
        f"**{selection.get('holdout', 'pending')}**. Selection is geometry-only.", "",
        "Every cell is evaluated at 10,000 steps. A cell without one complete "
        "visit–dwell–exit tour stops there; survivors rerun deterministically to 20,000 "
        "steps for repeat-loop evidence. Rejected 10k rows remain archived.", "",
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
        "## Development and holdout gates", "",
        f"Approach winner: **{result['approach_winner'] or 'none'}**.",
        f"Holdout primary: **{'PASS' if result['holdout_pass'] else 'PENDING/FAIL'}**.",
        f"Holdout repeatability: **{'PASS' if result['repeat_pass'] else 'PENDING/FAIL'}**.",
        "",
        "No waypoint guidance, replay model, or controller-performance map selection is used.",
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


def dry_run() -> dict[str, int]:
    """Return the gated maximum row count without running a controller."""
    return {
        "screen_cap": len(SCREEN_ARMS) * 6,
        "screen_full_max": len(SCREEN_ARMS) * 6,
        "approach_cap": len(APPROACH_ARMS) * 3 * 6,
        "approach_full_max": len(APPROACH_ARMS) * 3 * 6,
        "holdout_cap": 2 * 3 * 18,
        "holdout_full_max": 2 * 3 * 18,
    }


def _json_ready(value):
    """Replace non-finite report sentinels with JSON null recursively."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_winner_config(root: Path, output: Path) -> Path:
    """Materialize the holdout-accepted pillar profile for online flights."""
    result = evaluate(root)
    if not result["holdout_pass"] or not result["approach_winner"]:
        raise ValueError("holdout has not accepted a pillar profile")
    overrides = _merge(
        SCREEN_ARMS[result["screen_winner"]],
        APPROACH_ARMS[result["approach_winner"]],
    )
    data = yaml.safe_load(Path("configs/uav_profile.yaml").read_text(encoding="utf-8"))
    keys = {
        "T": ("mppi", "T"), "K": ("mppi", "K"),
        "lam_max": ("mppi", "lam_max"),
        "memory_time": ("stein", "memory_time"),
        "fine_bandwidth": ("stein", "fine_bandwidth"),
    }
    for name, value in overrides.items():
        section, key = keys[name]
        data[section][key] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--qualification", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--stage", required=True, choices=("screen", "approach", "holdout"))
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
    value = sub.add_parser("value")
    value.add_argument("--path", required=True, type=Path)
    value.add_argument("--field", required=True)
    config = sub.add_parser("config")
    config.add_argument("--root", required=True, type=Path)
    config.add_argument("--output", required=True, type=Path)
    sub.add_parser("dry-run")
    arguments = parser.parse_args()
    if arguments.command == "select":
        print(json.dumps(write_split_selection(arguments.qualification, arguments.output)))
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
    elif arguments.command == "value":
        value = json.loads(arguments.path.read_text(encoding="utf-8"))[arguments.field]
        print(" ".join(map(str, value)) if isinstance(value, list) else value)
    elif arguments.command == "config":
        print(write_winner_config(arguments.root, arguments.output))
    else:
        print(json.dumps(dry_run(), indent=2))


if __name__ == "__main__":
    main()

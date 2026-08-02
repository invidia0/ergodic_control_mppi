"""Qualify pillar maps and report the preregistered planar UAV campaign."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ergodic_control_mppi.experiments.common import append_csv

QUALIFICATION_FIELDS = [
    "map_seed",
    "status",
    "raw_occupied_cells",
    "free_fraction",
    "blocked_mode_segments",
    "qualifies",
    "selected",
    "representative",
    "log",
]


def parse_probe_log(text: str, seed: int, log: str = "") -> dict[str, object]:
    """Extract one geometry-only qualification row from a ROS launch log."""
    armed = re.findall(
        r"armed: (\d+) occupied cells, free space ([\d.]+)%, .*"
        r"blocked_mode_segments=(\d+)",
        text,
    )
    free = re.findall(r"free space ([\d.]+)%", text)
    blocked = int(armed[-1][2]) if armed else 0
    reasons = re.findall(r"\[FATAL\].*?: (.+)", text)
    return {
        "map_seed": seed,
        "status": "armed" if armed else (reasons[-1] if reasons else "probe failed"),
        "raw_occupied_cells": int(armed[-1][0]) if armed else "",
        "free_fraction": float((armed[-1][1] if armed else free[-1]) if free else 0.0) / 100.0,
        "blocked_mode_segments": blocked,
        "qualifies": int(bool(armed and blocked >= 2)),
        "selected": 0,
        "representative": 0,
        "log": log,
    }


def record_probe(log_path: Path, seed: int, output: Path) -> dict[str, object]:
    """Append one probe result unless that map seed is already archived."""
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["map_seed"]) == seed:
                    return row
    row = parse_probe_log(log_path.read_text(encoding="utf-8", errors="replace"), seed, str(log_path))
    append_csv(output, row, QUALIFICATION_FIELDS)
    return row


def select_maps(rows: list[dict[str, str]], count: int = 3) -> tuple[list[int], int]:
    """Select the first qualifying seeds and their median-free-space representative."""
    selected = sorted((row for row in rows if int(row["qualifies"])), key=lambda row: int(row["map_seed"]))[:count]
    if len(selected) != count:
        raise ValueError(f"found {len(selected)} qualifying maps; need {count}")
    representative = sorted(selected, key=lambda row: (float(row["free_fraction"]), int(row["map_seed"])))[count // 2]
    return [int(row["map_seed"]) for row in selected], int(representative["map_seed"])


def write_selection(qualification: Path, selection: Path, count: int = 3) -> dict:
    """Mark selected rows and write the immutable geometry-based selection."""
    with qualification.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seeds, representative = select_maps(rows, count)
    for row in rows:
        seed = int(row["map_seed"])
        row["selected"] = int(seed in seeds)
        row["representative"] = int(seed == representative)
    with qualification.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUALIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    value = {"maps": seeds, "representative": representative}
    selection.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _median_iqr(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _grid_hash(path: Path) -> str:
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        return hashlib.sha1(np.asarray(arrays["occupancy"]).tobytes()).hexdigest()[:12]


def _paired_values(
    uav: dict[str, dict[str, str]],
    ideal: dict[str, dict[str, str]],
    field: str,
) -> list[float]:
    """Return UAV-to-ideal ratios for run IDs present in both inputs."""
    return [
        float(uav[run_id][field]) / float(ideal[run_id][field])
        for run_id in sorted(uav.keys() & ideal.keys())
    ]


def build_report(root: Path) -> str:
    """Build the complete pillar campaign report from archived scalar rows and arrays."""
    qualification = _read_csv(root / "qualification.csv")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    offline = _read_csv(root / "offline.csv")
    online = _read_csv(root / "summary.csv")

    map_rows = [
        [row["map_seed"], row["status"], row["free_fraction"], row["blocked_mode_segments"], row["selected"], row["representative"]]
        for row in qualification
        if int(row["selected"])
    ]

    grouped = defaultdict(list)
    for row in offline:
        grouped[(int(row["map_seed"]), row["arm"])].append(row)
    offline_rows = []
    for (map_seed, arm), rows in sorted(grouped.items()):
        mse, mse_iqr = _median_iqr(float(row["occupancy_mse"]) for row in rows)
        fourier, _ = _median_iqr(float(row["fourier_ergodic"]) for row in rows)
        reached = sum(int(row["all_modes_reached"]) for row in rows)
        offline_rows.append([map_seed, arm, len(rows), f"{mse:.3e} ± {mse_iqr:.1e}", f"{fourier:.4f}", f"{reached}/{len(rows)}"])

    ratios = []
    baseline = {
        (int(row["map_seed"]), int(row["seed"])): row
        for row in offline
        if row["arm"] == "baseline"
    }
    for arm in ("h_0.94", "h_6.6"):
        mine = {
            (int(row["map_seed"]), int(row["seed"])): row
            for row in offline
            if row["arm"] == arm
        }
        shared = sorted(baseline.keys() & mine.keys())
        mse = [float(mine[key]["occupancy_mse"]) / float(baseline[key]["occupancy_mse"]) for key in shared]
        fourier = [float(mine[key]["fourier_ergodic"]) / float(baseline[key]["fourier_ergodic"]) for key in shared]
        ratios.append([
            arm,
            len(shared),
            f"{np.median(mse):.3f}" if mse else "n/a",
            f"{sum(value < 1 for value in mse)}/{len(mse)}",
            f"{np.median(fourier):.3f}" if fourier else "n/a",
            f"{sum(value < 1 for value in fourier)}/{len(fourier)}",
        ])

    uav = {row["run_id"]: row for row in online if row["mode"] == "uav"}
    ideal = {row["run_id"]: row for row in online if row["mode"] == "ideal"}
    online_rows = []
    altitude_p95 = []
    for run_id, row in sorted(uav.items()):
        with np.load(root / run_id / "arrays.npz", allow_pickle=False) as arrays:
            deviations = np.abs(np.asarray(arrays["odometry"])[:, 3] - 0.75)
        p95 = float(np.percentile(deviations, 95))
        altitude_p95.append(p95)
        twin = ideal.get(run_id)
        online_rows.append([
            run_id,
            row["collisions"],
            "yes" if np.isfinite(float(row["first_all_modes_s"])) else "no",
            f"{float(row['mode_cycles']):.1f}",
            f"{float(row['guard_fraction']):.4f}",
            f"{float(row['step_p99_ms']):.2f}",
            f"{float(row['deadline_miss_fraction']):.2e}",
            f"{float(row['achieved_rate_hz']):.2f}",
            f"{float(row['real_time_factor']):.4f}",
            f"{p95:.3f}",
            f"{float(row['occupancy_mse']) / float(twin['occupancy_mse']):.3f}" if twin else "missing",
        ])

    mse_pairs = _paired_values(uav, ideal, "occupancy_mse")
    fourier_pairs = _paired_values(uav, ideal, "fourier_ergodic")
    online_pair_rows = [
        ["occupancy MSE", len(mse_pairs), f"{np.median(mse_pairs):.3f}", f"{sum(value < 1 for value in mse_pairs)}/{len(mse_pairs)}"],
        ["Fourier", len(fourier_pairs), f"{np.median(fourier_pairs):.3f}", f"{sum(value < 1 for value in fourier_pairs)}/{len(fourier_pairs)}"],
    ]

    representative = int(selection["representative"])
    map_hashes = {seed: _grid_hash(root / "maps" / f"map_{seed}") for seed in selection["maps"]}
    repeat = root / "maps" / f"map_{representative}_repeat"
    deterministic = repeat.exists() and _grid_hash(repeat) == map_hashes[representative]
    distinct = len(set(map_hashes.values())) == len(map_hashes)
    complete_geometry = {
        int(row["map_seed"]) for row in qualification
    } == set(range(511, 611)) and len(qualification) == 100
    offline_keys = {
        (row["arm"], int(row["map_seed"]), int(row["seed"])) for row in offline
    }
    expected_offline = {
        (arm, map_seed, seed)
        for arm in ("baseline", "h_0.94", "h_6.6")
        for map_seed in selection["maps"]
        for seed in range(43, 61)
    }
    complete_offline = offline_keys == expected_offline and all(
        int(row["steps"]) == 20000 for row in offline
    )
    finite_offline = complete_offline and all(
        np.isfinite(float(row[field]))
        for row in offline
        for field in ("occupancy_mse", "fourier_ergodic", "wall_seconds")
    )
    expected_runs = {f"online_{representative}_s{seed}" for seed in range(43, 48)}
    complete_online = (
        len(online) == 10
        and set(uav) == expected_runs
        and set(ideal) == expected_runs
        and all(
            int(uav[run_id]["map_seed"]) == representative
            and int(uav[run_id]["seed"]) == int(ideal[run_id]["seed"])
            for run_id in expected_runs
        )
    )
    finite_online = complete_online and all(
        np.isfinite(float(row[field]))
        for row in online
        for field in ("occupancy_mse", "fourier_ergodic", "wall_seconds")
    )
    fixed_online = complete_online and len({row["config_hash"] for row in uav.values()}) == 1
    geometry = {
        "pillar_count": 45,
        "pillar_radius_m": [0.3, 0.6],
        "pillar_height_m": [2.0, 3.0],
        "pillar_min_distance_m": 1.2,
        "ring_count": 0,
    }
    fixed_geometry = complete_online and all(
        json.loads((root / run_id / "manifest.json").read_text(encoding="utf-8"))
        .get("map_parameters") == geometry
        for run_id in uav
    )
    fixed_online_map = complete_online and all(
        _grid_hash(root / run_id) == map_hashes[representative] for run_id in uav
    )
    all_safe = bool(uav) and all(int(row["collisions"]) == 0 for row in uav.values())
    timing = bool(uav) and all(
        float(row["step_p99_ms"]) < 16.0
        and float(row["deadline_miss_fraction"]) < 0.001
        and float(row["guard_fraction"]) < 0.01
        and 49.0 <= float(row["achieved_rate_hz"]) <= 51.0
        and 0.98 <= float(row["real_time_factor"]) <= 1.02
        for row in uav.values()
    )
    planar = bool(altitude_p95) and max(altitude_p95) <= 0.05
    visits = sum(np.isfinite(float(row["first_all_modes_s"])) for row in uav.values())
    ideal_visits = sum(np.isfinite(float(row["first_all_modes_s"])) for row in ideal.values())
    planar_claim = all_safe and planar and visits > 0

    return f"""# Planar pillar UAV campaign

The maps were selected from geometry alone before controller evaluation. Controller seeds are repeated trials within three maps, not 54 independent environments. The five online pairs are descriptive.

## Validity checks

| check | result |
| --- | --- |
| all 100 geometry rows archived | {'PASS' if complete_geometry else 'FAIL'} |
| all 162 unique offline rows archived and finite | {'PASS' if finite_offline else 'FAIL'} |
| five UAV flights and five exact twins archived and finite | {'PASS' if finite_online else 'FAIL'} |
| fixed online configuration hash | {'PASS' if fixed_online else 'FAIL'} |
| fixed 45-pillar, zero-ring flight manifests | {'PASS' if fixed_geometry else 'FAIL'} |
| every flight matches the selected raw occupancy | {'PASS' if fixed_online_map else 'FAIL'} |
| same-seed occupancy hash repeats | {'PASS' if deterministic else 'FAIL'} |
| selected map hashes are distinct | {'PASS' if distinct else 'FAIL'} |
| zero raw-map collisions online | {'PASS' if all_safe else 'FAIL'} |
| p99/deadline/guard/control-rate gates | {'PASS' if timing else 'FAIL'} |
| altitude p95 error <= 0.05 m | {'PASS' if planar else 'FAIL'} |
| successful three-mode traversals | {'PASS' if visits else 'FAIL'} ({visits}/5) |
| planar-navigation claim | {'PASS' if planar_claim else 'FAIL'} |

## Geometry-qualified maps

{_table(['seed', 'status', 'free fraction', 'blocked mode segments', 'selected', 'representative'], map_rows)}

Each selected map is connected after the 1.14 m safety inflation and blocks at least two of three straight mode-to-mode segments. Pillars have 0.3–0.6 m radius, 2–3 m height, and no tilted ring obstacles.

## Offline bandwidth sensitivity

{_table(['map', 'arm', 'runs', 'occupancy MSE median ± IQR', 'Fourier median', 'all modes'], offline_rows)}

{_table(['arm vs h=5.0', 'paired paths', 'median MSE ratio', 'MSE wins', 'median Fourier ratio', 'Fourier wins'], ratios)}

These arms are sensitivity evidence only; the evaluation profile remains the preregistered h=5.0 configuration.

## Online SO3 flights and exact ideal twins

{_table(['run', 'collisions', 'all modes', 'cycles', 'guard', 'p99 ms', 'deadline misses', 'rate Hz', 'RTF', 'altitude p95 m', 'UAV/ideal MSE'], online_rows)}

{_table(['paired metric', 'pairs', 'median UAV/ideal ratio', 'UAV wins'], online_pair_rows)}

The online flights reached all three modes in {visits}/5 runs; their exact ideal twins did so in {ideal_visits}/5. These five pairs are descriptive only.

**Negative result:** the planar-navigation claim fails. Although all flights avoided raw-map collisions and 2/5 traversed the three-mode obstacle field, every flight exceeded the preregistered 0.05 m altitude-p95 limit (range {min(altitude_p95):.3f}–{max(altitude_p95):.3f} m). The online-versus-offline visitation mechanism remains unresolved; this campaign shows that the gap persists without supplying a speculative replay model.

See `snapshot.png` for the representative flown path and `scripts/replay_uav.sh pillar/<run>` for bag replay. The plotted pillar footprints are exact raw-map cells; their heights are visualization extrusions because the archived occupancy contains only the planar slice.
"""


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--log", required=True, type=Path)
    probe.add_argument("--seed", required=True, type=int)
    probe.add_argument("--output", required=True, type=Path)
    select = subparsers.add_parser("select")
    select.add_argument("--qualification", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    value = subparsers.add_parser("value")
    value.add_argument("--selection", required=True, type=Path)
    value.add_argument("--field", required=True, choices=("maps", "representative"))
    report = subparsers.add_parser("report")
    report.add_argument("--root", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    if arguments.command == "probe":
        print(json.dumps(record_probe(arguments.log, arguments.seed, arguments.output)))
    elif arguments.command == "select":
        print(json.dumps(write_selection(arguments.qualification, arguments.output)))
    elif arguments.command == "value":
        selected = json.loads(arguments.selection.read_text(encoding="utf-8"))[arguments.field]
        print(" ".join(map(str, selected)) if isinstance(selected, list) else selected)
    else:
        arguments.output.write_text(build_report(arguments.root), encoding="utf-8")
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

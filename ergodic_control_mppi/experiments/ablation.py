"""Staged ablation campaign runner.

Every cell is one closed-loop run of the shipped controller with a patched
config. Parameters are addressed by their dotted YAML path (``stein.memory_gain``,
``mppi.T``), so adding an axis is a config edit, not a code change -- the same
patch-then-``load_config`` mechanism the throwaway A/B runner used, promoted and
given an archive.

    python -m ergodic_control_mppi.experiments.ablation --dry-run
    python -m ergodic_control_mppi.experiments.ablation --stage smoke
    python -m ergodic_control_mppi.experiments.ablation --stage screening --device gpu

Cells are ordered so that runs sharing an XLA-static signature are contiguous:
``mppi.K``, ``mppi.T``, ``mppi.memory_length`` and ``stein.memory_scales`` are
static pytree fields, so each distinct combination costs a full recompile.

Results are written so that no future question needs the GPU again: a scalar CSV
index, the exact config that produced each run, and an ``.npz`` holding the raw
executed path plus everything needed to recompute a metric from it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.common import append_csv
from ergodic_control_mppi.metrics.ergodicity import (
    compute_cumulative_fourier_ergodic_metric,
    compute_cumulative_team_ergodic_error,
    compute_fourier_ergodic_metric,
    compute_reachable_mask,
    compute_team_ergodic_error,
)
from ergodic_control_mppi.simulation import run_simulation

DEFAULT_CAMPAIGN = "configs/experiments/ablation_campaign.yaml"

# Output stride for the stored convergence series. The occupancy series costs
# O(steps * bins) in Python, so computing it every step at 20k steps is seconds
# per run for resolution nothing consumes; full resolution stays recoverable
# from the stored paths.
SERIES_STRIDE = 25

ROW_FIELDS = [
    "stage", "cell_id", "arm", "density", "seed", "obstacle_seed", "steps",
    "axes", "occupancy_mse", "fourier_ergodic", "ms_per_step",
    "wall_seconds", "compiled", "device", "git_sha",
]


@dataclass(frozen=True)
class Cell:
    """One run: a set of patched config values plus its environment."""

    stage: str
    arm: str
    axes: dict[str, Any]
    seed: int
    obstacle_seed: int
    density: str
    steps: int

    @property
    def cell_id(self) -> str:
        """Stable short hash of everything that defines this run."""
        payload = json.dumps(
            {
                "stage": self.stage,
                "arm": self.arm,
                "axes": {k: _canonical(v) for k, v in sorted(self.axes.items())},
                "seed": self.seed,
                "obstacle_seed": self.obstacle_seed,
                "density": self.density,
                "steps": self.steps,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Campaign:
    """Parsed campaign definition."""

    base_config: Path
    output_root: Path
    densities: dict[str, dict[str, Any]]
    stages: dict[str, dict[str, Any]]
    defaults: dict[str, Any] = field(default_factory=dict)


def _canonical(value: Any) -> Any:
    """Round floats so an axis value hashes identically across YAML round-trips."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign ``data["a"]["b"] = value`` for ``dotted == "a.b"``."""
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"axis '{dotted}' does not address a config mapping")
        node = child
    node[keys[-1]] = value


def _get_dotted(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _static_key(data: dict[str, Any]) -> tuple:
    """Group key for XLA recompiles.

    The static pytree fields are ``mppi.K``, ``mppi.T``, ``mppi.memory_length``
    and ``stein.memory_scales``. ``memory_length`` is derived from
    ``stein.memory_time`` and ``model.delta_t`` when absent, so those two raw
    values belong in the key rather than the derivation being duplicated here:
    the resolved signature is a pure function of this tuple, so sorting by it
    groups compiles exactly.
    """
    # -1.0 stands in for an absent key so the tuple stays orderable.
    return tuple(
        -1.0 if (value := _get_dotted(data, dotted)) is None else float(value)
        for dotted in (
            "mppi.K",
            "mppi.T",
            "mppi.memory_length",
            "stein.memory_time",
            "stein.memory_scales",
            "model.delta_t",
        )
    )


def load_campaign(path: str | Path = DEFAULT_CAMPAIGN) -> Campaign:
    """Read and lightly validate a campaign definition."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("campaign YAML must be a mapping")
    for key in ("base_config", "output_root", "stages"):
        if key not in raw:
            raise ValueError(f"campaign YAML is missing '{key}'")
    return Campaign(
        base_config=Path(raw["base_config"]),
        output_root=Path(raw["output_root"]),
        densities=raw.get("densities", {}),
        stages=raw["stages"],
        defaults=raw.get("defaults", {}),
    )


def _seeds_of(spec: dict[str, Any], defaults: dict[str, Any]) -> list[int]:
    seeds = spec.get("seeds", defaults.get("seeds", [43]))
    if isinstance(seeds, int):
        return list(range(seeds))
    return [int(s) for s in seeds]


def _cells_for_stage(name: str, spec: dict[str, Any], defaults: dict[str, Any]) -> list[Cell]:
    """Expand one stage definition into its cells.

    Four stage kinds cover the whole campaign:

    ``ofat``       one axis at a time around the shipped default
    ``grid``       full factorial over ``pairs`` of axes (interaction heatmaps)
    ``arms``       named structural variants, each a dict of axis overrides
    ``factorial``  named arms crossed with densities and obstacle counts
    """
    kind = spec.get("kind", "ofat")
    steps = int(spec.get("steps", defaults.get("steps", 10000)))
    seeds = _seeds_of(spec, defaults)
    obstacle_seed = int(spec.get("obstacle_seed", defaults.get("obstacle_seed", 43)))
    density = str(spec.get("density", defaults.get("density", "trimodal")))
    cells: list[Cell] = []

    if kind == "ofat":
        # The all-defaults cell is emitted once as the shared reference arm.
        for seed in seeds:
            cells.append(Cell(name, "default", {}, seed, obstacle_seed, density, steps))
        for axis, levels in spec.get("axes", {}).items():
            for level in levels:
                for seed in seeds:
                    cells.append(
                        Cell(name, axis, {axis: level}, seed, obstacle_seed, density, steps)
                    )
    elif kind == "grid":
        for pair in spec.get("pairs", []):
            (axis_a, levels_a), (axis_b, levels_b) = pair["axes"].items()
            arm = pair.get("name", f"{axis_a}__{axis_b}")
            for level_a in levels_a:
                for level_b in levels_b:
                    for seed in seeds:
                        cells.append(
                            Cell(
                                name, arm, {axis_a: level_a, axis_b: level_b},
                                seed, obstacle_seed, density, steps,
                            )
                        )
    elif kind == "arms":
        for arm, overrides in spec.get("arms", {}).items():
            for seed in seeds:
                cells.append(
                    Cell(name, arm, dict(overrides or {}), seed, obstacle_seed, density, steps)
                )
    elif kind == "factorial":
        for arm, overrides in spec.get("arms", {}).items():
            for density_name in spec.get("densities", [density]):
                for count in spec.get("obstacle_counts", [3]):
                    axes = dict(overrides or {})
                    axes["map.obstacles.num_obstacles"] = int(count)
                    for seed in seeds:
                        cells.append(
                            Cell(
                                name, arm, axes, seed,
                                # One layout per obstacle count, shared by every
                                # arm, so arms are compared on the same maps.
                                obstacle_seed + int(count),
                                str(density_name), steps,
                            )
                        )
    else:
        raise ValueError(f"unknown stage kind '{kind}' in stage '{name}'")
    return cells


def expand(campaign: Campaign, stages: list[str] | None = None) -> list[Cell]:
    """Build the ordered cell list, grouped so XLA recompiles are amortized."""
    selected = stages or list(campaign.stages)
    unknown = [s for s in selected if s not in campaign.stages]
    if unknown:
        raise ValueError(f"unknown stage(s): {unknown}; have {sorted(campaign.stages)}")

    cells: list[Cell] = []
    seen: set[str] = set()
    for name in selected:
        for cell in _cells_for_stage(name, campaign.stages[name], campaign.defaults):
            if cell.cell_id not in seen:
                seen.add(cell.cell_id)
                cells.append(cell)

    base = yaml.safe_load(campaign.base_config.read_text(encoding="utf-8"))
    return sorted(
        cells,
        key=lambda c: (
            _static_key(_patched(base, campaign, c)),
            c.stage,
            c.arm,
            json.dumps({k: _canonical(v) for k, v in sorted(c.axes.items())}, sort_keys=True),
            c.seed,
        ),
    )


def _patched(base: dict[str, Any], campaign: Campaign, cell: Cell) -> dict[str, Any]:
    """Return the base config with this cell's density, seeds and axes applied."""
    data = copy.deepcopy(base)
    if cell.density:
        spec = campaign.densities.get(cell.density)
        if spec is None:
            raise ValueError(
                f"density '{cell.density}' is not defined in the campaign YAML"
            )
        data["density"] = copy.deepcopy(spec)
    data["seed"] = cell.seed
    data["steps"] = cell.steps
    data.setdefault("map", {}).setdefault("obstacles", {})["seed"] = cell.obstacle_seed
    for axis, value in cell.axes.items():
        _set_dotted(data, axis, value)
    return data


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _grids(config) -> tuple[np.ndarray, tuple, tuple, tuple, np.ndarray]:
    """Target density, map limits, occupancy bins and reachable mask.

    Built exactly as plotting/simulation.py and the metrics module expect, so a
    stored run can be re-scored offline without reloading the config.
    """
    from ergodic_control_mppi.mppi.stein import pdf
    import jax.numpy as jnp

    workspace = config.controller.workspace
    limits_x = tuple(map(float, workspace.x_limits))
    limits_y = tuple(map(float, workspace.y_limits))
    nx = max(2, int((limits_x[1] - limits_x[0]) / config.run.resolution))
    ny = max(2, int((limits_y[1] - limits_y[0]) / config.run.resolution))
    axis_x = jnp.linspace(limits_x[0], limits_x[1], nx)
    axis_y = jnp.linspace(limits_y[0], limits_y[1], ny)
    grid_x, grid_y = jnp.meshgrid(axis_x, axis_y)
    density = np.asarray(pdf(jnp.stack((grid_x, grid_y), axis=-1), config.controller.gmm))
    mask = compute_reachable_mask(
        workspace.obstacles, float(workspace.safe_distance), limits_x, limits_y, (nx, ny)
    )
    return density / density.sum(), limits_x, limits_y, (nx, ny), mask


def _completed(index_csv: Path) -> set[str]:
    if not index_csv.exists():
        return set()
    with index_csv.open("r", encoding="utf-8") as handle:
        return {row["cell_id"] for row in csv.DictReader(handle) if row.get("cell_id")}


def _write_manifest(root: Path, campaign: Campaign, cells: list[Cell], device: str) -> None:
    import jax

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "git_sha": _git_sha(),
                "jax_version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "requested_device": device,
                "base_config": str(campaign.base_config),
                "stages": sorted({c.stage for c in cells}),
                "total_cells": len(cells),
                "series_stride": SERIES_STRIDE,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_cell(
    campaign: Campaign,
    base: dict[str, Any],
    cell: Cell,
    root: Path,
    device: str,
    fourier_order: int,
    compiled: bool = False,
) -> dict[str, Any]:
    """Execute one cell, archive its raw arrays, and return its CSV row.

    ``compiled`` marks a cell that triggered an XLA recompile, so its
    ``ms_per_step`` includes compilation and must be excluded from timing
    aggregates. experiments/timing.py is the authority on per-step cost; this
    column is a coarse by-product.
    """
    config_path = root / "configs" / cell.stage / f"{cell.cell_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = _patched(base, campaign, cell)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)

    started = time.perf_counter()
    result = run_simulation(config, device)
    wall_seconds = time.perf_counter() - started

    target, limits_x, limits_y, bins, mask = _grids(config)
    occupancy_mse = compute_team_ergodic_error(
        result.paths, target, limits_x, limits_y, bins, reachable_mask=mask
    )
    fourier = compute_fourier_ergodic_metric(
        result.paths, target, limits_x, limits_y, order=fourier_order, reachable_mask=mask
    )
    series_occupancy = compute_cumulative_team_ergodic_error(
        result.paths, target, limits_x, limits_y, bins,
        reachable_mask=mask, stride=SERIES_STRIDE,
    )
    series_fourier = compute_cumulative_fourier_ergodic_metric(
        result.paths, target, limits_x, limits_y,
        order=fourier_order, reachable_mask=mask, stride=SERIES_STRIDE,
    )

    run_path = root / "runs" / cell.stage / f"{cell.cell_id}.npz"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_path,
        paths=result.paths.astype(np.float32),
        convergence_occupancy=np.asarray(series_occupancy, dtype=np.float64),
        convergence_fourier=np.asarray(series_fourier, dtype=np.float64),
        series_stride=np.asarray(SERIES_STRIDE, dtype=np.int64),
        initial_state=result.initial_states[0].astype(np.float32),
        obstacles=np.asarray(config.controller.workspace.obstacles, dtype=np.float32),
        target_grid=target.astype(np.float32),
        reachable_mask=mask,
        map_limits=np.asarray([limits_x, limits_y], dtype=np.float64),
    )

    return {
        "stage": cell.stage,
        "cell_id": cell.cell_id,
        "arm": cell.arm,
        "density": cell.density,
        "seed": cell.seed,
        "obstacle_seed": cell.obstacle_seed,
        "steps": cell.steps,
        "axes": json.dumps({k: _canonical(v) for k, v in sorted(cell.axes.items())}, sort_keys=True),
        "occupancy_mse": occupancy_mse,
        "fourier_ergodic": fourier,
        "ms_per_step": 1000.0 * wall_seconds / max(cell.steps, 1),
        "wall_seconds": round(wall_seconds, 3),
        "compiled": int(compiled),
        "device": result.device,
        "git_sha": _git_sha(),
    }


def run_campaign(
    campaign_path: str | Path = DEFAULT_CAMPAIGN,
    *,
    stages: list[str] | None = None,
    device: str = "auto",
    shard: int = 0,
    shards: int = 1,
    fourier_order: int = 10,
    dry_run: bool = False,
    limit: int | None = None,
) -> int:
    """Run (or count) the campaign. Returns the number of cells executed."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("require shards >= 1 and 0 <= shard < shards")

    campaign = load_campaign(campaign_path)
    cells = expand(campaign, stages)
    base = yaml.safe_load(campaign.base_config.read_text(encoding="utf-8"))
    mine = cells[shard::shards]

    if dry_run:
        _report(campaign, cells, mine, base, shard, shards)
        return 0

    root = campaign.output_root
    _write_manifest(root, campaign, cells, device)
    executed = 0
    previous_static: tuple | None = None
    for position, cell in enumerate(mine, start=1):
        index_csv = root / f"{cell.stage}.csv"
        if cell.cell_id in _completed(index_csv):
            continue
        static = _static_key(_patched(base, campaign, cell))
        compiled = static != previous_static
        if compiled:
            print(f"# recompile: static signature {static}", flush=True)
            previous_static = static
        row = run_cell(campaign, base, cell, root, device, fourier_order, compiled=compiled)
        append_csv(index_csv, row, ROW_FIELDS)
        executed += 1
        # Machine-greppable progress line: the remote poll counts these.
        print(
            f"ROW={position}/{len(mine)} stage={cell.stage} arm={cell.arm} "
            f"seed={cell.seed} mse={row['occupancy_mse']:.4g} "
            f"eps={row['fourier_ergodic']:.4g} {row['wall_seconds']:.1f}s",
            flush=True,
        )
        if limit is not None and executed >= limit:
            break
    print(f"EXIT_OK executed={executed} of {len(mine)}", flush=True)
    return executed


# Cost model for the dry-run estimate, least-squares fitted to the 25 points
# measured by experiments/timing.py on the RTX 5090 that runs the campaign
# (median |error| 4.0%, measured bracket 1.2-9.0 ms/step). Terms: constant
# overhead, rollouts O(K*T), occupancy KDE O(Q*P^2), attraction O(T^2). A flat
# per-step rate underestimates the expensive corners by ~5x, so budgeting needs
# at least this much structure.
#
# Refit after any hardware change:
#   python -m ergodic_control_mppi.experiments.timing --device gpu --scaling
# then least-squares the four terms against the recorded total_ms values.
COST_MS = (2.367, 8.151e-7, 1.44e-8, 1.378e-5)


def _estimate_ms(data: dict[str, Any]) -> float:
    """Estimated ms/step for a patched config."""
    samples = float(_get_dotted(data, "mppi.K") or 1000)
    horizon = float(_get_dotted(data, "mppi.T") or 350)
    scales = float(_get_dotted(data, "stein.memory_scales") or 3)
    length = _get_dotted(data, "mppi.memory_length")
    if length is None:
        memory_time = float(_get_dotted(data, "stein.memory_time") or 10.0)
        delta_t = float(_get_dotted(data, "model.delta_t") or 0.02)
        length = math.ceil(3.0 * memory_time / delta_t)
    constant, rollout, kde, attraction = COST_MS
    return (
        constant
        + rollout * samples * horizon
        + kde * scales * float(length) ** 2
        + attraction * horizon * horizon
    )


def _report(campaign, cells, mine, base, shard: int, shards: int) -> None:
    """Print the cell budget, recompile count and wall-clock estimate."""
    from collections import Counter

    by_stage = Counter(c.stage for c in cells)
    statics = {_static_key(_patched(base, campaign, c)) for c in cells}
    seconds: dict[str, float] = {}
    for cell in cells:
        estimate = _estimate_ms(_patched(base, campaign, cell)) * cell.steps / 1000.0
        seconds[cell.stage] = seconds.get(cell.stage, 0.0) + estimate

    print(f"campaign      : {campaign.base_config} -> {campaign.output_root}")
    print(f"  {'stage':<15}{'cells':>7}{'Msteps':>9}{'est. hours':>12}")
    for stage, count in by_stage.items():
        stage_steps = sum(c.steps for c in cells if c.stage == stage)
        print(f"  {stage:<15}{count:>7}{stage_steps/1e6:>9.2f}{seconds[stage]/3600:>12.1f}")
    print(f"  {'TOTAL':<15}{len(cells):>7}"
          f"{sum(c.steps for c in cells)/1e6:>9.2f}{sum(seconds.values())/3600:>12.1f}")
    shard_seconds = sum(
        _estimate_ms(_patched(base, campaign, c)) * c.steps / 1000.0 for c in mine
    )
    print(f"this shard    : {len(mine)} cells ({shard + 1}/{shards}), "
          f"{shard_seconds/3600:.1f} h estimated")
    # ~1.2 s per compile measured at K=1000, T=350; negligible against the runs.
    print(f"XLA recompiles: {len(statics)} distinct static signatures "
          f"(~{len(statics) * 1.2 / 60:.0f} min total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--stage", action="append", dest="stages",
                        help="stage name; repeatable. Default: every stage.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--fourier-order", type=int, default=10)
    parser.add_argument("--limit", type=int, help="stop after N executed cells")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cell budget and recompile count, run nothing")
    args = parser.parse_args()
    run_campaign(
        args.config,
        stages=args.stages,
        device=args.device,
        shard=args.shard,
        shards=args.shards,
        fourier_order=args.fourier_order,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

"""The final ablation campaign: 38 arms x 6 maps x 3 densities x 6 seeds, one branch.

Two conclusions in this project were drawn from a single map and both failed when a third
was added, each measured on map 516, which turned out to be anomalous on both axes. So
every arm here is measured on six maps spanning three obstacle densities, and the promotion
gate is a *consistency* gate: same sign on at least 4 of 6 maps, not merely a small pooled
p-value.

All six maps share one grid shape and one start state, so a single ``run_batch`` call holds
an entire arm across every map and seed. That is what makes 1 368 cells a six-hour run.

**Fixed lane count is load-bearing.** Batched execution is a different numerical branch than
sequential, and changing the lane count changes the branch again. Comparing an arm in one
call against the baseline in another is only valid if a fixed width with different
companions is bit-identical -- verify with ``--verify-branch`` before spending the night.

    uv run python scripts/final_ablation.py maps      # build the campaign map manifest
    uv run python scripts/final_ablation.py verify    # the lane-count invariance gate
    uv run python scripts/final_ablation.py run       # the campaign, resumable
"""

import argparse
import csv
import hashlib
import json
import socket
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.uav_ablation import (
    _BY_NAME,
    FINAL_ARMS,
    _apply,
)
from ergodic_control_mppi.experiments.uav_pillar_tuning import (
    PREFLIGHT_STEPS,
    _grid_config,
    score_run,
)
from ergodic_control_mppi.mppi.single import run_batch, stack_params
from ergodic_control_mppi.simulation import controller_key, select_device

# `arm`, `axis` and `value` make a row ablation-shaped for scripts/report_figures.py.
# `obs_num` is the density factor, reported and never pooled away. `lanes` and `execution`
# record the numerical branch: a row from a different batch width answers a different
# question and must not satisfy this cell.
FIELDS = [
    "arm", "axis", "value", "map_seed", "obs_num", "seed", "steps",
    "release_ratio", "alpha", "lam_max",
    "ess_settled_median", "temperature_settled_median", "temperature_cap_fraction",
    "all_modes_reached", "first_all_modes_s", "mode_visits", "mode_cycles",
    "mode_dwell_median_s", "in_mode_fraction", "occupancy_mse", "fourier_ergodic",
    "collisions", "min_clearance_m", "path_length_m", "wall_seconds",
    "device", "hardware", "execution", "lanes", "jax_version",
]

MAP_MANIFEST = Path("results/uav/campaign_maps.json")
DEFAULT_OUTPUT = Path("results/uav/ablation_final.csv")
STOP_FILE = Path("results/uav/STOP")

# `K = 1000` quadruples the rollout tensor, so the K axis may not fit the campaign width.
# It runs split into this many equal chunks instead -- every K chunk the same width as every
# other K chunk, including the baseline replicate, so the K comparison is internally exact
# even though it is not on the same branch as the rest.
#
# A chunk *count* rather than a lane count: the width has to follow the map and seed counts,
# and a hard-coded 27 silently stops dividing the moment either changes.
AXIS_CHUNKS = {"K": 4}

# Densities the campaign spans. Two maps each. 10-20 pillars is well inside the runnable
# band: 15 gives 0.90 reachable fraction and 25 gives 0.84, both flown without incident, so
# 20 is comfortably conservative.
DENSITIES = (10, 15, 20)
MAPS_PER_DENSITY = 2
# No pins and no flown maps. Every recorded flight was flown by the Stein controller, so
# under same-version control none of them describes this campaign's controller -- pinning a
# map to keep a flight comparison alive would keep a comparison that is already void.
PINNED: dict[int, tuple[int, ...]] = {}


# --------------------------------------------------------------------------- maps


def build_map_manifest(output: Path, roots: dict[int, Path]) -> dict:
    """Pick two maps per density and record them, refusing anything not a pillar field.

    Selection is `select_split`'s ordering -- qualifying seeds ranked by worst-mode blocked
    target mass -- except where :data:`PINNED` overrides it. Every map is checked here rather
    than at run time so a wrong map cannot reach the campaign at all.
    """
    entries = []
    for obs_num in DENSITIES:
        root = roots[obs_num]
        selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
        ordered = list(selection["development"]) + list(selection["holdout"])
        chosen = list(PINNED.get(obs_num, ()))
        for seed in ordered:
            if len(chosen) == MAPS_PER_DENSITY:
                break
            if seed not in chosen:
                chosen.append(seed)
        if len(chosen) != MAPS_PER_DENSITY:
            raise SystemExit(f"density {obs_num}: only {len(chosen)} maps available")
        for map_seed in chosen:
            run_dir = root / "maps" / f"map_{map_seed}"
            entries.append(_check_map(run_dir, map_seed, obs_num))
    _assert_distinct(entries)
    manifest = {"densities": list(DENSITIES), "maps": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _check_map(run_dir: Path, map_seed: int, obs_num: int) -> dict:
    """Validate one map and return its manifest entry.

    Every map in this campaign is a ``random_forest`` pillar field. Mixing generators would
    put a map-*family* difference inside an ablation measuring a density difference, and any
    per-map disagreement would then be unattributable between the two. `results/uav/paper01`
    is perlin and carries no ``map_source`` key at all, so it fails here by construction
    rather than by anyone remembering to exclude it.
    """
    if not (run_dir / "arrays.npz").exists():
        raise SystemExit(f"{run_dir} has no arrays.npz -- build it before selecting")
    meta = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source = meta.get("map_source")
    if source != "random_forest":
        raise SystemExit(
            f"{run_dir} has map_source={source!r}, not 'random_forest'. Every campaign map "
            "must be a pillar field; a perlin or hand-built map is a different map family, "
            "not a different density."
        )
    if int(meta["map_seed"]) != map_seed:
        raise SystemExit(f"{run_dir} records map_seed={meta['map_seed']}, expected {map_seed}")
    # The recorder does not write obs_num into a map manifest, so the density cannot be
    # confirmed from the map itself. The path is the next best witness and it catches the
    # error that can actually happen: a wrong root passed for a density. Without this a
    # 15-pillar map could be labelled 35 and the density factor would be quietly fictional.
    if f"density_{obs_num}" not in run_dir.parts and f"density_{obs_num}" not in str(run_dir):
        raise SystemExit(
            f"{run_dir} does not live under density_{obs_num}/. The density label comes "
            "from the root, so a mismatched path means the label is wrong."
        )
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=False)
    return {
        "map_seed": map_seed,
        "obs_num": obs_num,
        "run_dir": str(run_dir),
        "occupied_cells": int(np.asarray(arrays["occupancy"]).sum()),
        # The witness `_assert_distinct` compares. Recorded in the manifest rather than
        # recomputed, so a later reader can check the claim without the arrays.
        "occupancy_digest": hashlib.sha256(
            np.ascontiguousarray(np.asarray(arrays["occupancy"], dtype=np.uint8)).tobytes()
        ).hexdigest()[:16],
        "reachable_fraction": float(np.asarray(arrays["reachable_mask"]).mean()),
        "grid_shape": list(np.asarray(arrays["grid"]).shape),
        "initial_state": [float(v) for v in np.asarray(arrays["initial_state"])],
    }


def _assert_distinct(entries: list[dict]) -> None:
    """Refuse a manifest that names the same field twice.

    Seed 525 qualified at both 15 and 25 pillars, and the driver flew the 15-pillar field
    under both labels -- silently, because every downstream check keys on the label rather
    than on the array. That is why the old gate read 6-of-8 and why ``report_figures.py``
    carried a ``DUPLICATE_MAPS`` patch.

    Two assertions, because either alone misses a case. ``(obs_num, map_seed)`` catches the
    manifest listing one entry twice; ``occupancy_digest`` catches two *differently labelled*
    entries that are the same field, which is the failure that actually happened.

    The digest is a hash of the occupancy array itself, not a summary of it. An earlier
    version compared ``occupied_cells``, which is an integer count over a quantized grid and
    therefore collides between genuinely different pillar draws: maps 513 and 530 of
    ``density_15`` both have 457 occupied cells and differ in 908 of them. A summary can only
    give false positives here; the array is the thing the claim is about, so it is what is
    compared.
    """
    labels = [(e["obs_num"], e["map_seed"]) for e in entries]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"map manifest repeats an (obs_num, map_seed): {labels}")
    digests = [e["occupancy_digest"] for e in entries]
    if len(set(digests)) != len(digests):
        seen: dict[str, tuple] = {}
        for entry in entries:
            key = entry["occupancy_digest"]
            if key in seen:
                raise SystemExit(
                    f"maps {seen[key]} and {(entry['obs_num'], entry['map_seed'])} have "
                    f"identical occupancy arrays: they are the same field under two labels"
                )
            seen[key] = (entry["obs_num"], entry["map_seed"])


def load_maps(path: Path) -> list[dict]:
    """Load the campaign maps and assert the invariants batching depends on."""
    entries = json.loads(path.read_text(encoding="utf-8"))["maps"]
    shapes = {tuple(e["grid_shape"]) for e in entries}
    if len(shapes) != 1:
        raise SystemExit(f"maps disagree on grid shape: {shapes}. Lanes cannot be stacked.")
    starts = {tuple(round(v, 6) for v in e["initial_state"]) for e in entries}
    if len(starts) != 1:
        # run_batch broadcasts one start state across lanes, so a second one would silently
        # fly some maps from the wrong place.
        raise SystemExit(f"maps disagree on start state: {starts}")
    return entries


# --------------------------------------------------------------------------- lanes


def groups(maps: list[dict], seeds: range, arms=FINAL_ARMS):
    """Yield ``(label, execution, [lane, ...])`` with every group at exactly one width.

    A lane is ``(map_entry, arm, seed)``. One arm's 108 cells are one group at the default
    width; a quarantined axis is chunked into equal-width groups instead, which keeps its
    comparison internally exact without forcing the whole campaign down to that width.
    """
    full = len(maps) * len(seeds)
    for arm in arms:
        axis = _BY_NAME[arm][0] if arm != "baseline" else "-"
        # The default width is one arm's whole cell set, derived rather than hard-coded: it
        # must track the map and seed counts or a smaller campaign would silently run at a
        # width nothing else used. AXIS_CHUNKS quarantines only the axes that will not fit.
        width = full // AXIS_CHUNKS.get(axis, 1)
        lanes = [(entry, arm, seed) for entry in maps for seed in seeds]
        if len(lanes) % width:
            raise SystemExit(f"{len(lanes)} lanes is not divisible by width {width}")
        for index in range(0, len(lanes), width):
            chunk = lanes[index:index + width]
            suffix = f"_c{index // width}" if width != len(lanes) else ""
            yield f"{arm}{suffix}", f"batch{width}", chunk

    # The quarantined axes need a baseline measured at *their* width, or their arms have no
    # comparator on their own branch.
    for axis, chunks in AXIS_CHUNKS.items():
        width = full // chunks
        if not any(_BY_NAME[a][0] == axis for a in arms if a != "baseline"):
            continue
        lanes = [(entry, "baseline", seed) for entry in maps for seed in seeds]
        for index in range(0, len(lanes), width):
            yield (f"baseline_{axis}_c{index // width}", f"batch{width}",
                   lanes[index:index + width])


def identity(lane, steps: int, hardware: str, execution: str) -> tuple:
    """Resume key for one cell. Branch selectors are part of it, not metadata.

    ``obs_num`` is in the key because a map seed is **not** unique across densities: all
    three ``prepare`` runs probe seeds 511-610, so the same seed can qualify and be selected
    at two densities, where it is a completely different field. Keyed on the seed alone,
    those two cells collide -- resume would skip one that never ran, and the analysis would
    pair rows from different densities against each other.
    """
    entry, arm, seed = lane
    return (arm, str(entry["obs_num"]), str(entry["map_seed"]), str(seed), str(steps),
            hardware, execution)


# --------------------------------------------------------------------------- running


def _configs(cache: dict, entry: dict, config_path: str = "configs/uav_profile.yaml") -> tuple:
    """Load and cache one map's base config, manifest and arrays.

    Keyed on density *and* seed. Not the seed alone: all three ``prepare`` runs probe
    511-610, so a seed can be selected at two densities where it is a completely different
    field. Seed 525 was, and a seed-keyed cache flew the 15-pillar map under both labels --
    silently, because every downstream check keys on the label rather than on the array. See
    :func:`identity`, which carries ``obs_num`` for the same reason.
    """
    key = (entry["obs_num"], entry["map_seed"])
    if key not in cache:
        cache[key] = _grid_config(Path(entry["run_dir"]), config_path)
    return cache[key]


def run_group(label: str, execution: str, lanes: list, cache: dict, arguments) -> list[dict]:
    """Run one whole group as a single fused call and score every lane against its own map.

    Unlike the gate driver this group spans maps, so the config and the scoring arrays are
    looked up per lane. Only the grid differs between maps and the grid is a traced leaf, so
    the lanes still share one static signature and stack cleanly.
    """
    device = select_device(arguments.device)
    lane_configs, lane_arrays, lane_manifests = [], [], []
    for entry, arm, _ in lanes:
        base, manifest, arrays = _configs(cache, entry, arguments.config)
        overrides = dict(_BY_NAME[arm][2]) if arm != "baseline" else {}
        lane_configs.append(_apply(base, overrides))
        lane_arrays.append(arrays)
        lane_manifests.append(manifest)

    stacked = stack_params([jax.device_put(c.controller, device) for c in lane_configs])
    keys = jnp.stack([controller_key(seed) for *_, seed in lanes])
    initial = jnp.asarray(
        np.asarray(lane_arrays[0]["initial_state"]), dtype=jnp.float32
    )
    controls = jnp.zeros((lane_configs[0].controller.mppi.horizon, 3), dtype=jnp.float32)

    print(f"[{label}] {len(lanes)} lanes, compiling and running...", flush=True)
    started = time.perf_counter()
    result = jax.jit(run_batch, static_argnames=("steps", "preflight_steps"))(
        stacked, initial, controls, keys,
        steps=arguments.steps, preflight_steps=PREFLIGHT_STEPS,
    )
    jax.block_until_ready(result.path)
    wall = time.perf_counter() - started
    paths = np.asarray(result.path)
    ess = np.asarray(result.ess_fraction)
    temperatures = np.asarray(result.temperature)
    print(f"[{label}] {wall:.0f}s total, {wall / len(lanes):.1f}s per cell", flush=True)

    rows = []
    for index, (entry, arm, seed) in enumerate(lanes):
        row = score_run(
            lane_configs[index], lane_arrays[index], lane_manifests[index], seed,
            arguments.steps,
            positions=paths[index, :, :2], velocities=paths[index, :, 2:4],
            ess_fractions=ess[index], temperatures=temperatures[index],
            wall=wall / len(lanes), device=device.platform,
        )
        axis, value, _ = _BY_NAME[arm] if arm != "baseline" else ("-", "-", {})
        controller = lane_configs[index].controller
        row.update({
            "arm": arm, "axis": axis, "value": value,
            "map_seed": entry["map_seed"], "obs_num": entry["obs_num"],
            # Read back from the realised config, not the override dict, so a lane that did
            # not set an axis records what it actually ran with rather than a blank.
            "release_ratio": float(controller.field.release_ratio),
            "alpha": float(controller.mppi.alpha),
            "lam_max": float(controller.mppi.temperature_max),
            "hardware": arguments.hardware, "execution": execution, "lanes": len(lanes),
        })
        rows.append(row)
    return rows


def append_rows(output: Path, rows: list[dict]) -> None:
    """Write a whole group in one call, header-checked.

    Per-group rather than per-row on purpose. Scoring runs ~0.55 s per lane, so appending
    inside that loop leaves a minute-long window in which an interrupt writes half a group
    -- which the resume logic then refuses, requiring manual deletion. A group is now
    atomically present or absent.

    The header is compared against ``FIELDS`` every time: ``DictWriter`` emits values in
    ``FIELDS`` order regardless of what the file says, so adding a column and appending to
    an existing archive shifts every later row by one and corrupts the file while looking
    like it worked. That happened once.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (output.exists() and output.stat().st_size)
    if not fresh:
        with output.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
        if header != FIELDS:
            missing = [f for f in FIELDS if f not in header]
            extra = [f for f in header if f not in FIELDS]
            raise SystemExit(
                f"{output} has a stale header (missing {missing}, unexpected {extra}). "
                "Appending would write misaligned rows -- migrate or move the file first."
            )
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def completed(output: Path) -> set:
    """Identities already in the archive."""
    if not output.exists():
        return set()
    with output.open(encoding="utf-8", newline="") as stream:
        return {
            (r["arm"], r["obs_num"], r["map_seed"], r["seed"], r["steps"], r["hardware"],
             r["execution"])
            for r in csv.DictReader(stream)
        }


def verify_branch(maps: list[dict], arguments) -> bool:
    """The gate: at a fixed lane count, does a lane depend on which companions it has?

    Changing the width is already known to change the result. This asks the narrower
    question the campaign's grouping rests on -- same width, different companions -- by
    running one lane set twice with different neighbours and comparing bit patterns. If it
    fails, every arm must be grouped with its own baseline instead.
    """
    cache: dict = {}
    entry = maps[0]
    # At the width the campaign actually uses. XLA can lower a batch of 8 and a batch of 108
    # differently, so a gate passed at 8 says nothing about 108 -- and 108 is what the
    # comparison rests on.
    width = arguments.verify_lanes or (len(maps) * arguments.seeds)
    half = width // 2
    shared = [(entry, "baseline", seed) for seed in range(43, 43 + half)]
    # Two companions that differ in a traced leaf only, so the static signature -- and
    # therefore the lowering -- is the same in both calls. If the shared half still moves,
    # it moved because of its neighbours' *values*, which is exactly what the gate asks.
    first = shared + [(entry, "gain_30", seed) for seed in range(43, 43 + half)]
    second = shared + [(entry, "gain_120", seed) for seed in range(43, 43 + half)]

    outputs = []
    for label, lanes in (("A", first), ("B", second)):
        device = select_device(arguments.device)
        configs = []
        for lane_entry, arm, _ in lanes:
            base, _, _ = _configs(cache, lane_entry, arguments.config)
            configs.append(_apply(base, dict(_BY_NAME[arm][2]) if arm != "baseline" else {}))
        stacked = stack_params([jax.device_put(c.controller, device) for c in configs])
        keys = jnp.stack([controller_key(seed) for *_, seed in lanes])
        arrays = _configs(cache, entry, arguments.config)[2]
        result = jax.jit(run_batch, static_argnames=("steps", "preflight_steps"))(
            stacked,
            jnp.asarray(np.asarray(arrays["initial_state"]), dtype=jnp.float32),
            jnp.zeros((configs[0].controller.mppi.horizon, 3), dtype=jnp.float32),
            keys, steps=arguments.steps, preflight_steps=PREFLIGHT_STEPS,
        )
        outputs.append(np.asarray(result.path)[:half])
        print(f"  branch {label}: {len(lanes)} lanes done", flush=True)

    identical = np.array_equal(outputs[0], outputs[1])
    if identical:
        print("GATE PASS: shared lanes are bit-identical across different companions.")
        print("           One group per arm at a fixed width is valid.")
    else:
        delta = np.abs(outputs[0] - outputs[1])
        print("GATE FAIL: shared lanes differ despite equal lane count.")
        print(f"           max |delta| = {delta.max():.3e}, "
              f"first differing step = {int(np.argmax(delta.max(axis=(0, 2)) > 0))}")
        print("           Fall back to grouping each axis with its own baseline.")
    return identical


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["maps", "verify", "run", "plan"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maps", type=Path, default=MAP_MANIFEST)
    parser.add_argument("--config", default="configs/uav_profile.yaml",
                        help="Controller config; varied only by the target-generalization check")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--first-seed", type=int, default=43)
    parser.add_argument("--seeds", type=int, default=6,
                        help="Seeds per map. 6 x 6 maps = 36 paired cells per arm, which is "
                             "the registered design and the width the branch gate is run at.")
    parser.add_argument("--device", default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--hardware", default=socket.gethostname())
    parser.add_argument("--arms", default="", help="Comma-separated subset of FINAL_ARMS")
    parser.add_argument("--verify-lanes", type=int, default=0,
                        help="Lane count for the branch gate; 0 derives the campaign's own "
                             "width from the map manifest, which is what it must match.")
    parser.add_argument("--stop-file", type=Path, default=STOP_FILE,
                        help="Touch this to finish the current group and exit cleanly.")
    arguments = parser.parse_args()
    seeds = range(arguments.first_seed, arguments.first_seed + arguments.seeds)

    if arguments.action == "maps":
        roots = {obs: Path(f"results/uav/density_{obs}") for obs in DENSITIES}
        manifest = build_map_manifest(arguments.maps, roots)
        for entry in manifest["maps"]:
            print(f"  obs={entry['obs_num']:2d} map={entry['map_seed']} "
                  f"occupied={entry['occupied_cells']:4d} "
                  f"reachable={entry['reachable_fraction']:.4f}")
        print(f"wrote {arguments.maps}")
        return

    maps = load_maps(arguments.maps)
    if arguments.action == "verify":
        raise SystemExit(0 if verify_branch(maps, arguments) else 1)

    wanted = tuple(filter(None, arguments.arms.split(","))) or FINAL_ARMS
    all_groups = list(groups(maps, seeds, wanted))
    if arguments.action == "plan":
        cells = sum(len(lanes) for _, _, lanes in all_groups)
        print(f"{len(all_groups)} groups, {cells} cells, {len(maps)} maps, "
              f"{len(seeds)} seeds, {arguments.steps} steps")
        print(f"at 13 s/cell that is {cells * 13 / 3600:.1f} h")
        return

    done = completed(arguments.output)
    cache: dict = {}
    for label, execution, lanes in all_groups:
        if arguments.stop_file.exists():
            print(f"stop file {arguments.stop_file} present -- exiting cleanly", flush=True)
            break
        keys = [identity(lane, arguments.steps, arguments.hardware, execution)
                for lane in lanes]
        present = [key in done for key in keys]
        if all(present):
            print(f"SKIP [{label}] {len(lanes)} lanes already complete", flush=True)
            continue
        if any(present):
            raise SystemExit(
                f"[{label}] is partially present ({sum(present)}/{len(lanes)}). A batched "
                "group must be whole -- delete its rows and re-run the group."
            )
        rows = run_group(label, execution, lanes, cache, arguments)
        append_rows(arguments.output, rows)
        done.update(keys)
    print("campaign driver done", flush=True)


if __name__ == "__main__":
    main()

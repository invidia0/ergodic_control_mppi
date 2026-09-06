"""Measure every quantity Sec. "guarantees" names, on the campaign maps it is claimed for.

The analysis is one chain of inequalities, and each link has two computable sides:

    eps_avg, eps_FM --Prop.4--> eps_track --Thm.2--> TV(rho*, p*) --Prop.3--> E_K

This driver flies the shipped profile on the campaign's nine pillar fields, records the
per-step error budget by strided replay, and scores each cell's stationary coverage error in
the *ball* metric the propositions are actually stated in. It reports both sides of every
inequality plus the slack, so the paper can say where the budget is spent and where the
bounds are loose rather than merely asserting they hold.

    uv run python scripts/theory_audit.py run          # the audit, resumable
    uv run python scripts/theory_audit.py assumptions  # the exactly-checkable conditions

Branch discipline is the campaign's: every cell compared against another must come from one
lane width, so a partial resume at a different width is refused rather than pooled.
"""

import argparse
import csv
import json
import socket
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.common import (
    artifact_digests, ensure_bundle, execution_record, fingerprint, numerical_record, verified_rows,
)
from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.theory_audit import (
    RESIDUAL_FIELDS,
    endpoint_jacobian,
    ideal_batch,
    residual_batch,
)
from ergodic_control_mppi.experiments.uav_pillar_tuning import PREFLIGHT_STEPS, _grid_config
from ergodic_control_mppi.metrics.ergodicity import (
    compute_ball_ergodic_metric,
    compute_team_occupancy_grid,
)
from ergodic_control_mppi.mppi.field import responsibility_gaps
from ergodic_control_mppi.mppi.single import stack_params
from ergodic_control_mppi.simulation import controller_key, select_device

MAP_MANIFEST = Path("results/uav/campaign_maps.json")
DEFAULT_OUTPUT = Path("results/uav/theory_audit.csv")
STOP_FILE = Path("results/uav/STOP")

#: Largest ball radius in the multiscale metric. Half the workspace's short side: past this
#: every ball covers the whole free space and the integrand is identically zero.
MAX_RADIUS = 5.0

FIELDS = [
    "map_seed", "obs_num", "seed", "steps", "stride", "samples",
    *RESIDUAL_FIELDS,
    "eps_track_p95", "slack_k0", "slack_full",
    "ball_ergodic", "tv", "l1", "kl", "reachable_area", "bound_sup",
    "bound_tv", "bound_l1", "bound_kl", "bound_l1_matches_tv", "kl_tail_share",
    "slack_bound_tv", "slack_bound_l1", "slack_bound_kl",
    "c_realized", "outside_fraction", "max_excursion_m", "max_speed",
    "projected_fraction", "inside_obstacle_fraction",
    "wall_seconds", "device", "hardware", "execution", "lanes", "jax_version",
    "inits", "init_x", "init_y", "start_index", "config_hash", "bundle_hash",
    "tv_rows", "tv_columns",
]


# --------------------------------------------------------------------------- scoring


def coverage_terms(
    positions: np.ndarray, arrays, limits_x, limits_y
) -> dict[str, float]:
    """Score one trajectory's stationary coverage error and every Prop. 3 bound.

    The bound is the *lens* bound, not the sup bound. Writing ``mu = rho* - p*`` for the
    signed difference and expanding the ball integral as a convolution of indicators,

        int_Omega mu(B(z,r))^2 dz = iint |Omega cap B(x,r) cap B(y,r)| dmu(x) dmu(y)
                                  <= pi_d r^d |mu|(Omega)^2 = 4 pi_d r^d TV^2,

    since the lens kernel is a positive-definite convolution of indicators and the integrand
    is non-negative, so restricting to ``Omega`` only lowers it. Integrating ``r`` over
    ``[0, R]`` gives ``(4 pi_d R^(d+1) / (d+1)) TV^2``. Two things follow: at ``d = 2``,
    ``R = 5`` the constant is ``4 pi R^3 / 3 = 524`` against the sup bound's ``|Omega| R``
    = 4000, a 7.6x tightening; and the workspace area drops out entirely, so the bound no
    longer degrades as ``Omega`` grows.

    The text then rewrites that via the TV--L1 identity and relaxes it via Pinsker. Those
    are *not* three independent bounds: since ``TV = ||.||_1 / 2`` whenever a density
    exists, the L1 form is the TV bound in other symbols -- equal here to machine precision,
    which ``bound_l1_matches_tv`` asserts rather than merely reporting twice. Only KL is a
    distinct functional, and being a Pinsker *relaxation* of the same bound it can never be
    tighter; measured, it is 3--4x looser. So TV is the best of the three, not the worst.
    """
    target = np.asarray(arrays["target_grid"], dtype=np.float64)
    mask = np.asarray(arrays["reachable_mask"], dtype=bool)
    bins = (target.shape[1], target.shape[0])
    occupancy = compute_team_occupancy_grid(positions, limits_x, limits_y, bins)

    visited = occupancy * mask
    desired = target * mask
    visited = visited / visited.sum() if visited.sum() > 0 else visited
    desired = desired / desired.sum()

    difference = visited - desired
    l1 = float(np.abs(difference).sum())
    total_variation = 0.5 * l1
    # KL is finite only where the trajectory went; cells with no visits contribute nothing to
    # KL(rho||p) since 0 log 0 = 0. Guard the target instead, which is positive on the mask.
    support = visited > 0
    kl_terms = visited[support] * np.log(visited[support] / desired[support])
    kl = float(np.sum(kl_terms))
    # KL divides by the target, and a Gaussian mixture's tails are genuinely tiny -- 4.1e-14
    # at the far corner here, an ordinary float64 and not an underflow. That is KL behaving
    # as defined rather than misbehaving: it is a support-mismatch divergence, so visits to
    # near-zero-target cells are supposed to cost heavily. The practical consequence is that
    # KL is strongly resolution-dependent, since coarsening merges those cells into fatter
    # ones: measured on map 513, KL runs 5.82 -> 2.41 from 80x80 to 5x5 while this tail share
    # runs 63% -> 0%. So the KL slack is only meaningful with its resolution and this share
    # quoted beside it. TV moves too (0.85 -> 0.58) but less, and for a different reason --
    # it has no ratio, so a cell contributes at most its own mass however small the target.
    kl_tail = float(np.sum(kl_terms[desired[support] < 1e-6]))

    ball = compute_ball_ergodic_metric(
        occupancy, target, limits_x, limits_y,
        max_radius=MAX_RADIUS, radii=32, reachable_mask=mask,
    )
    # The lens constant 4 pi_d R^(d+1) / (d+1) at d = 2, where pi_2 = pi. The workspace area
    # does not appear: the lens bound is over R^d and restriction to Omega only lowers it.
    # `area` is still reported so the sup bound this replaces stays reconstructible from the
    # CSV -- the tightening is a claim, and a claim needs its comparator.
    area = (limits_x[1] - limits_x[0]) * (limits_y[1] - limits_y[0]) * float(mask.mean())
    scale = 4.0 * np.pi * MAX_RADIUS ** 3 / 3.0
    bounds = {
        "bound_tv": scale * total_variation ** 2,
        "bound_l1": scale * l1 * l1 / 4.0,
        "bound_kl": scale * kl / 2.0,
    }
    terms = {"ball_ergodic": ball, "tv": total_variation, "l1": l1, "kl": kl,
             "kl_tail_share": kl_tail / kl if kl > 0 else float("nan"),
             "reachable_area": area, "tv_rows": target.shape[0], "tv_columns": target.shape[1],
             # The withdrawn sup bound, kept as the comparator for the lens tightening.
             "bound_sup": area * MAX_RADIUS * total_variation ** 2, **bounds}
    # The L1 and TV forms are one inequality in two notations. Keep both columns -- the
    # equality is worth exhibiting -- but treat any drift between them as a bug in the
    # restriction/renormalization above, which is the only way they could come apart.
    terms["bound_l1_matches_tv"] = bool(
        np.isclose(bounds["bound_l1"], bounds["bound_tv"], rtol=1e-12, atol=0.0)
    )
    for name, value in bounds.items():
        terms[f"slack_{name}"] = value / ball if ball > 0 else float("nan")
    return terms


def invariance_terms(path: np.ndarray, limits_x, limits_y) -> dict[str, float]:
    """As. "compactness": did the executed state ever leave the admissible workspace?"""
    position = path[:, :2]
    beyond = np.maximum.reduce([
        limits_x[0] - position[:, 0], position[:, 0] - limits_x[1],
        limits_y[0] - position[:, 1], position[:, 1] - limits_y[1],
    ])
    return {
        "outside_fraction": float(np.mean(beyond > 0.0)),
        "max_excursion_m": float(max(beyond.max(), 0.0)),
        "max_speed": float(np.linalg.norm(path[:, 2:4], axis=1).max()),
    }


def score_cell(entry: dict, seed: int, path: np.ndarray, residuals: np.ndarray,
               arrays, config, arguments, wall: float, device: str) -> dict:
    """Fold one lane's path and residual history into a report row."""
    limits_x = tuple(float(v) for v in config.controller.workspace.x_limits)
    limits_y = tuple(float(v) for v in config.controller.workspace.y_limits)
    row = {
        "map_seed": entry["map_seed"], "obs_num": entry["obs_num"], "seed": seed,
        "steps": arguments.steps, "stride": arguments.stride,
        "samples": int(residuals.shape[0]),
        "start_index": getattr(arguments, "start_index", None),
        "config_hash": arguments.config_hashes[(entry["obs_num"], entry["map_seed"], seed)],
        "bundle_hash": arguments.bundle_hash,
        "wall_seconds": round(wall, 3), "device": device, "inits": arguments.inits,
        # Recomputed rather than threaded through: it is a pure function of the arguments
        # already here, and recording the actual start is what makes the init groups
        # recoverable from the CSV alone.
        **dict(zip(("init_x", "init_y"),
                   (float(v) for v in dispersed_initial_state(arrays, entry, seed, arguments, config)[:2]))),
        "hardware": arguments.hardware, "execution": f"batch{arguments.lanes}",
        "lanes": arguments.lanes, "jax_version": jax.__version__,
    }
    means = residuals.mean(axis=0)
    row.update({name: float(means[index]) for index, name in enumerate(RESIDUAL_FIELDS)})
    track = residuals[:, RESIDUAL_FIELDS.index("eps_track")]
    row["eps_track_p95"] = float(np.percentile(track, 95))
    for form in ("k0", "full"):
        right = residuals[:, RESIDUAL_FIELDS.index(f"rhs_{form}")]
        row[f"slack_{form}"] = float(np.mean(right / np.maximum(track, 1e-30)))

    row.update(coverage_terms(path[:, :2], arrays, limits_x, limits_y))
    row.update(invariance_terms(path, limits_x, limits_y))
    from ergodic_control_mppi.deploy.grid import clearance_along

    clearance = clearance_along(np.asarray(arrays["occupancy"]),
                                tuple(np.asarray(arrays["grid_origin"])),
                                float(arrays["grid_resolution"]), path[:, :2])
    if np.any(clearance < 0.30) or row["outside_fraction"] > 0:
        raise ValueError("adopted profile collided or left the workspace; audit stopped")
    # Thm. 2 asserts a constant c = L_ker/alpha_m exists with TV <= c sqrt(eps_track). We do
    # not estimate its two factors separately -- that needs kernel-TV estimates in a 6+2P
    # dimensional space, where the estimator would dominate. We report the value the
    # inequality actually realizes here, which is the smallest constant consistent with this
    # cell. Stability of it across clutter is the finding; a drift means the assumption
    # strains where the clutter is.
    row["c_realized"] = row["tv"] / np.sqrt(max(row["eps_track"], 1e-30))
    return row


# --------------------------------------------------------------------------- running


def load_maps(path: Path) -> list[dict]:
    """Load the campaign maps, asserting the invariants batching depends on."""
    entries = json.loads(path.read_text(encoding="utf-8"))["maps"]
    shapes = {tuple(e["grid_shape"]) for e in entries}
    if len(shapes) != 1:
        raise SystemExit(f"maps disagree on grid shape: {shapes}. Lanes cannot be stacked.")
    starts = {tuple(round(v, 6) for v in e["initial_state"]) for e in entries}
    if len(starts) != 1:
        raise SystemExit(f"maps disagree on start state: {starts}")
    return entries


def completed(output: Path) -> set:
    """Identities already in the archive. The lane width is part of the key, not metadata."""
    columns = ("map_seed", "obs_num", "seed", "steps", "stride", "hardware", "execution",
               "inits", "start_index", "config_hash")
    return {tuple(r[k] for k in columns) for r in verified_rows(output, columns)}


def append_rows(output: Path, rows: list[dict]) -> None:
    """Append a whole group at once, header-checked against ``FIELDS``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (output.exists() and output.stat().st_size)
    if not fresh:
        with output.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
        if header != FIELDS:
            raise SystemExit(
                f"{output} has a stale header. Appending would misalign every row -- "
                "migrate or move the file first."
            )
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def dispersed_initial_state(arrays, entry, seed: int, arguments, config) -> np.ndarray:
    """Pick this lane's start. With ``--inits 1`` (default) every lane keeps the archived one.

    Thm. 1 claims convergence "for every initial reduced state", but a campaign that starts
    every lane from one point cannot see that -- it varies only the noise. With ``--inits n``
    the seeds are partitioned into ``n`` groups, each given a different free-space start, so
    between-init and within-init spread can be compared at equal K. The lane *count* is
    untouched, which keeps the vmap width, and therefore the numerical branch, identical to
    every group already collected.
    """
    state = np.asarray(arrays["initial_state"], dtype=np.float64).copy()
    start_index = getattr(arguments, "start_index", None)
    if arguments.inits < 1 or (start_index is not None and not 0 <= start_index < arguments.inits):
        raise ValueError("start-index must satisfy 0 <= start-index < inits, with inits >= 1")
    if arguments.inits <= 1:
        return state
    group = (seed * arguments.inits // max(arguments.seeds, 1)
             if start_index is None else start_index)
    mask = np.asarray(arrays["reachable_mask"], dtype=bool)
    x_limits = np.asarray(config.controller.workspace.x_limits, dtype=np.float64)
    y_limits = np.asarray(config.controller.workspace.y_limits, dtype=np.float64)
    origin = np.array([x_limits[0], y_limits[0]])
    # Cell centres of the free space, ordered, then sampled deterministically per group so a
    # rerun reproduces the same starts.
    rows, columns = np.nonzero(mask)
    extent = np.array([np.diff(x_limits)[0] / mask.shape[1],
                       np.diff(y_limits)[0] / mask.shape[0]])
    if len(rows) < arguments.inits:
        raise ValueError("fewer free cells than requested initial positions")
    centres = origin + (np.stack([columns, rows], axis=1) + 0.5) * extent
    picker = np.random.default_rng(abs(hash((int(entry["map_seed"]), int(entry["obs_num"])))) % 2**32)
    choice = picker.permutation(len(centres))[group % len(centres)]
    state[:2] = centres[choice]
    return state


def audit_inputs(arguments, kind: str):
    """Resolve every lane, check provenance and require an intact CSV/path pair on resume."""
    maps = load_maps(arguments.maps)
    lanes = [(entry, seed) for entry in maps for seed in range(arguments.seeds)]
    if not lanes:
        raise ValueError("audit requires maps and seeds")
    arguments.lanes = len(lanes)
    device = select_device(arguments.device)
    cache = {}
    configs, arrays_per_lane = [], []
    records = {}
    arguments.config_hashes = {}
    for entry, seed in lanes:
        key = (entry["obs_num"], entry["map_seed"])
        if key not in cache:
            cache[key] = _grid_config(Path(entry["run_dir"]), arguments.config)
        config, manifest, arrays = cache[key]
        configs.append(config)
        arrays_per_lane.append(arrays)
        record = numerical_record({
            "controller": config.controller, "arrays": arrays, "scoring": manifest,
            "initial_state": dispersed_initial_state(arrays, entry, seed, arguments, config),
            "steps": arguments.steps, "stride": arguments.stride, "inits": arguments.inits,
            "start_index": getattr(arguments, "start_index", None), "kind": kind,
        })
        digest = fingerprint(record)
        arguments.config_hashes[(*key, seed)] = digest
        records[digest] = record
    arguments.bundle_hash = ensure_bundle(arguments.output, {
        "configurations": records, "maps": maps, "seeds": list(range(arguments.seeds)),
        "width": len(lanes), "kind": kind,
        "execution": execution_record("scripts/theory_audit.py", str(device)),
    }, getattr(arguments, "overwrite", False))
    done = completed(arguments.output)
    expected = {
        (str(e["map_seed"]), str(e["obs_num"]), str(seed), str(arguments.steps),
         str(arguments.stride), arguments.hardware, f"batch{len(lanes)}", str(arguments.inits),
         "" if getattr(arguments, "start_index", None) is None else str(arguments.start_index),
         arguments.config_hashes[(e["obs_num"], e["map_seed"], seed)]) for e, seed in lanes
    }
    path_file = arguments.output.with_name(arguments.output.stem + "_paths.npz")
    if done:
        if done != expected or not path_file.exists():
            raise ValueError("partial or incompatible audit CSV/path pair; use --overwrite or a fresh path")
        receipt = arguments.output.with_suffix(".artifacts.json")
        if not receipt.exists() or json.loads(receipt.read_text()) != artifact_digests([arguments.output, path_file]):
            raise ValueError("audit artifacts are incomplete or changed")
        with np.load(path_file, allow_pickle=False) as bundle:
            if str(bundle["bundle_hash"]) != arguments.bundle_hash:
                raise ValueError("audit paths belong to a different bundle")
            actual = list(zip(bundle["obs_num"], bundle["map_seed"], bundle["seed"]))
            if actual != [(e["obs_num"], e["map_seed"], seed) for e, seed in lanes]:
                raise ValueError("audit path lane identities do not match CSV")
            if bundle["positions"].shape != (len(lanes), arguments.steps, 2):
                raise ValueError("audit path shape does not match requested run")
        print(f"all {len(lanes)} verified cells already in {arguments.output}")
        return None
    if path_file.exists():
        raise ValueError("orphan audit paths; use --overwrite or a fresh path")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    return lanes, device, configs, arrays_per_lane


def run(arguments) -> None:
    """Fly every cell at one lane width and append the audit rows."""
    if STOP_FILE.exists():
        raise SystemExit(f"{STOP_FILE} exists; refusing to start")
    prepared = audit_inputs(arguments, "run")
    if prepared is None:
        return
    lanes, device, configs, arrays_per_lane = prepared
    stacked = stack_params([jax.device_put(c.controller, device) for c in configs])
    keys = jnp.stack([controller_key(seed) for _, seed in lanes])
    initial = jnp.asarray(
        np.stack([
            dispersed_initial_state(arrays_per_lane[index], entry, seed, arguments, configs[index])
            for index, (entry, seed) in enumerate(lanes)
        ]),
        dtype=jnp.float32,
    )
    controls = jnp.zeros((configs[0].controller.mppi.horizon, 3), dtype=jnp.float32)

    print(f"[audit] {len(lanes)} lanes x {arguments.steps} steps, "
          f"stride {arguments.stride}; compiling...", flush=True)
    started = time.perf_counter()
    paths, residuals = jax.jit(
        residual_batch, static_argnames=("steps", "stride", "preflight_steps")
    )(
        stacked, initial, controls, keys,
        steps=arguments.steps, stride=arguments.stride, preflight_steps=PREFLIGHT_STEPS,
    )
    jax.block_until_ready(paths)
    wall = time.perf_counter() - started
    paths = np.asarray(paths)
    if not np.isfinite(paths).all():
        raise ValueError("nonfinite audit trajectory")
    residuals = np.asarray(residuals)
    print(f"[audit] {wall:.0f}s total, {wall / len(lanes):.1f}s per cell", flush=True)

    rows = [
        score_cell(entry, seed, paths[index], residuals[index], arrays_per_lane[index],
                   configs[index], arguments, wall / len(lanes), device.platform)
        for index, (entry, seed) in enumerate(lanes)
    ]
    # Keep the executed positions. Every resolution/K diagnostic on the TV estimator is
    # pure post-processing over these, so dumping them here is what makes those sweeps cost
    # numpy seconds instead of another GPU campaign. Positions only, float32: 17 MB at 108
    # lanes x 20k steps.
    path_file = arguments.output.with_name(arguments.output.stem + "_paths.npz")
    np.savez_compressed(
        path_file,
        positions=paths[:, :, :2].astype(np.float32),
        map_seed=np.array([e["map_seed"] for e, _ in lanes]),
        obs_num=np.array([e["obs_num"] for e, _ in lanes]),
        seed=np.array([s for _, s in lanes]),
        steps=arguments.steps, bundle_hash=arguments.bundle_hash,
        config_hash=np.array([arguments.config_hashes[(e["obs_num"], e["map_seed"], seed)]
                              for e, seed in lanes]),
    )
    print(f"[audit] wrote {path_file} "
          f"({path_file.stat().st_size / 1e6:.0f} MB, {paths.shape[1]} steps/lane)", flush=True)

    append_rows(arguments.output, rows)
    arguments.output.with_suffix(".artifacts.json").write_text(
        json.dumps(artifact_digests([arguments.output, path_file]), indent=2) + "\n")
    summarize(rows)


def _inside_obstacle_fraction(positions, arrays, limits_x, limits_y) -> float:
    """Share of a path's samples falling in cells the reachable mask excludes."""
    mask = np.asarray(arrays["reachable_mask"], dtype=bool)
    rows, columns = mask.shape
    column = np.clip(((positions[:, 0] - limits_x[0]) / (limits_x[1] - limits_x[0])
                      * columns).astype(int), 0, columns - 1)
    row = np.clip(((positions[:, 1] - limits_y[0]) / (limits_y[1] - limits_y[0])
                   * rows).astype(int), 0, rows - 1)
    return float(np.mean(~mask[row, column]))


def ideal(arguments) -> None:
    """Fly the As. 7 comparison kernel and ask whether its visitation law is ``p*``.

    Cor. "flow_matching_consistency" says perfect tracking gives exact coverage. That rests
    entirely on As. 7's clause that P_0's invariant measure has spatial marginal ``p*``, which
    the paper states and does not establish. This flies a controller with ``eps_track``
    identically zero and measures the TV its stationary law actually realizes against ``p*``.
    A value near zero supports the clause; a value near the flown controller's says perfect
    tracking would *still* miss, and the corollary is idealizing away nothing.
    """
    prepared = audit_inputs(arguments, "ideal")
    if prepared is None:
        return
    lanes, device, configs, arrays_per_lane = prepared
    stacked = stack_params([jax.device_put(c.controller, device) for c in configs])
    keys = jnp.stack([controller_key(seed) for _, seed in lanes])
    initial = jnp.asarray(
        np.stack([dispersed_initial_state(arrays_per_lane[i], e, s, arguments, configs[i])
                  for i, (e, s) in enumerate(lanes)]), dtype=jnp.float32
    )
    controls = jnp.zeros((configs[0].controller.mppi.horizon, 3), dtype=jnp.float32)

    print(f"[ideal] {len(lanes)} lanes x {arguments.steps} steps; compiling...", flush=True)
    started = time.perf_counter()
    paths = jax.jit(ideal_batch, static_argnames=("steps", "preflight_steps"))(
        stacked, initial, controls, keys,
        steps=arguments.steps, preflight_steps=PREFLIGHT_STEPS,
    )
    jax.block_until_ready(paths)
    wall = time.perf_counter() - started
    paths = np.asarray(paths)
    if not np.isfinite(paths).all():
        raise ValueError("nonfinite audit trajectory")
    print(f"[ideal] {wall:.0f}s total, {wall / len(lanes):.1f}s per cell", flush=True)

    rows = []
    for index, (entry, seed) in enumerate(lanes):
        config = configs[index]
        limits_x = tuple(float(v) for v in config.controller.workspace.x_limits)
        limits_y = tuple(float(v) for v in config.controller.workspace.y_limits)
        row = {"map_seed": entry["map_seed"], "obs_num": entry["obs_num"], "seed": seed,
               "steps": arguments.steps, "stride": arguments.stride, "samples": arguments.steps,
               "wall_seconds": round(wall / len(lanes), 3), "device": device.platform,
               "inits": arguments.inits, "hardware": arguments.hardware,
               "execution": f"batch{len(lanes)}", "lanes": len(lanes),
               "jax_version": jax.__version__,
               "start_index": getattr(arguments, "start_index", None),
               "config_hash": arguments.config_hashes[(entry["obs_num"], entry["map_seed"], seed)],
               "bundle_hash": arguments.bundle_hash,
               **dict(zip(("init_x", "init_y"), map(float,
                   dispersed_initial_state(arrays_per_lane[index], entry, seed, arguments, config)[:2])))}
        row.update({name: float("nan") for name in RESIDUAL_FIELDS})
        # eps_track is zero wherever the projection is inactive, which is the definition of
        # this kernel. Where the field points out of the admissible set the two cannot both
        # hold, and what the projection costs is exactly this residual -- so measure it
        # rather than assert it. The path records both the realized motion and the reference
        # velocity that produced it, so no extra state was needed to recover it.
        lane = paths[index]
        realized = np.diff(lane[:, :2], axis=0) / float(config.controller.model.delta_t)
        # lane[k, 2:4] is the flow that *produced* the move into k, so the reference for the
        # step from k-1 to k is lane[k], not lane[k-1]. Off by one here silently turns a
        # near-zero residual into an O(1) one, since consecutive flows differ.
        residual = np.sum((realized - lane[1:, 2:4]) ** 2, axis=1)
        row["eps_track"] = float(residual.mean())
        row["eps_track_p95"] = float(np.percentile(residual, 95))
        # 1e-6 sits far above float32 round-trip noise (position rounds at ~1e-6 m,
        # so the differenced velocity carries ~1e-9 of squared residual) and far below
        # a real projection event, which clips an O(1 m/s) component.
        row["projected_fraction"] = float(np.mean(residual > 1e-6))
        # The box clip does not exclude pillars on rasterized maps. TV already masks those
        # cells away, so this does not bias it -- but a law that lives inside obstacles would
        # make the TV a statement about a small free-space remnant, so report the share.
        row["inside_obstacle_fraction"] = _inside_obstacle_fraction(
            lane[:, :2], arrays_per_lane[index], limits_x, limits_y
        )
        row.update(coverage_terms(paths[index][:, :2], arrays_per_lane[index], limits_x, limits_y))
        row.update(invariance_terms(paths[index], limits_x, limits_y))
        row["c_realized"] = float("nan")
        rows.append(row)

    path_file = arguments.output.with_name(arguments.output.stem + "_paths.npz")
    np.savez_compressed(
        path_file, positions=paths[:, :, :2].astype(np.float32),
        map_seed=np.array([e["map_seed"] for e, _ in lanes]),
        obs_num=np.array([e["obs_num"] for e, _ in lanes]),
        seed=np.array([s for _, s in lanes]), steps=arguments.steps,
        bundle_hash=arguments.bundle_hash,
        config_hash=np.array([arguments.config_hashes[(e["obs_num"], e["map_seed"], seed)]
                              for e, seed in lanes]),
    )
    append_rows(arguments.output, rows)
    arguments.output.with_suffix(".artifacts.json").write_text(
        json.dumps(artifact_digests([arguments.output, path_file]), indent=2) + "\n")
    print("\n--- ideal-flow coverage (median over cells) " + "-" * 26)
    print(f"{'density':>8} {'n':>4} {'TV vs p*':>10} {'E_K':>10} {'outside':>9} "
          f"{'proj frac':>10} {'eps_track':>10} {'in obst':>10}")
    for density in sorted({r["obs_num"] for r in rows}):
        subset = [r for r in rows if r["obs_num"] == density]
        stat = lambda name: float(np.median([r[name] for r in subset]))  # noqa: E731
        print(f"{density:>8} {len(subset):>4} {stat('tv'):>10.4f} {stat('ball_ergodic'):>10.3f} "
              f"{stat('outside_fraction'):>9.3g} {stat('projected_fraction'):>10.4f} "
              f"{stat('eps_track'):>10.4g} {stat('inside_obstacle_fraction'):>10.4f}")
    print(f"\nwrote {arguments.output} and {path_file}")


# ----------------------------------------------------------------- TV estimator sweep


def _coarsen(grid: np.ndarray, factor: int, how: str) -> np.ndarray:
    """Block-reduce a square grid by an integer factor: sum for mass, any for a mask."""
    n = grid.shape[0] // factor
    blocks = grid[: n * factor, : n * factor].reshape(n, factor, n, factor)
    return blocks.sum(axis=(1, 3)) if how == "sum" else blocks.any(axis=(1, 3))


def _tv(positions: np.ndarray, target: np.ndarray, mask: np.ndarray,
        limits_x, limits_y) -> float:
    """TV between a path's binned occupancy and the target, both restricted and renormalized.

    Same convention as :func:`coverage_terms`, so a sweep value at the native resolution
    reproduces the ``tv`` column rather than merely resembling it.
    """
    bins = (target.shape[1], target.shape[0])
    occupancy = compute_team_occupancy_grid(positions, limits_x, limits_y, bins)
    visited = occupancy * mask
    desired = target * mask
    if visited.sum() <= 0:
        return float("nan")
    return float(0.5 * np.abs(visited / visited.sum() - desired / desired.sum()).sum())


def sweep(arguments) -> None:
    """Map the TV estimator's bias over (grid resolution, K), and bracket the true value.

    Three independent handles on the same question, in increasing strength:

    * a **planted null** -- draw K iid samples from ``p*`` itself, where the answer is known
      to be zero, so whatever the pipeline reports is pure bias. Understates the bias for a
      correlated path, which is why it is not the last word;
    * a **split-half** between seeds -- two runs on one map both sample the same stationary
      law, so the TV between them is sampling noise carrying the real autocorrelation, and
      ``p*`` never enters. No model, no fit;
    * a **K-ladder by prefix** -- the first K steps of one trajectory *are* the estimator at
      sample size K, so one deep run gives the whole ladder with no run-to-run variation.

    The triangle inequality then brackets the quantity the theorems name:
    ``|TV(rho_K, p*) - TV(rho*, p*)| <= TV(rho_K, rho*)``, the last estimated by split-half.
    """
    bundle = np.load(arguments.paths, allow_pickle=False)
    source_csv = arguments.paths.with_name(arguments.paths.stem.removesuffix("_paths") + ".csv")
    receipt = source_csv.with_suffix(".artifacts.json")
    if not receipt.exists() or json.loads(receipt.read_text()) != artifact_digests([source_csv, arguments.paths]):
        raise ValueError("sweep source artifacts are incomplete or changed")
    source_rows = verified_rows(source_csv, ("obs_num", "map_seed", "seed"))
    if not source_rows or "bundle_hash" not in bundle or {r["bundle_hash"] for r in source_rows} != {str(bundle["bundle_hash"])}:
        raise ValueError("sweep requires a verified matching CSV/path bundle")
    source_manifest = json.loads(source_csv.with_suffix(".manifest.json").read_text())
    for row in source_rows:
        recorded = source_manifest["inputs"]["configurations"][row["config_hash"]]
        entry = next(e for e in load_maps(arguments.maps)
                     if str(e["map_seed"]) == row["map_seed"] and str(e["obs_num"]) == row["obs_num"])
        config, _, arrays = _grid_config(Path(entry["run_dir"]), arguments.config)
        if recorded["controller"] != numerical_record(config.controller) or recorded["arrays"] != numerical_record(arrays):
            raise ValueError("sweep config/maps differ from the path-producing inputs")
    out = arguments.output
    split_out = out.with_name(out.stem + "_split.csv")
    artifact_receipt = out.with_suffix(".artifacts.json")
    record = {"source_bundle": str(bundle["bundle_hash"]),
              "source_paths": fingerprint(dict(bundle)),
              "execution": execution_record("scripts/theory_audit.py", "offline")}
    overwrite = getattr(arguments, "overwrite", False)
    bundle_hash = ensure_bundle(out, record, overwrite)
    if ensure_bundle(split_out, record, overwrite) != bundle_hash:
        raise AssertionError("split output bundle differs from primary output")
    if overwrite:
        artifact_receipt.unlink(missing_ok=True)
    existing = (out.exists(), split_out.exists(), artifact_receipt.exists())
    if any(existing):
        if all(existing) and json.loads(artifact_receipt.read_text()) == artifact_digests([out, split_out]):
            verified_rows(out, ("obs_num", "map_seed", "seed", "factor", "k"))
            verified_rows(split_out, ("obs_num", "map_seed", "seed_a", "seed_b", "factor", "k"))
            print(f"[sweep] verified existing {out} and {split_out}")
            return
        raise ValueError("sweep outputs are incomplete or changed; use --overwrite")
    positions = bundle["positions"]
    map_seed, obs_num, seed = bundle["map_seed"], bundle["obs_num"], bundle["seed"]
    k_max = positions.shape[1]
    ladder = [k for k in (5000, 10000, 20000, 40000, 100000, 200000, 400000) if k <= k_max]
    if not ladder or ladder[-1] != k_max:
        ladder.append(k_max)
    factors = [f for f in (1, 2, 4, 5, 8, 10, 16) if f <= 80]
    rng = np.random.default_rng(0)

    maps = {(e["map_seed"], e["obs_num"]): e for e in load_maps(arguments.maps)}
    cache: dict = {}
    for key, entry in maps.items():
        config, _, arrays = _grid_config(Path(entry["run_dir"]), arguments.config)
        cache[key] = (
            np.asarray(arrays["target_grid"], dtype=np.float64),
            np.asarray(arrays["reachable_mask"], dtype=bool),
            tuple(float(v) for v in config.controller.workspace.x_limits),
            tuple(float(v) for v in config.controller.workspace.y_limits),
        )

    print(f"[sweep] {len(positions)} lanes, K up to {k_max}, "
          f"{len(factors)} resolutions x {len(ladder)} K values", flush=True)

    rows = []
    for index in range(len(positions)):
        key = (int(map_seed[index]), int(obs_num[index]))
        target, mask, limits_x, limits_y = cache[key]
        for factor in factors:
            tgt = _coarsen(target, factor, "sum")
            msk = _coarsen(mask, factor, "any")
            cells = int(msk.sum())
            probability = (tgt * msk).ravel()
            probability = probability / probability.sum()
            for k in ladder:
                null = np.mean([
                    0.5 * np.abs(rng.multinomial(k, probability) / k - probability).sum()
                    for _ in range(4)
                ])
                rows.append({
                    "map_seed": key[0], "obs_num": key[1], "seed": int(seed[index]),
                    "factor": factor, "grid": tgt.shape[0], "cells": cells, "k": k,
                    "tv": _tv(positions[index, :k], tgt, msk, limits_x, limits_y),
                    "tv_null_iid": float(null), "bundle_hash": bundle_hash,
                })

    # Split-half: seed pairs on one map, same K and resolution. Both sample the same
    # stationary law (Thm. 1), so this is estimator noise with the real correlation in it.
    halves = []
    for key in maps:
        members = [i for i in range(len(positions))
                   if (int(map_seed[i]), int(obs_num[i])) == key]
        target, mask, limits_x, limits_y = cache[key]
        for factor in factors:
            tgt = _coarsen(target, factor, "sum")
            msk = _coarsen(mask, factor, "any")
            bins = (tgt.shape[1], tgt.shape[0])
            for k in ladder:
                grids = []
                for i in members:
                    occupancy = compute_team_occupancy_grid(
                        positions[i, :k], limits_x, limits_y, bins
                    ) * msk
                    grids.append(occupancy / occupancy.sum())
                for a in range(len(grids)):
                    for b in range(a + 1, len(grids)):
                        halves.append({
                            "map_seed": key[0], "obs_num": key[1],
                            "seed_a": int(seed[members[a]]), "seed_b": int(seed[members[b]]),
                            "factor": factor,
                            "grid": tgt.shape[0], "cells": int(msk.sum()), "k": k,
                            "tv_split": float(0.5 * np.abs(grids[a] - grids[b]).sum()),
                            "bundle_hash": bundle_hash,
                        })

    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with split_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(halves[0]))
        writer.writeheader()
        writer.writerows(halves)
    artifact_receipt.write_text(json.dumps(artifact_digests([out, split_out]), indent=2) + "\n")
    print(f"[sweep] wrote {out} ({len(rows)} rows) and {split_out} ({len(halves)} rows)")


# --------------------------------------------------------------------------- reporting


def summarize(rows: list[dict]) -> None:
    """Print the error budget by obstacle density, which is how the paper reports it."""
    def stat(subset, name):
        return float(np.median([r[name] for r in subset]))

    print("\n--- error budget (median over cells) " + "-" * 32)
    header = ("density  n   eps_avg     eps_track  eps_FM(k0)  eps_FM(T)   "
              "slack_k0  slack_T    E_K        TV      c_hat")
    print(header)
    for density in sorted({r["obs_num"] for r in rows}):
        subset = [r for r in rows if r["obs_num"] == density]
        print(f"{density:>7}  {len(subset):<3} {stat(subset,'eps_avg'):<11.3g} "
              f"{stat(subset,'eps_track'):<10.4f} {stat(subset,'eps_fm_k0'):<11.4f} "
              f"{stat(subset,'eps_fm_full'):<11.4g} {stat(subset,'slack_k0'):<9.3f} "
              f"{stat(subset,'slack_full'):<10.4g} {stat(subset,'ball_ergodic'):<10.4g} "
              f"{stat(subset,'tv'):<7.4f} {stat(subset,'c_realized'):.4f}")

    holds = all(r["eps_track"] <= r["rhs_k0"] * (1 + 1e-5) for r in rows)
    print(f"\nProp. 4 holds on the cell means: {holds} ({len(rows)} cells)")
    # Two distinct bounds, not three: the L1 form is the TV form rewritten, so it is checked
    # as an identity instead of being quoted as a second result.
    print("Prop. 3 bounds hold:", {
        form: all(r[f"bound_{form}"] >= r["ball_ergodic"] for r in rows)
        for form in ("tv", "kl")
    })
    identical = all(r["bound_l1_matches_tv"] for r in rows)
    ratio = np.median([r["slack_bound_kl"] / r["slack_bound_tv"] for r in rows])
    print(f"Prop. 3 L1 form is the TV form (TV = |.|_1/2): {identical} ({len(rows)} cells)")
    tail = np.median([r["kl_tail_share"] for r in rows])
    print(f"Prop. 3 KL form is a Pinsker relaxation, looser by a median {ratio:.1f}x, "
          f"and {100 * tail:.0f}% of that KL comes from cells where the target is below "
          f"1e-6 (a real Gaussian tail, not an underflow) -- so it moves with resolution")
    outside = max(r["outside_fraction"] for r in rows)
    print(f"As. 1 forward invariance: max outside-fraction {outside:.3g}, "
          f"max excursion {max(r['max_excursion_m'] for r in rows):.4g} m")
    print(f"As. 2 speed gauge regularized: max fraction "
          f"{max(r['gauge_regularized'] for r in rows):.3g}")
    print(f"executed control saturated: max fraction "
          f"{max(r['saturated_fraction'] for r in rows):.3g}")


def assumptions(arguments) -> None:
    """Report the conditions that are checkable exactly, with no simulation."""
    config = load_config(arguments.config)
    params = config.controller
    state = jnp.zeros((6,), jnp.float32)
    print(f"--- exactly-checkable conditions, {arguments.config} " + "-" * 20)
    for horizon in (1, 2, 3):
        jacobian = endpoint_jacobian(params, state, jnp.zeros((horizon, 3), jnp.float32))
        singular = np.linalg.svd(jacobian, compute_uv=False)
        print(f"As. endpoint  n*={horizon}: rank {np.linalg.matrix_rank(jacobian)}/6, "
              f"sigma_min {singular.min():.6g}, sigma_max {singular.max():.6g}")
    saturated = endpoint_jacobian(params, state, jnp.full((2, 3), 1e3, jnp.float32))
    print(f"As. endpoint  saturated witness: rank {np.linalg.matrix_rank(saturated)} "
          "(the assumption asks for an interior one, which is why)")

    covariance = np.asarray(params.mppi.covariance, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)
    print(f"As. sampling  N={params.mppi.samples}, lambda in "
          f"[{params.mppi.temperature_min}, {params.mppi.temperature_max}], "
          f"min eig(Sigma) {eigenvalues.min():.4g}, cond(Sigma) "
          f"{eigenvalues.max() / eigenvalues.min():.4g}")

    gaps = np.asarray(responsibility_gaps(params.gmm), dtype=np.float64)
    ceiling = float(params.field.deficit_ceiling)
    promotion = np.log((ceiling + 1.0) / ceiling) if ceiling > 0 else 0.0
    print(f"Sec. III-E    Delta_j {np.round(gaps, 3).tolist()} nats; promotion capped at "
          f"log((c+1)/c) = {promotion:.3f}, so it cannot overturn the smallest margin "
          f"({gaps.min():.3f}) -- the demotion term is what empties a basin")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "assumptions", "sweep", "ideal"))
    parser.add_argument("--maps", type=Path, default=MAP_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="configs/uav_profile.yaml")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hardware", default=socket.gethostname())
    parser.add_argument("--inits", type=int, default=1,
                        help="distinct free-space start states to partition the seeds over")
    parser.add_argument("--paths", type=Path, default=None,
                        help="sweep: the *_paths.npz written by run")
    parser.add_argument("--start-index", type=int, default=None,
                        help="use one of --inits deterministic starts for every seed")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.seeds, arguments.stride, arguments.inits) < 1:
        parser.error("steps, seeds, stride and inits must be positive")
    if arguments.start_index is not None and not 0 <= arguments.start_index < arguments.inits:
        parser.error("start-index must satisfy 0 <= start-index < inits")
    if arguments.command == "assumptions":
        assumptions(arguments)
    elif arguments.command == "ideal":
        if arguments.output == DEFAULT_OUTPUT:
            arguments.output = Path("results/uav/theory_audit_ideal.csv")
        ideal(arguments)
    elif arguments.command == "sweep":
        if arguments.paths is None:
            raise SystemExit("sweep needs --paths pointing at a *_paths.npz from run")
        if arguments.output == DEFAULT_OUTPUT:
            arguments.output = Path("results/uav/theory_audit_sweep.csv")
        sweep(arguments)
    else:
        run(arguments)


if __name__ == "__main__":
    main()

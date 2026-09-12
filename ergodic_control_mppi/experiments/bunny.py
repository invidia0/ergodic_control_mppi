"""Surface coverage of the Stanford bunny: ours against the ergodic baselines, in 3D.

uv run python -m ergodic_control_mppi.experiments.bunny run [--methods ...]
uv run python -m ergodic_control_mppi.experiments.bunny rescore results/bunny/<run>.csv ..."""

import argparse
import csv
import hashlib
import io
import multiprocessing
import os
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.baselines import (
    BaselineConfig,
    _sves_planner,
    _sves_step,
    _unit_field,
    certificate_columns,
)
from ergodic_control_mppi.experiments.common import (
    append_csv,
    ensure_bundle,
    execution_record,
    verified_rows,
)
from ergodic_control_mppi.experiments.dimension import (
    GRID_PITCH,
    PATH_STRIDE,
    PROFILE,
    ROBOT_RADIUS,
    _normal,
    _track,
    _wall_bias,
    fly_ours,
    lift,
    start_state,
)
from ergodic_control_mppi.experiments.literature_methods import _limit_speed
from ergodic_control_mppi.metrics.ergodicity import fourier_wavenumbers
from ergodic_control_mppi.mppi.field import responsibility_gaps

DRIVER = "ergodic_control_mppi/experiments/bunny.py"
MESH_URL = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
MESH_MEMBER = "bunny/reconstruction/bun_zipper_res2.ply"
MESH_SHA256 = "8faa052bb08cf1625eec97b6508baceee11eec9cf28f83f6a9d4547ab15d7761"
SCALE = 40.0              # the scan is 0.15 m tall; x40 makes it about 6.2 m
# The floor is 1.5 m below the ground, so the boundary margin ends at z = 0, not above it.
LIMITS = ((-7.0, 7.0), (-7.0, 7.0), (-1.5, 9.0))
LOWER = np.array([low for low, _ in LIMITS])
SPAN = np.array([high - low for low, high in LIMITS])
VOXEL = 0.1
SHAPE = tuple(int(round((high - low) / VOXEL)) for low, high in reversed(LIMITS))  # (Z, Y, X)
STANDOFF = 0.75           # the target shell's distance from the surface
SHELL_FLOOR = 0.5         # shell points lower than this sit in the ground's clearance
PATCHES = 24
# m^2 on every patch covariance: the shell is one voxel thick. At 0.04 (0.2 m across the
# shell) the target reaching into the 0.3 m keep-out is 8% of its peak.
PATCH_FLOOR = 0.04
START = (0.0, -5.0, 3.0)
SURFACE_SAMPLES = 200_000
SEEN = STANDOFF + 0.5     # a surface point counts as covered once the path passes this close
CURVE_STRIDE = 10         # coverage times are read on every 10th surface sample
COVER_LEVELS = (50, 90, 95, 99)  # percent of the reachable surface
STEPS = 60_000            # 1200 s: long enough for the coverage times to be read
METHODS = ("ours", "hedac", "fmec", "smc", "sves")
SOLVER_PITCH = 0.25       # the planar baselines' grid pitch
PLANAR_CELLS = 80 * 160   # the planar solver grid, 40 x 20 m at that pitch

FIELDS = ["method", "seed", "mmd_final", "mmd_bound", "mmd_trivial", "mmd_weighted_gap",
          "mmd_prefix_holds", "mmd_beats_trivial", "mmd_samples", "mmd_stride", "collided",
          "contact_fraction", "surface_final", *(f"t{level}" for level in COVER_LEVELS),
          "bundle_hash"]


# ------------------------------------------------------------------------------ scene


def load_mesh(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """The scan's vertices ``(V, 3)`` in metres, z up, base on the ground; faces ``(F, 3)``."""
    local = Path(cache) / Path(MESH_MEMBER).name
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MESH_URL, timeout=60) as response:
            archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
        local.write_bytes(archive.extractfile(MESH_MEMBER).read())
    raw = local.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MESH_SHA256:
        raise ValueError(f"{local} is not the pinned bunny scan; delete it to refetch")
    header, body = raw.decode().split("end_header", 1)
    vertex_count, face_count = (int(header.split(f"element {name} ")[1].split()[0])
                                for name in ("vertex", "face"))
    lines = body.strip().splitlines()
    scan = np.array([line.split()[:3] for line in lines[:vertex_count]], float)
    faces = np.array([line.split()[1:4] for line in lines[vertex_count:vertex_count + face_count]],
                     int)
    vertices = SCALE * scan[:, [0, 2, 1]] * [1.0, -1.0, 1.0]  # y up to z up, a rotation
    vertices -= [*(vertices[:, :2].min(0) + vertices[:, :2].max(0)) / 2, vertices[:, 2].min()]
    return vertices, faces


def surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int,
                    seed: int = 0) -> np.ndarray:
    """Sample ``count`` points uniformly over the mesh surface."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), count, p=area / area.sum())
    u, v = rng.random((2, count))
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return a[pick] + u[:, None] * (b - a)[pick] + v[:, None] * (c - a)[pick]


def _cells(points: np.ndarray, lower=LOWER, shape=SHAPE) -> tuple[np.ndarray, ...]:
    """``(z, y, x)`` voxel indices of ``(N, 3)`` points, clipped to the lattice."""
    cell = np.clip(np.floor((points - lower) / VOXEL).astype(int), 0, np.array(shape[::-1]) - 1)
    return cell[:, 2], cell[:, 1], cell[:, 0]

def body_voxels(samples: np.ndarray) -> np.ndarray:
    """The bunny's solid body on the ``(Z, Y, X)`` lattice."""
    surface = np.zeros(SHAPE, bool)
    surface[_cells(samples)] = True
    return np.stack([binary_fill_holes(level) for level in surface])


def scene(samples: np.ndarray, safe_distance: float):
    """``(body, distance, clearance, blocked)`` voxel arrays."""
    body = body_voxels(samples)
    distance = distance_transform_edt(~body) * VOXEL
    solid = body.copy()
    solid[: int(round(-LOWER[2] / VOXEL))] = True
    clearance = signed_distance(solid)
    # With a tolerance: 3 * 0.1 is 0.30000000000000004 in floating point, so a bare
    # `<= 0.3` silently drops the outermost layer and the margin shrinks by a voxel.
    blocked = clearance <= safe_distance + 1e-6 * VOXEL
    return body, distance, clearance, blocked

def signed_distance(solid: np.ndarray) -> np.ndarray:
    """Distance to a ``(Z, Y, X)`` voxel solid in metres, negative inside, centre to centre."""
    return (distance_transform_edt(~solid) - distance_transform_edt(solid)) * VOXEL


def surface_target(distance: np.ndarray, lower, pitch: float, patches: int = PATCHES,
                   standoff: float = STANDOFF) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a mixture to the voxel shell ``standoff`` off the surface."""
    z, y, x = np.nonzero(np.abs(distance - standoff) <= pitch / 2)
    shell = np.asarray(lower) + (np.column_stack([x, y, z]) + 0.5) * pitch
    shell = shell[shell[:, 2] >= SHELL_FLOOR]
    _, labels = kmeans2(shell, patches, minit="++", rng=0)
    members = [shell[labels == j] for j in range(patches) if np.sum(labels == j) > 3]
    means = np.stack([m.mean(axis=0) for m in members])
    covariances = np.stack([np.cov(m.T) + PATCH_FLOOR * np.eye(3) for m in members])
    weights = np.array([len(m) for m in members], float)
    return means, covariances, weights / weights.sum()


def _density(points, means, covariances, weights) -> np.ndarray:
    return sum(w * _normal(points, m, c) for w, m, c in zip(weights, means, covariances))


def lattice_target(means, covariances, weights, body) -> tuple[np.ndarray, np.ndarray]:
    """The certificate's discrete target: the mixture on the free lattice above the ground."""
    bounds = (LIMITS[0], LIMITS[1], (0.0, LIMITS[2][1]))
    axes = [np.linspace(lo, hi, int(round((hi - lo) / GRID_PITCH)) + 1) for lo, hi in bounds]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    density = _density(points, means, covariances, weights)
    keep = ~body[_cells(points)] & (density > 0.0)
    return points[keep], density[keep] / density[keep].sum()


@lru_cache(maxsize=1)
def world(cache: str, safe_distance: float) -> dict:
    """Return the scene arrays and target used by bunny flights and scoring."""
    vertices, faces = load_mesh(Path(cache))
    samples = surface_samples(vertices, faces, SURFACE_SAMPLES)
    body, distance, clearance, blocked = scene(samples, safe_distance)
    means, covariances, weights = surface_target(distance, LOWER, VOXEL)
    support, target = lattice_target(means, covariances, weights, body)
    z, y, x = np.nonzero(~blocked)
    free = LOWER + (np.column_stack([x, y, z]) + 0.5) * VOXEL
    curve = samples[::CURVE_STRIDE]
    curve = curve[curve[:, 2] >= safe_distance]  # the band below lies in the ground's keep-out
    reachable = cKDTree(free).query(curve, distance_upper_bound=SEEN)[0] < SEEN
    return {"vertices": vertices, "faces": faces, "samples": samples, "body": body,
            "clearance": clearance, "blocked": blocked, "means": means,
            "covariances": covariances, "weights": weights, "support": support,
            "target": target, "reachable": curve[reachable],
            "reachable_share": float(reachable.mean())}


# ---------------------------------------------------------------------------- scoring


def coverage_times(path: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index of the first path sample within ``SEEN`` of each point; ``inf`` if never."""
    first = np.full(len(points), np.inf)
    for index, hits in enumerate(cKDTree(points).query_ball_point(path, SEEN)):
        hits = np.asarray(hits, int)
        first[hits[np.isinf(first[hits])]] = index
    return first

def score(path: np.ndarray, cache: str, safe_distance: float, bandwidth: float,
          time_step: float) -> dict:
    """Certificate columns, contact with the scan or the ground, and coverage times."""
    scene_ = world(cache, safe_distance)
    near = cKDTree(scene_["samples"]).query(path)[0]
    contact = scene_["body"][_cells(path)] | (near < ROBOT_RADIUS) | (path[:, 2] < ROBOT_RADIUS)
    return {**certificate_columns(path, scene_["support"], scene_["target"], bandwidth),
            "collided": int(contact.any()), "contact_fraction": float(contact.mean()),
            **coverage_columns(path[::PATH_STRIDE], scene_["reachable"], time_step)}


def coverage_columns(strided: np.ndarray, reachable: np.ndarray, time_step: float) -> dict:
    """``surface_final`` and ``t<level>`` of a path already strided by ``PATH_STRIDE``."""
    first = np.sort(coverage_times(strided, reachable))
    times = {f"t{level}": first[int(np.ceil(level / 100 * len(first))) - 1]
             * PATH_STRIDE * time_step for level in COVER_LEVELS}
    return {"surface_final": float(np.isfinite(first).mean()),
            **{key: (float(value) if np.isfinite(value) else "") for key, value in times.items()}}


# -------------------------------------------------------------------------- baselines


def _solver_centres() -> list[np.ndarray]:
    """Cell centres of the baseline solver grid per axis, ``x, y, z``."""
    return [low + (np.arange(int(round(span / SOLVER_PITCH))) + 0.5) * SOLVER_PITCH
            for low, span in zip(LOWER, SPAN)]


def fourier_target(target: np.ndarray, centres, order: int):
    """Wavenumbers ``(K, 3)``, weights ``(K,)`` and target coefficients ``(K,)`` in 3D."""
    k, lam = fourier_wavenumbers(order, 3)
    tables = [np.cos(np.pi * np.arange(order + 1)[:, None] * (c - low) / span)
              for c, low, span in zip(centres, LOWER, SPAN)]
    full = np.einsum("zyx,ix,jy,kz->ijk", target, *tables)
    index = k.astype(int)
    return k, lam, full[index[:, 0], index[:, 1], index[:, 2]]

def no_flux_gradient(field: np.ndarray, blocked: np.ndarray, cell: tuple, pitch: float
                     ) -> np.ndarray:
    """Central difference of a ``(Z, Y, X)`` field at ``cell``, returned as ``(x, y, z)``."""
    gradient = np.empty(3)
    for axis in range(3):
        values = []
        for step in (-1, 1):
            neighbour = list(cell)
            neighbour[axis] += step
            inside = 0 <= neighbour[axis] < field.shape[axis]
            values.append(field[tuple(neighbour)] if inside and not blocked[tuple(neighbour)]
                          else field[cell])
        gradient[axis] = (values[1] - values[0]) / (2.0 * pitch)
    return gradient[::-1]


def avoidance(position: np.ndarray, clearance: np.ndarray, surfaces, reach: float,
              gain: float) -> np.ndarray:
    """The shared push of `baselines._avoidance`, off a non-convex body."""
    signed = float(clearance[_cells(position[None])][0])
    depth = reach - signed
    if depth <= 0.0:
        return np.zeros(3)
    solid_side, free_side = surfaces
    offset = (position - solid_side.data[solid_side.query(position)[1]] if signed > 0.0
              else free_side.data[free_side.query(position)[1]] - position)
    return gain * depth * offset / max(np.linalg.norm(offset), 1e-9)


def surface_trees(clearance: np.ndarray) -> tuple[cKDTree, cKDTree]:
    """The voxel layers either side of the surface: solid at ``-VOXEL``, free at ``+VOXEL``."""
    def layer(offset):
        z, y, x = np.nonzero(np.abs(clearance - offset) <= VOXEL / 2)
        return cKDTree(LOWER + (np.column_stack([x, y, z]) + 0.5) * VOXEL)

    return layer(-VOXEL), layer(VOXEL)

def _worker_init() -> None:
    """Baseline workers run on the CPU: SVES's JAX planner must not claim the GPU once each."""
    jax.config.update("jax_platforms", "cpu")


def fly_baseline(method: str, seed: int, steps: int, cache: str, safe_distance: float,
                 reach: float) -> np.ndarray:
    """
    Fly one baseline from the shared start through the shared tracker.
    
    Returns:
            Executed positions, ``(steps, 3)``.
    """
    cfg = BaselineConfig()
    scene_ = world(cache, safe_distance)
    model = load_config(PROFILE).controller.model
    step_model = (float(model.delta_t), float(model.max_accel_lin_abs),
                  float(model.max_accel_ang_abs))
    limits = np.asarray(LIMITS)
    centres = _solver_centres()
    grid = np.stack(np.meshgrid(centres[2], centres[1], centres[0], indexing="ij")[::-1], -1)
    shape = grid.shape[:3]
    blocked = scene_["blocked"][_cells(grid.reshape(-1, 3))].reshape(shape)
    target = np.where(blocked, 0.0, _density(grid.reshape(-1, 3), scene_["means"],
                                             scene_["covariances"],
                                             scene_["weights"]).reshape(shape))
    target /= target.sum()
    # FMEC's floor is a per-cell probability, selected on the planar grid; kept at the same
    # multiple of the uniform density here, or on 10x the cells it swamps the thin shell and
    # the target looks flat everywhere.
    fmec_floor = cfg.fmec_floor * PLANAR_CELLS / target.size
    clearance = scene_["clearance"]
    surfaces = surface_trees(clearance)

    k, lam, phi = fourier_target(target, centres, cfg.fourier_order)
    scale = np.pi * k / SPAN

    def basis(position):
        argument = scale * (position - LOWER)
        return np.prod(np.cos(argument), axis=1)

    def basis_gradient(position):
        argument = scale * (position - LOWER)
        cos, sin = np.cos(argument), np.sin(argument)
        return np.stack([-scale[:, j] * sin[j] * np.prod(np.delete(cos, j, 0), 0)
                         for j in range(3)], axis=1)

    walls = []
    for axis in range(3):
        for shift in (1, -1):
            wall = np.roll(blocked, shift, axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            wall[tuple(edge)] = True
            walls.append((axis, shift, wall))
    damping = SOLVER_PITCH ** 2 / cfg.hedac_alpha

    if method == "sves":
        scale_j, lower_j = jnp.asarray(scale, jnp.float32), jnp.asarray(LOWER, jnp.float32)
        context = SimpleNamespace(lambda_k_jax=jnp.asarray(lam, jnp.float32),
                                  phi_k_jax=jnp.asarray(phi, jnp.float32))
        planner, rollout = _sves_planner(
            context, model, cfg, dims=3,
            basis=lambda points: jnp.prod(jnp.cos(scale_j * (points[..., None, :] - lower_j)),
                                          axis=-1))
        controls = cfg.sves_init * float(model.max_accel_lin_abs) * jax.random.normal(
            jax.random.PRNGKey(seed), (cfg.sves_particles, cfg.sves_horizon, 4), jnp.float32)
        best = 0

    state = start_state(START, seed, LIMITS)
    counts = np.zeros(shape)
    coefficients = np.zeros(len(k))
    potential = np.zeros(shape)
    path = np.empty((steps, 3))
    for step in range(steps):
        position = state[:3]
        cell = tuple(np.clip(((position - LOWER) / SOLVER_PITCH).astype(int), 0,
                             np.array(shape[::-1]) - 1)[::-1])
        counts[cell] += 1.0
        coefficients += basis(position)
        if method == "hedac":
            coverage = gaussian_filter(counts, cfg.hedac_sensor / SOLVER_PITCH, mode="nearest")
            source = np.where(blocked, 0.0, target - coverage / coverage.sum())
            for _ in range(cfg.hedac_iterations):
                stencil = sum(np.where(wall, potential, np.roll(potential, shift, axis))
                              for axis, shift, wall in walls)
                potential = np.where(blocked, 0.0, (stencil + damping * source) / (6.0 + damping))
            desired = cfg.hedac_gradient_gain * no_flux_gradient(potential, blocked, cell,
                                                                 SOLVER_PITCH)
        elif method == "fmec":
            smoothed = gaussian_filter(counts, cfg.fmec_bandwidth / SOLVER_PITCH, mode="nearest")
            ratio = (np.log(target + fmec_floor)
                     - np.log(smoothed / smoothed.sum() + fmec_floor))
            desired = cfg.fmec_gain * no_flux_gradient(ratio, blocked, cell, SOLVER_PITCH)
        elif method == "smc":
            desired = -(lam * (coefficients / (step + 1) - phi)) @ basis_gradient(position)
        elif method == "sves":
            desired, controls, best = _sves_step(planner, rollout, controls, best, state[None],
                                                 coefficients, step, cfg)
            desired = desired[0]
        else:
            raise ValueError(f"unknown method {method!r}")
        push = avoidance(position, clearance, surfaces, reach, cfg.avoid_gain)
        desired = _unit_field(desired[None], cfg.desired_speed)[0] + _wall_bias(position, limits)
        desired = _limit_speed((desired + push)[None], cfg.desired_speed)[0]
        state = _track(state, desired, step_model, cfg.tracker_gain, limits)
        path[step] = state[:3]
    return path


def _job(method: str, seed: int, path, steps: int, cache: str, safe_distance: float,
         reach: float, bandwidth: float, time_step: float):
    """One worker job: fly a baseline (ours arrives already flown), then score it."""
    if path is None:
        path = fly_baseline(method, seed, steps, cache, safe_distance, reach)
    return path, score(path, cache, safe_distance, bandwidth, time_step)


# ----------------------------------------------------------------------------- driver


def run(output: Path, methods: list[str], seeds: list[int], steps: int, workers: int,
        overwrite: bool, avoid_reach: float | None = None) -> None:
    """Fly every ``(method, seed)`` not yet recorded and append one scored row each."""
    params = load_config(PROFILE).controller
    safe = float(params.workspace.safe_distance)
    reach = safe if avoid_reach is None else avoid_reach
    cache = str(output.parent)
    scene_ = world(cache, safe)
    cfg = BaselineConfig()
    lifted = lift(params, scene_["means"], scene_["covariances"], scene_["weights"], LIMITS)
    lifted = replace(lifted, workspace=replace(
        lifted.workspace, grid=jnp.asarray(scene_["blocked"], jnp.float32),
        grid_origin=jnp.asarray(LOWER, jnp.float32), grid_resolution=VOXEL))
    gaps = np.asarray(responsibility_gaps(lifted.gmm))
    record = {"profile": params, "steps": steps, "seeds": seeds, "methods": methods,
              "mesh": MESH_SHA256, "baselines": cfg.as_dict(), "solver_pitch": SOLVER_PITCH,
              "avoid_reach": reach,
              "target": [scene_["means"], scene_["covariances"], scene_["weights"]],
              "execution": execution_record(DRIVER, jax.default_backend())}
    bundle = ensure_bundle(output, record, overwrite)
    done = {(r["method"], r["seed"]) for r in verified_rows(output, ("method", "seed"))}
    store = output.with_name(output.stem + "_paths")
    store.mkdir(parents=True, exist_ok=True)
    print(f"bunny: body {int(scene_['body'].sum())} voxels, {len(gaps)} patches, gaps "
          f"{gaps.min():.2f}-{gaps.max():.2f} nats, reachable surface "
          f"{scene_['reachable_share']:.3f}", flush=True)
    common = {"steps": steps, "cache": cache, "safe_distance": safe, "reach": reach,
              "bandwidth": float(params.field.fine_bandwidth),
              "time_step": float(params.model.delta_t)}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_worker_init) as pool:
        futures = {pool.submit(_job, m, s, None, **common): (m, s)
                   for m in methods if m != "ours"
                   for s in seeds if (m, str(s)) not in done}
        todo = [s for s in seeds if "ours" in methods and ("ours", str(s)) not in done]
        if todo:
            print(f"bunny ours seeds={todo}", flush=True)
            paths = fly_ours(lifted, [start_state(START, s, LIMITS) for s in todo], todo, steps)
            futures.update({pool.submit(_job, "ours", s, p, **common): ("ours", s)
                            for s, p in zip(todo, paths)})
        for future in as_completed(futures):
            method, seed = futures[future]
            path, row = future.result()
            np.savez_compressed(store / f"{method}_s{seed}.npz", path=path[::PATH_STRIDE])
            append_csv(output, {"method": method, "seed": seed, **row, "bundle_hash": bundle},
                       FIELDS)
            print(f"bunny {method} seed={seed} mmd={row['mmd_final']:.4g} "
                  f"collided={row['collided']} surface={row['surface_final']:.3f} "
                  f"t90={row['t90']}", flush=True)


def rescore(output: Path) -> None:
    """Rewrite every row's coverage columns from its stored path; nothing is flown."""
    params = load_config(PROFILE).controller
    scene_ = world(str(output.parent), float(params.workspace.safe_distance))
    store = output.with_name(output.stem + "_paths")
    rows = verified_rows(output, ("method", "seed"))
    for row in rows:
        path = np.load(store / f"{row['method']}_s{row['seed']}.npz")["path"]
        row.update(coverage_columns(path, scene_["reachable"], float(params.model.delta_t)))
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rescored {len(rows)} rows of {output}")

def figure(paths: Path, method: str, seed: int, output: Path, azimuths,
           elevation: float, until: float | None = None) -> None:
    """Render one stored run, its first ``until`` seconds; a replay, so no GPU."""
    from ergodic_control_mppi.plotting.deployment import wrap_pdf
    from ergodic_control_mppi.plotting.dimension import bunny_snapshot

    params = load_config(PROFILE).controller
    scene_ = world(str(paths.parent), float(params.workspace.safe_distance))
    time_step = PATH_STRIDE * float(params.model.delta_t)
    path = np.load(paths / f"{method}_s{seed}.npz")["path"]
    path = path if until is None else path[: int(round(until / time_step)) + 1]
    written = bunny_snapshot(path, scene_["vertices"], scene_["faces"], scene_["body"], LOWER,
                             VOXEL, output, azimuths=azimuths, elevation=elevation)
    print(f"wrote {wrap_pdf(written)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    flight = commands.add_parser("run")
    flight.add_argument("--output", type=Path, default=Path("results/bunny/bunny.csv"))
    flight.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    flight.add_argument("--seeds", type=int, default=6)
    flight.add_argument("--steps", type=int, default=STEPS)
    flight.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    flight.add_argument("--overwrite", action="store_true")
    flight.add_argument("--avoid-reach", type=float,
                        help="metres from the surface where the baselines' shared push "
                             "starts; default our keep-out (the profile's safe distance)")
    again = commands.add_parser("rescore")
    again.add_argument("outputs", type=Path, nargs="+")
    render = commands.add_parser("figure")
    render.add_argument("--paths", type=Path, default=Path("results/bunny/bunny_paths"))
    render.add_argument("--method", choices=METHODS, default="ours")
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--azimuths", type=float, nargs="+", default=[-60.0, 120.0])
    render.add_argument("--elevation", type=float, default=20.0)
    render.add_argument("--until", type=float, help="seconds of the run to draw; all by default")
    render.add_argument("--output", type=Path, default=Path("results/report/fig_bunny.png"))
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.methods, list(range(args.seeds)), args.steps, args.workers,
            args.overwrite, args.avoid_reach)
    elif args.command == "rescore":
        for output in args.outputs:
            rescore(output)
    else:
        figure(args.paths, args.method, args.seed, args.output, args.azimuths, args.elevation,
               args.until)


if __name__ == "__main__":
    main()

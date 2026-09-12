"""Volumetric clutter from Perlin noise: blobs at every height, the three modes among them.

uv run python -m ergodic_control_mppi.experiments.perlin run [--seeds 6]
uv run python -m ergodic_control_mppi.experiments.perlin figure [--seed 0]"""

import argparse
import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import label

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.baselines import certificate_columns
from ergodic_control_mppi.experiments.bunny import VOXEL, _cells, _worker_init, signed_distance
from ergodic_control_mppi.experiments.common import (
    append_csv,
    ensure_bundle,
    execution_record,
    verified_rows,
)
from ergodic_control_mppi.experiments.dimension import (
    GRID_PITCH,
    LIMITS_3D,
    PATH_STRIDE,
    PROFILE,
    ROBOT_RADIUS,
    START_3D,
    _normal,
    fly_ours,
    lift,
    start_state,
    target_3d,
)

DRIVER = "ergodic_control_mppi/experiments/perlin.py"
LOWER = np.array([low for low, _ in LIMITS_3D])
SHAPE = tuple(int(round((high - low) / VOXEL)) for low, high in reversed(LIMITS_3D))  # (Z, Y, X)
FEATURE = 4.0             # metres between noise lattice points at the first octave
FILL = 0.10               # solid share of the slab: mockamap's SITL setting
OCTAVES = 2               # mockamap's `fractal`; each octave doubles the frequency
ATTENUATION = 0.5         # octave `it` weighs ATTENUATION / it, as in mockamap
MAP_SEED = 0
START_CLEAR = 3.0         # metres kept free around the start, as the pillar field keeps
STEPS = 20_000            # 400 s, as the pillar runs
CLOUD_PITCH = 3           # voxels per drawn cell: 0.3 m, as a map display shows one
FRAMES = (50.0, 100.0, 200.0, 400.0)  # seconds, each twice the last: early to late

FIELDS = ["seed", "mmd_final", "mmd_bound", "mmd_trivial", "mmd_weighted_gap",
          "mmd_prefix_holds", "mmd_beats_trivial", "mmd_samples", "mmd_stride", "collided",
          "contact_fraction", "altitude_mean", "altitude_std", "bundle_hash"]

# The twelve cube-edge directions: improved noise's gradient set.
EDGES = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
                  [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], float)


def perlin(points: np.ndarray, seed: int) -> np.ndarray:
    """Perlin's improved noise at ``(N, 3)`` points, zero at every integer point."""
    table = np.tile(np.random.default_rng(seed).permutation(256), 2)
    cell = np.floor(points).astype(int)
    local = points - cell
    cell &= 255
    fade = local ** 3 * (local * (local * 6.0 - 15.0) + 10.0)
    total = np.zeros(len(points))
    for corner in itertools.product((0, 1), repeat=3):
        i, j, k = (cell + corner).T
        gradient = EDGES[table[table[table[i] + j] + k] % 12]
        weight = np.prod(np.where(corner, fade, 1.0 - fade), axis=1)
        total += weight * np.sum(gradient * (local - corner), axis=1)
    return total


@lru_cache(maxsize=1)
def world() -> dict:
    """The map, the controller's voxel grid and the certificate's target, from the constants."""
    params = load_config(PROFILE).controller
    axes = [LOWER[k] + (np.arange(SHAPE[2 - k]) + 0.5) * VOXEL for k in (2, 1, 0)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    points = np.stack([x, y, z], -1)
    unit = points.reshape(-1, 3) / FEATURE
    noise = sum(ATTENUATION / it * perlin(2.0 ** (it - 1) * unit, MAP_SEED)
                for it in range(1, OCTAVES + 1)).reshape(SHAPE)
    solid = noise > np.quantile(noise, 1.0 - FILL)  # mockamap's rule
    solid &= np.linalg.norm(points - START_3D, axis=-1) > START_CLEAR
    clearance = signed_distance(solid)
    blocked = clearance <= float(params.workspace.safe_distance) + 1e-6 * VOXEL  # as `bunny.scene`
    start = _cells(np.asarray([START_3D]), LOWER, SHAPE)
    flyable, _ = label(~blocked)
    free, _ = label(~solid)
    # The certificate's support: the target on a lattice of the free space the start reaches.
    means, covariances, weights = target_3d(params)
    lattice = np.stack(np.meshgrid(*[np.linspace(lo, hi, int(round((hi - lo) / GRID_PITCH)) + 1)
                                     for lo, hi in LIMITS_3D], indexing="ij"), -1).reshape(-1, 3)
    density = sum(w * _normal(lattice, m, c) for w, m, c in zip(weights, means, covariances))
    keep = (free == free[start][0])[_cells(lattice, LOWER, SHAPE)] & (density > 0.0)
    return {"solid": solid, "clearance": clearance, "blocked": blocked,
            "support": lattice[keep], "target": density[keep] / density[keep].sum(),
            "solid_share": float(solid.mean()),
            "flyable_share": float((flyable == flyable[start][0]).sum() / (~blocked).sum()),
            "support_mass": float(density[keep].sum() / density.sum())}


def score(path: np.ndarray, bandwidth: float) -> dict:
    """Certificate columns, contact within the robot radius of the noise, and altitude use."""
    scene = world()
    contact = scene["clearance"][_cells(path, LOWER, SHAPE)] < ROBOT_RADIUS
    return {**certificate_columns(path, scene["support"], scene["target"], bandwidth),
            "collided": int(contact.any()), "contact_fraction": float(contact.mean()),
            "altitude_mean": float(path[:, 2].mean()), "altitude_std": float(path[:, 2].std())}


def cloud(solid: np.ndarray, pitch: int = CLOUD_PITCH) -> np.ndarray:
    """Centres ``(M, 3)`` of the cells, ``pitch`` voxels a side, that are mostly solid."""
    z, y, x = (n // pitch for n in solid.shape)
    occupied = solid[:z * pitch, :y * pitch, :x * pitch].reshape(
        z, pitch, y, pitch, x, pitch).mean((1, 3, 5)) >= 0.5
    z, y, x = np.nonzero(occupied)
    return LOWER + (np.column_stack([x, y, z]) + 0.5) * pitch * VOXEL


def run(output: Path, seeds: list[int], steps: int, workers: int, overwrite: bool) -> None:
    """Fly every seed not yet recorded in one batch, then append one scored row each."""
    params = load_config(PROFILE).controller
    scene = world()
    means, covariances, weights = target_3d(params)
    lifted = lift(params, means, covariances, weights, LIMITS_3D)
    lifted = replace(lifted, workspace=replace(
        lifted.workspace, grid=jnp.asarray(scene["blocked"], jnp.float32),
        grid_origin=jnp.asarray(LOWER, jnp.float32), grid_resolution=VOXEL))
    record = {"profile": params, "steps": steps, "seeds": seeds,
              "map": {"seed": MAP_SEED, "feature": FEATURE, "fill": FILL, "octaves": OCTAVES,
                      "attenuation": ATTENUATION, "start_clear": START_CLEAR},
              "target": [means, covariances, weights],
              "execution": execution_record(DRIVER, jax.default_backend())}
    bundle = ensure_bundle(output, record, overwrite)
    print(f"perlin: solid {scene['solid_share']:.3f} of the slab, flyable component "
          f"{scene['flyable_share']:.3f} of the free space, target mass on the support "
          f"{scene['support_mass']:.3f}", flush=True)
    done = {r["seed"] for r in verified_rows(output, ("seed",))}
    todo = [s for s in seeds if str(s) not in done]
    if not todo:
        return
    store = output.with_name(output.stem + "_paths")
    store.mkdir(parents=True, exist_ok=True)
    print(f"perlin seeds={todo}", flush=True)
    paths = fly_ours(lifted, [start_state(START_3D, s, LIMITS_3D) for s in todo], todo, steps)
    for seed, path in zip(todo, paths):
        np.savez_compressed(store / f"s{seed}.npz", path=path[::PATH_STRIDE])
    scorer = partial(score, bandwidth=float(params.field.fine_bandwidth))
    with ProcessPoolExecutor(min(workers, len(todo)), initializer=_worker_init,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for seed, row in zip(todo, pool.map(scorer, paths)):
            append_csv(output, {"seed": seed, **row, "bundle_hash": bundle}, FIELDS)
            print(f"perlin seed={seed} mmd={row['mmd_final']:.4g} "
                  f"collided={row['collided']}", flush=True)


def figure(paths: Path, seed: int, output: Path, times, azimuth: float,
           elevation: float) -> None:
    """Render frames of one stored run as ``<output>_t<seconds>`` images."""
    from ergodic_control_mppi.plotting.deployment import wrap_pdf
    from ergodic_control_mppi.plotting.dimension import perlin_snapshot

    params = load_config(PROFILE).controller
    solid = world()["solid"]
    means, covariances, weights = target_3d(params)
    path = np.load(paths / f"s{seed}.npz")["path"]
    step = PATH_STRIDE * float(params.model.delta_t)
    points = cloud(solid)
    for time in times:
        frame = output.with_name(f"{output.stem}_t{time:g}{output.suffix}")
        written = perlin_snapshot(path[: int(round(time / step)) + 1], points, means,
                                  covariances, weights, LIMITS_3D, solid, LOWER, VOXEL, frame,
                                  azimuth=azimuth, elevation=elevation)
        print(f"wrote {wrap_pdf(written)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    flight = commands.add_parser("run")
    flight.add_argument("--output", type=Path, default=Path("results/perlin/perlin.csv"))
    flight.add_argument("--seeds", type=int, default=6)
    flight.add_argument("--steps", type=int, default=STEPS)
    flight.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    flight.add_argument("--overwrite", action="store_true")
    render = commands.add_parser("figure")
    render.add_argument("--paths", type=Path, default=Path("results/perlin/perlin_paths"))
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--times", type=float, nargs="+", default=list(FRAMES),
                        help="seconds into the run, one frame each")
    render.add_argument("--azimuth", type=float, default=-75.0)
    render.add_argument("--elevation", type=float, default=30.0)
    render.add_argument("--output", type=Path, default=Path("results/report/fig_perlin.png"))
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, list(range(args.seeds)), args.steps, args.workers, args.overwrite)
    else:
        figure(args.paths, args.seed, args.output, args.times, args.azimuth, args.elevation)


if __name__ == "__main__":
    main()

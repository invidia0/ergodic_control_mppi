"""Beyond the plane: a volumetric pillar field and a workspace-dimension sweep.

uv run python -m ergodic_control_mppi.experiments.dimension clutter3d [--output PATH]
uv run python -m ergodic_control_mppi.experiments.dimension scaling [--output PATH]"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial, reduce
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.baselines import (
    CERTIFICATE_STRIDE,
    BaselineConfig,
    _unit_field,
    certificate_columns,
)
from ergodic_control_mppi.experiments.common import (
    append_csv,
    ensure_bundle,
    execution_record,
    verified_rows,
)
from ergodic_control_mppi.experiments.literature_methods import _limit_speed
from ergodic_control_mppi.experiments.timing import _time
from ergodic_control_mppi.metrics.ergodicity import fourier_wavenumbers
from ergodic_control_mppi.mppi.single import initialize_single, run_single, single_step
from ergodic_control_mppi.parameters import ControllerParams, GMMParams
from ergodic_control_mppi.simulation import controller_key

PROFILE = "configs/uav_profile_T150.yaml"
DRIVER = "ergodic_control_mppi/experiments/dimension.py"
STEPS = 20000
PATH_STRIDE = 5           # stored paths: 0.1 s, so a transit turn still draws as a curve
ROBOT_RADIUS = 0.30       # contact is scored against the raw geometry inflated by this

# clutter3d: the profile's three modes, each at its own altitude, in a 7 m slab.
LIMITS_3D = ((-20.0, 20.0), (-10.0, 10.0), (0.0, 7.0))
ALTITUDES = (2.5, 4.5, 3.5)
Z_VARIANCE = 0.5
PILLAR_COUNTS = (10, 15, 20)
START_3D = (0.0, 0.0, 3.5)
GRID_PITCH = 0.5          # certificate support spacing, metres

# scaling: four isotropic modes in [-10, 10]^d.
DIMS = (2, 3, 4, 6)
HALF_SIDE = 10.0
MODE_COUNT = 4
MODE_VARIANCE = 2.0
HEDAC_CELLS = 32          # per axis: 0.625 m pitch, under the kernel's 0.69 m peak separation
HEDAC_MAX_CELLS = 2 ** 22  # a larger mesh is projected from the measured per-cell cost

FIELDS_3D = ["arm", "pillars", "seed", "mmd_final", "mmd_bound", "mmd_trivial",
             "mmd_weighted_gap", "mmd_prefix_holds", "mmd_beats_trivial", "mmd_samples",
             "mmd_stride", "collided", "contact_fraction", "altitude_mean", "altitude_std",
             "bundle_hash"]
FIELDS_SCALING = ["method", "dim", "seed", "mmd_final", "exceeds_budget", "bundle_hash"]
FIELDS_TIMING = ["method", "dim", "size", "ms_median", "ms_iqr", "projected", "bundle_hash"]


# ------------------------------------------------------------------------- controller


def lift(params: ControllerParams, means, covariances, weights, limits,
         obstacles=None) -> ControllerParams:
    """
    Lift the deployed controller to a ``d``-dimensional workspace.
    
    Args:
            params: Deployed planar controller parameters.
            means: Mixture means, ``(J, d)``.
            covariances: Mixture covariances, ``(J, d, d)``.
            weights: Mixture weights, ``(J,)``.
            limits: Workspace bounds per axis, ``(d, 2)``.
            obstacles: Pillars ``(n, 3)`` or ``(n, 4)``; none when omitted.
    Returns:
            Parameters whose position dimension is ``d``.
    """
    means = np.asarray(means, np.float64)
    covariances = np.asarray(covariances, np.float64)
    limits = np.asarray(limits, np.float32)
    d = means.shape[1]
    gmm = GMMParams(
        means=jnp.asarray(means, jnp.float32),
        covariance=jnp.asarray(covariances, jnp.float32),
        covariance_inverse=jnp.asarray(np.linalg.inv(covariances), jnp.float32),
        log_weights=jnp.log(jnp.asarray(weights, jnp.float32)),
        log_normalizers=jnp.asarray(
            -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(covariances)[1]), jnp.float32),
    )
    variance = np.diag(np.asarray(params.mppi.covariance))
    noise = np.diag(np.r_[np.full(d, variance[0]), variance[-1]]).astype(np.float32)
    obstacles = np.zeros((0, 3)) if obstacles is None else obstacles
    return replace(
        params,
        gmm=gmm,
        mppi=replace(params.mppi, covariance=jnp.asarray(noise),
                     covariance_inverse=jnp.asarray(np.linalg.inv(noise))),
        workspace=replace(
            params.workspace, x_limits=jnp.asarray(limits[0]), y_limits=jnp.asarray(limits[1]),
            extra_limits=jnp.asarray(limits[2:]), obstacles=jnp.asarray(obstacles, jnp.float32)),
    )


def fly_ours(params: ControllerParams, starts, seeds, steps: int) -> np.ndarray:
    """
    One fused batch of closed loops, a lane per ``(start, seed)``.
    
    Returns:
            Executed positions, ``(lanes, steps, d)``.
    """
    d = params.gmm.means.shape[-1]
    controls = jnp.zeros((params.mppi.horizon, d + 1), jnp.float32)
    lanes = jax.jit(jax.vmap(
        lambda start, key: run_single(params, start, controls, key, steps).path))
    keys = jnp.stack([controller_key(int(seed)) for seed in seeds])
    return np.asarray(lanes(jnp.asarray(np.stack(starts), jnp.float32), keys))[..., :d]


def start_state(position, seed: int, limits) -> np.ndarray:
    """Return a jittered start pose at rest."""
    limits = np.asarray(limits, np.float64)
    position = np.asarray(position, np.float64).copy()
    rng = np.random.default_rng(seed)
    position[:2] = np.clip(position[:2] + rng.uniform(-1.5, 1.5, 2),
                           limits[:2, 0] + 1.0, limits[:2, 1] - 1.0)
    return np.r_[position, np.zeros(position.size + 2)]


# -------------------------------------------------------------------------- clutter3d


def target_3d(params: ControllerParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The profile's planar mixture with each mode lifted to its own altitude."""
    planar = np.asarray(params.gmm.covariance, np.float64)
    means = np.c_[np.asarray(params.gmm.means, np.float64), ALTITUDES]
    covariances = np.zeros((len(means), 3, 3))
    covariances[:, :2, :2] = planar
    covariances[:, 2, 2] = Z_VARIANCE
    weights = np.exp(np.asarray(params.gmm.log_weights, np.float64))
    return means, covariances, weights / weights.sum()

def pillar_field(count: int, seed: int) -> np.ndarray:
    """Pillars ``(count, 4)`` as ``(x, y, radius, top)``."""
    rng = np.random.default_rng(seed)
    pillars: list[tuple[float, float, float, float]] = []
    while len(pillars) < count:
        x, y, radius = rng.uniform(-16.0, 16.0), rng.uniform(-8.0, 8.0), rng.uniform(0.3, 0.6)
        clear = all(np.hypot(x - px, y - py) >= radius + pr + 1.2 for px, py, pr, _ in pillars)
        if clear and np.hypot(x - START_3D[0], y - START_3D[1]) >= 3.0 + radius:
            top = LIMITS_3D[2][1] + 1.0 if len(pillars) % 2 == 0 else rng.uniform(1.5, 3.0)
            pillars.append((x, y, radius, top))
    return np.asarray(pillars)


def _inside(points: np.ndarray, pillars: np.ndarray, margin: float) -> np.ndarray:
    """Whether each ``(N, 3)`` point is inside a pillar inflated by ``margin``."""
    horizontal = np.linalg.norm(points[:, None, :2] - pillars[None, :, :2], axis=-1)
    return np.any((horizontal <= pillars[:, 2] + margin)
              & (points[:, None, 2] <= pillars[:, 3] + margin), axis=1)

def _normal(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    delta = points - mean
    quadratic = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(covariance), delta)
    return np.exp(-0.5 * quadratic) / np.sqrt(np.linalg.det(2 * np.pi * covariance))

def support_3d(means, covariances, weights, pillars) -> tuple[np.ndarray, np.ndarray]:
    """The certificate's discrete target: the mixture on a lattice, pillar interiors removed."""
    axes = [np.linspace(lo, hi, int(round((hi - lo) / GRID_PITCH)) + 1) for lo, hi in LIMITS_3D]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    density = sum(w * _normal(points, m, c) for w, m, c in zip(weights, means, covariances))
    keep = ~_inside(points, pillars, 0.0) & (density > 0.0)
    return points[keep], density[keep] / density[keep].sum()


def score_3d(path: np.ndarray, support, weights, pillars, bandwidth: float) -> dict:
    """Certificate columns plus contact and altitude use of one executed 3D path."""
    contact = _inside(path, pillars, ROBOT_RADIUS)
    return {**certificate_columns(path, support, weights, bandwidth),
        "collided": int(contact.any()), "contact_fraction": float(contact.mean()),
        "altitude_mean": float(path[:, 2].mean()), "altitude_std": float(path[:, 2].std())}

def run_clutter3d(output: Path, seeds: list[int], steps: int, workers: int,
                  overwrite: bool) -> None:
    """Fly both arms on every pillar field and append one scored row per cell."""
    params = load_config(PROFILE).controller
    means, covariances, weights = target_3d(params)
    altitude = float(weights @ means[:, 2])
    maps = {count: pillar_field(count, seed=count) for count in PILLAR_COUNTS}
    record = {"profile": params, "steps": steps, "seeds": seeds, "maps": maps,
              "target": [means, covariances, weights], "altitude": altitude,
              "execution": execution_record(DRIVER, jax.default_backend())}
    bundle = ensure_bundle(output, record, overwrite)
    done = {(r["arm"], r["pillars"], r["seed"])
            for r in verified_rows(output, ("arm", "pillars", "seed"))}
    store = output.with_name(output.stem + "_paths")
    store.mkdir(parents=True, exist_ok=True)
    bandwidth = float(params.field.fine_bandwidth)
    blocking = float(params.workspace.safe_distance)
    arms = {
        "volumetric": lambda pillars: lift(params, means, covariances, weights, LIMITS_3D, pillars),
        "planar": lambda pillars: lift(params, means[:, :2], covariances[:, :2, :2], weights,
                                       LIMITS_3D[:2],
                                       pillars[pillars[:, 3] + blocking >= altitude, :3]),
    }
    for count, pillars in maps.items():
        support, target = support_3d(means, covariances, weights, pillars)
        for arm, build in arms.items():
            todo = [s for s in seeds if (arm, str(count), str(s)) not in done]
            if not todo:
                continue
            arm_params = build(pillars)
            d = arm_params.gmm.means.shape[-1]
            starts = [start_state(START_3D[:d], s, LIMITS_3D[:d]) for s in todo]
            print(f"clutter3d {arm} pillars={count} seeds={todo}", flush=True)
            paths = fly_ours(arm_params, starts, todo, steps)
            if d == 2:
                paths = np.concatenate((paths, np.full(paths.shape[:2] + (1,), altitude)), -1)
            for seed, path in zip(todo, paths):
                np.savez_compressed(store / f"{arm}_{count}_s{seed}.npz",
                                    path=path[::PATH_STRIDE], pillars=pillars)
            scorer = partial(score_3d, support=support, weights=target, pillars=pillars,
                             bandwidth=bandwidth)
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for seed, score in zip(todo, pool.map(scorer, paths)):
                    append_csv(output, {"arm": arm, "pillars": count, "seed": seed, **score,
                                        "bundle_hash": bundle}, FIELDS_3D)


# ---------------------------------------------------------------------------- scaling


def scaling_target(d: int) -> np.ndarray:
    """Return ``MODE_COUNT`` separated means in ``[-5, 5]^d``."""
    rng = np.random.default_rng(d)
    means: list[np.ndarray] = []
    while len(means) < MODE_COUNT:
        candidate = rng.uniform(-5.0, 5.0, d)
        if all(np.linalg.norm(candidate - other) >= 5.0 for other in means):
            means.append(candidate)
    return np.asarray(means)


def _box(d: int) -> np.ndarray:
    return np.tile([-HALF_SIDE, HALF_SIDE], (d, 1))


def continuous_mmd(path: np.ndarray, means: np.ndarray, variance: float,
                   bandwidth: float) -> float:
    """Squared MMD between a path and an equal-weight isotropic mixture."""
    d = means.shape[1]
    weight = 1.0 / len(means)

    def squared(a, b):
        return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)

    near, far = bandwidth + 2.0 * variance, bandwidth + 4.0 * variance
    embedding = weight * (bandwidth / near) ** (d / 2) * np.exp(-squared(path, means) / near).sum(1)
    norm = weight ** 2 * (bandwidth / far) ** (d / 2) * np.exp(-squared(means, means) / far).sum()
    return float(np.exp(-squared(path, path) / bandwidth).mean() - 2.0 * embedding.mean() + norm)


def coverage_law(method: str, means: np.ndarray, cfg: BaselineConfig):
    """Return ``law(position, step) -> velocity`` for SMC or HEDAC."""
    d = means.shape[1]
    lower, span = -HALF_SIDE, 2.0 * HALF_SIDE
    if method == "smc":
        k, lam = fourier_wavenumbers(cfg.fourier_order, d)
        scale = np.pi * k / span
        attenuation = np.exp(-0.5 * scale ** 2 * MODE_VARIANCE)
        phi = np.mean([np.prod(np.cos(scale * (m - lower)) * attenuation, axis=1)
                       for m in means], axis=0)
        coefficients = np.zeros(len(k))

        def smc(position, step):
            argument = scale * (position - lower)
            cos, sin = np.cos(argument), np.sin(argument)
            coefficients[:] += np.prod(cos, axis=1)
            grad = np.stack([-scale[:, j] * sin[:, j] * np.prod(np.delete(cos, j, 1), 1)
                             for j in range(d)], axis=1)
            weight = lam * (coefficients / max(step + 1, 1.0) - phi)
            return -(weight @ grad)

        return smc

    from scipy import ndimage

    pitch = span / HEDAC_CELLS
    centres = lower + (np.arange(HEDAC_CELLS) + 0.5) * pitch
    target = sum(
        reduce(np.multiply, np.meshgrid(
            *[np.exp(-0.5 * (centres - m[i]) ** 2 / MODE_VARIANCE) for i in range(d)],
            indexing="ij", sparse=True))
        for m in means)
    target /= target.sum()
    counts = np.zeros(target.shape)
    damping = pitch ** 2 / cfg.hedac_alpha
    state = {"potential": np.zeros(target.shape)}

    def hedac(position, step):
        cell = tuple(np.clip(((position - lower) / pitch).astype(int), 0, HEDAC_CELLS - 1))
        counts[cell] += 1.0
        coverage = ndimage.gaussian_filter(counts / counts.sum(), cfg.hedac_sensor / pitch,
                                           mode="nearest")
        source = target - coverage / max(coverage.sum(), 1e-12)
        potential = state["potential"]
        for _ in range(cfg.hedac_iterations):
            stencil = damping * source
            for axis in range(d):
                for shift in (1, -1):
                    neighbour = np.roll(potential, shift, axis)
                    edge = [slice(None)] * d
                    edge[axis] = 0 if shift == 1 else -1
                    neighbour[tuple(edge)] = potential[tuple(edge)]
                    stencil = stencil + neighbour
            potential = stencil / (2 * d + damping)
        state["potential"] = potential
        grad = np.empty(d)
        for axis in range(d):
            low, high = list(cell), list(cell)
            low[axis], high[axis] = max(cell[axis] - 1, 0), min(cell[axis] + 1, HEDAC_CELLS - 1)
            grad[axis] = ((potential[tuple(high)] - potential[tuple(low)])
                          / ((high[axis] - low[axis]) * pitch))
        return cfg.hedac_gradient_gain * grad

    return hedac


def _track(state: np.ndarray, desired: np.ndarray, model: tuple[float, float, float],
           gain: float, limits: np.ndarray) -> np.ndarray:
    """`literature_methods._tracker_step_np` in ``d`` dimensions."""
    d = len(limits)
    dt, linear, angular = model
    accel = np.clip(gain * (desired - state[d:2 * d]), -linear, linear)
    yaw_accel = np.clip(-gain * state[2 * d + 1], -angular, angular)
    position = state[:d] + state[d:2 * d] * dt + 0.5 * accel * dt ** 2
    velocity = state[d:2 * d] + accel * dt
    clipped = np.clip(position, limits[:, 0], limits[:, 1])
    velocity = np.where(clipped != position, 0.0, velocity)
    yaw = state[2 * d] + state[2 * d + 1] * dt + 0.5 * yaw_accel * dt ** 2
    return np.r_[clipped, velocity, yaw, state[2 * d + 1] + yaw_accel * dt]


def _wall_bias(position: np.ndarray, limits: np.ndarray, margin: float = 0.75,
           gain: float = 1.5) -> np.ndarray:
    """`literature_methods._boundary_bias` on every axis."""
    return gain * (np.clip(limits[:, 0] + margin - position, 0.0, None)
               - np.clip(position - (limits[:, 1] - margin), 0.0, None))


def fly_baseline(method: str, means: np.ndarray, seed: int, steps: int,
                 model: tuple[float, float, float]) -> np.ndarray:
    """Fly SMC or HEDAC on the open ``d``-box through the shared tracker."""
    cfg = BaselineConfig()
    limits = _box(means.shape[1])
    law = coverage_law(method, means, cfg)
    state = start_state(np.zeros(means.shape[1]), seed, limits)
    path = np.empty((steps, means.shape[1]))
    for step in range(steps):
        position = state[:means.shape[1]]
        desired = _unit_field(law(position, step)[None], cfg.desired_speed)[0]
        desired = _limit_speed((desired + _wall_bias(position, limits))[None],
                               cfg.desired_speed)[0]
        state = _track(state, desired, model, cfg.tracker_gain, limits)
        path[step] = state[:means.shape[1]]
    return path


def _scaling_params(params: ControllerParams, d: int) -> ControllerParams:
    means = scaling_target(d)
    return lift(params, means, np.tile(MODE_VARIANCE * np.eye(d), (len(means), 1, 1)),
                np.full(len(means), 1.0 / len(means)), _box(d))


def _timing_rows(params: ControllerParams, dims: list[int], cfg: BaselineConfig) -> list[dict]:
    """Per-step wall time for our MPPI step and each baseline coverage law."""
    rows = []
    hedac_rate = None
    for d in sorted(dims):
        lifted = _scaling_params(params, d)
        carry = initialize_single(lifted, jnp.zeros(2 * d + 2, jnp.float32),
                                  jnp.zeros((lifted.mppi.horizon, d + 1), jnp.float32),
                                  controller_key(0))
        advance = jax.jit(lambda c, p=lifted: single_step(p, c)[0])
        stats = _time(lambda: advance(carry), repeats=100)
        rows.append({"method": "ours", "dim": d, "size": lifted.mppi.samples * lifted.mppi.horizon,
                     "ms_median": stats["ms_median"], "ms_iqr": stats["ms_iqr"], "projected": 0})
        position = np.full(d, 0.3)
        smc = coverage_law("smc", scaling_target(d), cfg)
        stats = _time(lambda: smc(position, 10), repeats=50, warmup=2)
        rows.append({"method": "smc", "dim": d, "size": (cfg.fourier_order + 1) ** d - 1,
                     "ms_median": stats["ms_median"], "ms_iqr": stats["ms_iqr"], "projected": 0})
        cells = HEDAC_CELLS ** d
        if cells <= HEDAC_MAX_CELLS:
            hedac = coverage_law("hedac", scaling_target(d), cfg)
            stats = _time(lambda: hedac(position, 10), repeats=20, warmup=2)
            hedac_rate = stats["ms_median"] / cells
            rows.append({"method": "hedac", "dim": d, "size": cells, "projected": 0,
                         "ms_median": stats["ms_median"], "ms_iqr": stats["ms_iqr"]})
        elif hedac_rate is not None:
            rows.append({"method": "hedac", "dim": d, "size": cells, "projected": 1,
                         "ms_median": hedac_rate * cells, "ms_iqr": ""})
        print(f"timing d={d}: " + ", ".join(f"{r['method']} {r['ms_median']:.3g} ms"
                                           for r in rows if r["dim"] == d), flush=True)
    return rows


def run_scaling(output: Path, dims: list[int], seeds: list[int], steps: int, workers: int,
                overwrite: bool) -> None:
    """Time every method per ``d``, then fly and score every ``(method, d, seed)`` cell."""
    params = load_config(PROFILE).controller
    cfg = BaselineConfig()
    bandwidth = float(params.field.fine_bandwidth)
    record = {"profile": params, "steps": steps, "seeds": seeds, "dims": dims,
              "targets": {d: scaling_target(d) for d in dims}, "baselines": cfg.as_dict(),
              "execution": execution_record(DRIVER, jax.default_backend())}
    bundle = ensure_bundle(output, record, overwrite)
    timing = output.with_name(output.stem + "_timing.csv")
    ensure_bundle(timing, record, overwrite)
    if not verified_rows(timing, ("method", "dim")):
        for row in _timing_rows(params, dims, cfg):
            append_csv(timing, {**row, "bundle_hash": bundle}, FIELDS_TIMING)
    done = {(r["method"], r["dim"], r["seed"])
            for r in verified_rows(output, ("method", "dim", "seed"))}
    store = output.with_name(output.stem + "_paths")
    store.mkdir(parents=True, exist_ok=True)

    def record_row(method, d, seed, path):
        np.savez_compressed(store / f"{method}_d{d}_s{seed}.npz", path=path[::PATH_STRIDE])
        mmd = continuous_mmd(path[::CERTIFICATE_STRIDE], scaling_target(d), MODE_VARIANCE,
                             bandwidth)
        append_csv(output, {"method": method, "dim": d, "seed": seed, "mmd_final": mmd,
                            "exceeds_budget": 0, "bundle_hash": bundle}, FIELDS_SCALING)
        print(f"scaling {method} d={d} seed={seed} mmd={mmd:.4g}", flush=True)

    jobs = []
    for d in dims:
        todo = [s for s in seeds if ("ours", str(d), str(s)) not in done]
        if todo:
            paths = fly_ours(_scaling_params(params, d),
                             [start_state(np.zeros(d), s, _box(d)) for s in todo], todo, steps)
            for seed, path in zip(todo, paths):
                record_row("ours", d, seed, path)
        for method in ("smc", "hedac"):
            for seed in (s for s in seeds if (method, str(d), str(s)) not in done):
                if method == "hedac" and HEDAC_CELLS ** d > HEDAC_MAX_CELLS:
                    append_csv(output, {"method": method, "dim": d, "seed": seed,
                                        "mmd_final": "", "exceeds_budget": 1,
                                        "bundle_hash": bundle}, FIELDS_SCALING)
                else:
                    jobs.append((method, d, seed))
    model = tuple(float(v) for v in (params.model.delta_t, params.model.max_accel_lin_abs,
                                       params.model.max_accel_ang_abs))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(fly_baseline, m, scaling_target(d), s, steps, model): (m, d, s)
                   for m, d, s in jobs}
        for future in as_completed(futures):
            record_row(*futures[future], future.result())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    studies = parser.add_subparsers(dest="study", required=True)
    for name, seeds in (("clutter3d", 12), ("scaling", 6)):
        study = studies.add_parser(name)
        study.add_argument("--output", type=Path, default=Path(f"results/dimension/{name}.csv"))
        study.add_argument("--seeds", type=int, default=seeds)
        study.add_argument("--steps", type=int, default=STEPS)
        study.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
        study.add_argument("--overwrite", action="store_true")
    studies.choices["scaling"].add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    if args.study == "clutter3d":
        run_clutter3d(args.output, seeds, args.steps, args.workers, args.overwrite)
    else:
        run_scaling(args.output, args.dims, seeds, args.steps, args.workers, args.overwrite)


if __name__ == "__main__":
    main()

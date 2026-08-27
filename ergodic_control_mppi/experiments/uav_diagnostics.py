"""Focused diagnostics for MPPI weights and the JAX-to-UAV discrepancy.

The original ESS and cost diagnostics answer questions the coverage metrics cannot. The
discrepancy modes add a falsification-first comparison of pure JAX, ideal ROS feedback,
and the SO3 simulator without changing controller parameters.

``ess`` sweeps candidate temperature caps and penalty scales and reports where the adaptive
loop actually converges. ESS and temperature settle within ~350 steps, so a short run is
sufficient, and it runs on CPU so it need not contend with a sweep on the GPU. The CPU and
GPU backends give different trajectories, but ESS is set by the cost *scale* rather than
the path, so it screens combinations faithfully.

``costs`` decomposes one planning step's per-rollout cost into its terms. Only cross-rollout
variance matters -- the softmax sees differences, so a term with a huge mean but no spread
is invisible to the weighting, and a term with modest mean but large spread dominates it.

    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics ess --run-dir ...
    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics costs --run-dir ...
    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics discrepancy-jax ...
    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics discrepancy-ros ...
    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics discrepancy-report ...
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import time

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
from ergodic_control_mppi.mppi.core import (
    _grid_cost,
    _rollouts,
    effective_sample_fraction,
    sample_epsilon,
)
from ergodic_control_mppi.mppi.replay import restore_snapshot
from ergodic_control_mppi.mppi.single import (
    SingleControllerState,
    initialize_single,
    single_step,
    stationary_step,
)
from ergodic_control_mppi.simulation import controller_key, run_simulation, select_device

ESS_FIELDS = [
    "arm", "penalty_scale", "boundary_scale", "lam_max", "steps", "seed",
    "lambda_final", "lambda_pinned", "ess_percent_median", "ess_percent_settled",
    "ess_target_percent", "samples", "device",
]
COST_FIELDS = [
    "step", "term", "mean", "std", "spread_p99_min", "share_of_total_spread", "device",
]
DISCREPANCY_FIELDS = [
    "source", "condition", "vehicle", "map_seed", "seed", "repeat", "hardware",
    "perturbation_id", "ulp_count", "impulse_px", "impulse_py", "impulse_vx",
    "impulse_vy", "steps", "preflight_steps", "tour_count", "tours_per_1000s",
    "zero_tour", "first_all_modes_s", "restricted_first_tour_s", "mode_visits",
    "mode_switches", "mode_revisits", "mode_dwell_median_s", "mode_dwell_total_s",
    "mode_transitions", "mode_cycles", "in_mode_fraction", "occupancy_mse",
    "fourier_ergodic", "divergence_1cm_s", "divergence_10cm_s", "divergence_1m_s",
    "path_sha256", "feedback_pos_p50_m", "feedback_pos_p95_m",
    "feedback_vel_p50_mps", "feedback_vel_p95_mps", "callback_gap_p50_ms",
    "callback_gap_p95_ms", "callback_gap_max_ms", "odom_age_p95_ms",
    "guard_fraction", "wall_seconds", "device", "jax_version", "xla_flags", "run_id",
]


def _grid_config(run_directory: Path, config_path: str):
    """Load the profile with the recorded run's grid folded in."""
    archive = run_directory / "figure_data.npz"
    if not archive.exists():
        archive = run_directory / "arrays.npz"
    arrays = np.load(archive, allow_pickle=False)
    config = load_config(config_path)
    workspace = replace(
        config.controller.workspace,
        grid=jnp.asarray(np.asarray(arrays["grid"], dtype=np.float32)),
        grid_origin=jnp.asarray(np.asarray(arrays["grid_origin"], dtype=np.float32)),
        grid_resolution=float(arrays["grid_resolution"]),
    )
    return replace(config, controller=replace(config.controller, workspace=workspace)), arrays


def overwrite_observation(
    carry: SingleControllerState, observation: jax.Array
) -> SingleControllerState:
    """Overwrite feedback state and the newest memory position as the ROS node does."""
    return carry._replace(
        state=observation,
        memory=carry.memory.at[-1].set(observation[:2]),
    )


def nudge_float32(values: np.ndarray, sign_mask: int, ulps: int) -> np.ndarray:
    """Move four float32 values by an exact number of representable values."""
    if not 0 <= sign_mask < 16 or ulps < 1:
        raise ValueError("sign_mask must be 0..15 and ulps must be positive")
    result = np.asarray(values, dtype=np.float32).reshape(4).copy()
    for axis in range(4):
        target = np.float32(np.inf if sign_mask & (1 << axis) else -np.inf)
        for _ in range(ulps):
            result[axis] = np.nextafter(result[axis], target, dtype=np.float32)
    return result


def _perturb_state(state: jax.Array, impulse: jax.Array, sign_mask: int, ulps: int) -> jax.Array:
    """Apply either one measured impulse or a four-axis ULP perturbation."""
    if ulps:
        values = state[:4]
        for axis in range(4):
            target = jnp.asarray(
                jnp.inf if sign_mask & (1 << axis) else -jnp.inf, dtype=jnp.float32
            )
            values = values.at[axis].set(
                jax.lax.fori_loop(
                    0, ulps, lambda _, value: jnp.nextafter(value, target), values[axis]
                )
            )
        return state.at[:4].set(values)
    return state + impulse


def _run_first_perturbation(
    config, initial_state: np.ndarray, impulse: np.ndarray, sign_mask: int, ulps: int,
    device: str, preflight_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one closed loop with feedback changed once after its first transition."""
    selected = select_device(device)
    params = jax.device_put(config.controller, selected)
    initial = jax.device_put(jnp.asarray(initial_state, dtype=jnp.float32), selected)
    controls = jax.device_put(
        jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32), selected
    )
    impulse_array = jax.device_put(jnp.asarray(impulse, dtype=jnp.float32), selected)

    def execute(params, initial, controls, key, impulse_array):
        controller = initialize_single(params, initial, controls, key)

        def preflight(carry, _):
            held, _ = stationary_step(params, carry, initial)
            return held, None

        controller, _ = jax.lax.scan(
            preflight, controller, xs=None, length=preflight_steps
        )
        controller, result = single_step(params, controller)
        observed = _perturb_state(controller.state, impulse_array, sign_mask, ulps)
        controller = overwrite_observation(controller, observed)
        first = (
            controller.state,
            effective_sample_fraction(result.weights, params.mppi.samples),
            controller.temperature,
        )

        def advance(carry, _):
            next_carry, output = single_step(params, carry)
            return next_carry, (
                next_carry.state,
                effective_sample_fraction(output.weights, params.mppi.samples),
                next_carry.temperature,
            )

        _, rest = jax.lax.scan(
            advance, controller, xs=None, length=config.run.steps - 1
        )
        return tuple(jnp.concatenate((head[None], tail), axis=0) for head, tail in zip(first, rest))

    path, ess, temperature = jax.jit(execute)(
        params, initial, controls, jax.device_put(controller_key(config.run.seed), selected),
        impulse_array,
    )
    return np.asarray(path), np.asarray(ess), np.asarray(temperature)


def _run_stepwise(
    config, initial_state: np.ndarray, device: str, preflight_steps: int
) -> np.ndarray:
    """Run the controller one compiled step at a time, matching the ROS node."""
    selected = select_device(device)
    params = jax.device_put(config.controller, selected)
    initial = jax.device_put(jnp.asarray(initial_state, dtype=jnp.float32), selected)
    carry = initialize_single(
        params,
        initial,
        jax.device_put(jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32), selected),
        jax.device_put(controller_key(config.run.seed), selected),
    )
    step = jax.jit(single_step)
    for _ in range(preflight_steps):
        carry, _ = step(params, carry)
        carry = carry._replace(
            state=initial,
            memory=jnp.broadcast_to(initial[:2], carry.memory.shape),
            step_index=jnp.asarray(0, dtype=jnp.int32),
        )
        jax.block_until_ready(carry.state)
    path = []
    for _ in range(config.run.steps):
        carry, _ = step(params, carry)
        path.append(carry.state)
    return np.asarray(jnp.stack(path))


def path_sha256(path: np.ndarray) -> str:
    """Hash one path with shape and dtype made explicit by its contiguous byte stream."""
    values = np.ascontiguousarray(path)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def tour_count(metrics: dict[str, float]) -> int:
    """Count the first completed tour plus subsequent complete cycles."""
    return int(np.isfinite(float(metrics["first_all_modes_s"]))) + int(metrics["mode_cycles"])


def _divergence_times(path: np.ndarray, canonical: np.ndarray, delta_t: float) -> dict[str, float]:
    count = min(len(path), len(canonical))
    if not count:
        return {"divergence_1cm_s": float("nan"), "divergence_10cm_s": float("nan"),
                "divergence_1m_s": float("nan")}
    distance = np.linalg.norm(path[:count, :2] - canonical[:count, :2], axis=1)
    result = {}
    for name, threshold in (("1cm", 0.01), ("10cm", 0.1), ("1m", 1.0)):
        indices = np.flatnonzero(distance > threshold)
        result[f"divergence_{name}_s"] = (
            float(indices[0] * delta_t) if indices.size else float("nan")
        )
    return result


def _score_path(
    path: np.ndarray, arrays, config, canonical: np.ndarray | None
) -> dict[str, float | int | str]:
    params = config.controller
    workspace = params.workspace
    delta_t = float(params.model.delta_t)
    duration = config.run.steps * delta_t
    metrics = compute_mode_metrics(
        path[:, :2], np.asarray(params.gmm.means),
        np.asarray(params.gmm.covariance_inverse), delta_t,
    )
    tours = tour_count(metrics)
    scored: dict[str, float | int | str] = {
        **metrics,
        "tour_count": tours,
        "tours_per_1000s": tours / duration * 1000.0,
        "zero_tour": int(tours == 0),
        "restricted_first_tour_s": (
            float(metrics["first_all_modes_s"])
            if np.isfinite(metrics["first_all_modes_s"]) else duration
        ),
        "occupancy_mse": compute_team_ergodic_error(
            path[:, None, :2], arrays["target_grid"],
            tuple(map(float, workspace.x_limits)), tuple(map(float, workspace.y_limits)),
            reachable_mask=arrays["reachable_mask"],
        ),
        "fourier_ergodic": compute_fourier_ergodic_metric(
            path[:, None, :2], arrays["target_grid"],
            tuple(map(float, workspace.x_limits)), tuple(map(float, workspace.y_limits)),
            reachable_mask=arrays["reachable_mask"],
        ),
        "path_sha256": path_sha256(path),
    }
    scored.update(
        _divergence_times(path, canonical, delta_t) if canonical is not None else
        _divergence_times(np.zeros((0, 2)), np.zeros((0, 2)), delta_t)
    )
    return scored


def _read_canonical(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["path"])


def _row_exists(output: Path, identity: dict[str, object]) -> bool:
    if not output.exists():
        return False
    with output.open(encoding="utf-8", newline="") as stream:
        return any(all(row.get(key) == str(value) for key, value in identity.items())
                   for row in csv.DictReader(stream))


def discrepancy_jax(
    *, run_directory: Path, config_path: str, output: Path, condition: str,
    steps: int, seed: int, preflight_steps: int, device: str, map_seed: int,
    repeat: int, hardware: str, sign_mask: int, ulps: int,
    residual_run: Path | None, canonical_path: Path | None,
) -> None:
    """Run and archive one exact, ULP, or first-residual pure-JAX cell."""
    config, arrays = _grid_config(run_directory, config_path)
    config = replace(config, run=replace(config.run, seed=seed, steps=steps))
    initial_state = np.asarray(arrays["initial_state"], dtype=np.float32)
    impulse = np.zeros(6, dtype=np.float32)
    perturbation_id = "none"
    if condition == "ulp":
        if ulps not in (1, 8):
            raise ValueError("ULP cells use exactly one or eight ULPs")
        perturbation_id = f"ulp{ulps}_mask{sign_mask:02d}"
    elif condition == "measured":
        if residual_run is None:
            raise ValueError("--residual-run is required for a measured impulse")
        with np.load(residual_run / "arrays.npz", allow_pickle=False) as residual_arrays:
            residuals = feedback_residuals(residual_arrays)
        if not residuals["state"].size:
            raise ValueError(f"no valid feedback residual in {residual_run}")
        impulse[:4] = residuals["state"][0]
        perturbation_id = residual_run.name
    elif condition not in {"exact", "zero", "stepwise"}:
        raise ValueError(f"unknown JAX condition: {condition}")

    identity = {
        "source": "jax", "condition": condition, "map_seed": map_seed,
        "hardware": hardware, "repeat": repeat, "perturbation_id": perturbation_id,
    }
    if _row_exists(output, identity):
        print(f"already recorded {identity}")
        return
    started = time.perf_counter()
    if condition == "exact":
        simulation = run_simulation(
            config, device=device, initial_state=initial_state, preflight_steps=preflight_steps
        )
        path = simulation.paths[:, 0]
        selected_device = simulation.device
    elif condition == "stepwise":
        path = _run_stepwise(config, initial_state, device, preflight_steps)
        selected_device = select_device(device).platform
    else:
        path, _, _ = _run_first_perturbation(
            config, initial_state, impulse, sign_mask, ulps if condition == "ulp" else 0,
            device, preflight_steps,
        )
        selected_device = select_device(device).platform
    canonical = _read_canonical(canonical_path)
    row = dict.fromkeys(DISCREPANCY_FIELDS, "")
    row.update(identity)
    row.update({
        "vehicle": "none", "seed": seed, "ulp_count": ulps if condition == "ulp" else 0,
        "impulse_px": float(impulse[0]), "impulse_py": float(impulse[1]),
        "impulse_vx": float(impulse[2]), "impulse_vy": float(impulse[3]),
        "steps": steps, "preflight_steps": preflight_steps,
        "wall_seconds": time.perf_counter() - started, "device": selected_device,
        "jax_version": jax.__version__, "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "run_id": f"jax_{condition}_{map_seed}_{repeat}_{perturbation_id}",
    })
    row.update(_score_path(path, arrays, config, canonical))
    append_csv(output, row, DISCREPANCY_FIELDS)
    if canonical_path is not None and condition == "exact" and not canonical_path.exists():
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(canonical_path, path=path)
    print(f"{row['run_id']}: tours={row['tour_count']} hash={row['path_sha256'][:12]}")


def feedback_residuals(arrays) -> dict[str, np.ndarray]:
    """Return command-to-next-feedback residuals using only causally available odometry."""
    commands = np.asarray(arrays["cmd_raw"], dtype=np.float64).reshape(-1, 8)
    odometry = np.asarray(arrays["odometry"], dtype=np.float64).reshape(-1, 8)
    empty = {
        "state": np.zeros((0, 4)), "observed": np.zeros((0, 6)),
        "ages": np.zeros(0), "gaps": np.zeros(0),
    }
    if commands.shape[0] < 2 or not odometry.size:
        return empty
    feedback_times = commands[1:, 0]
    indices = np.searchsorted(odometry[:, 0], feedback_times, side="right") - 1
    valid = (indices >= 0) & (indices < odometry.shape[0])
    if not valid.any():
        return empty
    indices = indices[valid]
    predicted = commands[:-1][valid][:, [1, 2, 4, 5]]
    measured = odometry[indices][:, [1, 2, 4, 5]]
    observed = np.column_stack(
        (measured, np.zeros(measured.shape[0]), odometry[indices, 7])
    )
    return {
        "state": measured - predicted,
        "observed": observed,
        "ages": feedback_times[valid] - odometry[indices, 0],
        "gaps": np.diff(commands[:, 0])[:valid.size][valid],
    }


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def discrepancy_ros(
    *, run_directory: Path, config_path: str, output: Path, vehicle: str,
    repeat: int, hardware: str, canonical_path: Path | None,
) -> None:
    """Score one recorded ideal-vehicle or SO3 run for causal attribution."""
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    config, arrays = _grid_config(run_directory, config_path)
    config = replace(
        config,
        run=replace(
            config.run, seed=int(manifest["seed"]), steps=int(manifest["steps"])
        ),
    )
    identity = {
        "source": "ros", "condition": vehicle, "map_seed": int(manifest["map_seed"]),
        "hardware": hardware, "repeat": repeat, "perturbation_id": "none",
    }
    if _row_exists(output, identity):
        print(f"already recorded {identity}")
        return
    odometry = np.asarray(arrays["odometry"], dtype=np.float64).reshape(-1, 8)
    feedback = feedback_residuals(arrays)
    path = feedback["observed"]
    if not path.size:
        raise ValueError(f"no command-aligned feedback in {run_directory}")
    duration = float(config.run.steps * config.controller.model.delta_t)
    sample_period = float(config.controller.model.delta_t)
    params = config.controller
    metrics = compute_mode_metrics(
        path[:, :2], np.asarray(params.gmm.means),
        np.asarray(params.gmm.covariance_inverse), sample_period,
    )
    tours = tour_count(metrics)
    scored = {
        **metrics,
        "tour_count": tours,
        "tours_per_1000s": tours / duration * 1000.0,
        "zero_tour": int(tours == 0),
        "restricted_first_tour_s": (
            float(metrics["first_all_modes_s"])
            if np.isfinite(metrics["first_all_modes_s"]) else duration
        ),
        "occupancy_mse": compute_team_ergodic_error(
            path[:, None, :2], arrays["target_grid"],
            tuple(map(float, params.workspace.x_limits)),
            tuple(map(float, params.workspace.y_limits)),
            reachable_mask=arrays["reachable_mask"],
        ),
        "fourier_ergodic": compute_fourier_ergodic_metric(
            path[:, None, :2], arrays["target_grid"],
            tuple(map(float, params.workspace.x_limits)),
            tuple(map(float, params.workspace.y_limits)),
            reachable_mask=arrays["reachable_mask"],
        ),
        "path_sha256": path_sha256(path),
    }
    early = feedback["state"][:500]
    position_error = np.linalg.norm(early[:, :2], axis=1)
    velocity_error = np.linalg.norm(early[:, 2:], axis=1)
    gaps = feedback["gaps"][:500] * 1000.0
    canonical = _read_canonical(canonical_path)
    scored.update(
        _divergence_times(
            feedback["observed"], canonical, float(params.model.delta_t)
        ) if canonical is not None else
        _divergence_times(np.zeros((0, 2)), np.zeros((0, 2)), float(params.model.delta_t))
    )
    row = dict.fromkeys(DISCREPANCY_FIELDS, "")
    row.update(identity)
    row.update(scored)
    row.update({
        "vehicle": vehicle, "seed": config.run.seed, "ulp_count": 0,
        "steps": config.run.steps,
        "preflight_steps": int(manifest.get("preflight_steps", 0)),
        "feedback_pos_p50_m": _percentile(position_error, 50),
        "feedback_pos_p95_m": _percentile(position_error, 95),
        "feedback_vel_p50_mps": _percentile(velocity_error, 50),
        "feedback_vel_p95_mps": _percentile(velocity_error, 95),
        "callback_gap_p50_ms": _percentile(gaps, 50),
        "callback_gap_p95_ms": _percentile(gaps, 95),
        "callback_gap_max_ms": float(gaps.max()) if gaps.size else float("nan"),
        "odom_age_p95_ms": _percentile(feedback["ages"][:500] * 1000.0, 95),
        "guard_fraction": float(np.mean(arrays["guard_state"] != "pass"))
        if arrays["guard_state"].size else 0.0,
        "wall_seconds": float(odometry[-1, 0] - odometry[0, 0])
        if len(odometry) > 1 else 0.0,
        "device": manifest.get("device", "unknown"),
        "jax_version": manifest.get("jax_version", "unknown"),
        "xla_flags": manifest.get("xla_flags", ""),
        "run_id": manifest["run_id"],
    })
    append_csv(output, row, DISCREPANCY_FIELDS)
    print(f"{row['run_id']}: tours={tours} first-feedback-p95={row['feedback_pos_p95_m']:.4g} m")


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def bootstrap_difference(
    left: np.ndarray, right: np.ndarray, samples: int = 4000
) -> tuple[float, float, float]:
    """Return right-minus-left mean and a deterministic percentile interval."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not left.size or not right.size:
        return float("nan"), float("nan"), float("nan")
    generator = np.random.default_rng(0)
    differences = np.empty(samples)
    for index in range(samples):
        differences[index] = (
            generator.choice(right, right.size, replace=True).mean()
            - generator.choice(left, left.size, replace=True).mean()
        )
    return (
        float(right.mean() - left.mean()),
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def attribute_map(rows: list[dict[str, str]], map_seed: int) -> str:
    """Classify one map using the predeclared numerical/ROS/vehicle logic."""
    selected = [row for row in rows if int(float(row.get("map_seed", -1))) == map_seed]
    null = np.asarray([
        _number(row, "tour_count") for row in selected
        if row.get("source") == "jax"
        and row.get("condition") in {"exact", "zero", "ulp", "measured"}
    ])
    ideal = np.asarray([
        _number(row, "tour_count") for row in selected
        if row.get("source") == "ros" and row.get("vehicle") == "ideal"
    ])
    so3 = np.asarray([
        _number(row, "tour_count") for row in selected
        if row.get("source") == "ros" and row.get("vehicle") == "so3"
    ])
    if not null.size or not ideal.size or not so3.size:
        return "unresolved"
    _, lower, upper = bootstrap_difference(ideal, so3)
    if lower > 0 or upper < 0:
        return "vehicle-dynamics"
    low, high = float(null.min()), float(null.max())
    ideal_side = -1 if ideal.mean() < low else 1 if ideal.mean() > high else 0
    so3_side = -1 if so3.mean() < low else 1 if so3.mean() > high else 0
    if ideal_side and ideal_side == so3_side:
        return "ros-feedback/scheduling"
    if np.all((ideal >= low) & (ideal <= high)) and np.all((so3 >= low) & (so3 <= high)):
        return "numerical/rare-event"
    return "unresolved"


def _summary(values: np.ndarray) -> str:
    if not values.size:
        return "n=0"
    return (
        f"n={values.size}, mean={values.mean():.2f}, range={values.min():.0f}..{values.max():.0f}, "
        f"zero={100 * np.mean(values == 0):.0f}%"
    )


def _median(rows: list[dict[str, str]], key: str) -> float:
    values = np.asarray([_number(row, key) for row in rows])
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def build_discrepancy_report(rows: list[dict[str, str]]) -> str:
    """Build the falsification-first JAX-versus-UAV attribution report."""
    maps = sorted({int(float(row["map_seed"])) for row in rows if row.get("map_seed")})
    exact = [row for row in rows if row.get("source") == "jax" and row.get("condition") == "exact"]
    unstable = []
    incomplete = []
    determinism_lines = []
    for map_seed in maps:
        hardware = sorted({
            row["hardware"] for row in exact
            if int(float(row["map_seed"])) == map_seed
        })
        if len(hardware) < 2:
            incomplete.append((map_seed, "second hardware"))
        for machine in hardware:
            group = [row for row in exact if int(float(row["map_seed"])) == map_seed
                     and row["hardware"] == machine]
            hashes = {row["path_sha256"] for row in group}
            verdict = "STOP" if len(hashes) > 1 else "PASS" if len(group) >= 5 else "PENDING"
            determinism_lines.append(
                f"| {map_seed} | {machine} | {len(group)} | {len(hashes)} | "
                f"{verdict} |"
            )
            if len(hashes) > 1:
                unstable.append((map_seed, machine))
            elif len(group) < 5:
                incomplete.append((map_seed, machine))

    distribution_lines = []
    comparison_lines = []
    feedback_lines = []
    attributions = {}
    for map_seed in maps:
        for source, condition in (
            ("jax", "exact"), ("jax", "zero"), ("jax", "ulp"), ("jax", "measured"),
            ("ros", "ideal"), ("ros", "so3"),
        ):
            values = np.asarray([
                _number(row, "tour_count") for row in rows
                if int(float(row.get("map_seed", -1))) == map_seed
                and row.get("source") == source
                and (row.get("condition") == condition or row.get("vehicle") == condition)
            ])
            distribution_lines.append(f"| {map_seed} | {source}:{condition} | {_summary(values)} |")
        ideal = np.asarray([_number(row, "tour_count") for row in rows
                            if row.get("source") == "ros" and row.get("vehicle") == "ideal"
                            and int(float(row["map_seed"])) == map_seed])
        so3 = np.asarray([_number(row, "tour_count") for row in rows
                          if row.get("source") == "ros" and row.get("vehicle") == "so3"
                          and int(float(row["map_seed"])) == map_seed])
        difference, lower, upper = bootstrap_difference(ideal, so3)
        ideal_first = np.asarray([_number(row, "restricted_first_tour_s") for row in rows
                                  if row.get("source") == "ros" and row.get("vehicle") == "ideal"
                                  and int(float(row["map_seed"])) == map_seed])
        so3_first = np.asarray([_number(row, "restricted_first_tour_s") for row in rows
                                if row.get("source") == "ros" and row.get("vehicle") == "so3"
                                and int(float(row["map_seed"])) == map_seed])
        first_difference = (
            float(so3_first.mean() - ideal_first.mean())
            if ideal_first.size and so3_first.size else float("nan")
        )
        attributions[map_seed] = (
            "reproducibility-broken" if unstable else attribute_map(rows, map_seed)
        )
        if ideal.size and so3.size:
            comparison_lines.append(
                f"| {map_seed} | {difference:.2f} [{lower:.2f}, {upper:.2f}] | "
                f"{100 * np.mean(ideal == 0):.0f}% | {100 * np.mean(so3 == 0):.0f}% | "
                f"{first_difference:.1f} | {attributions[map_seed]} |"
            )
        else:
            comparison_lines.append(f"| {map_seed} | n/a | n/a | n/a | n/a | unresolved |")
        for vehicle in ("ideal", "so3"):
            group = [row for row in rows if row.get("source") == "ros"
                     and row.get("vehicle") == vehicle
                     and int(float(row["map_seed"])) == map_seed]
            feedback_lines.append(
                f"| {map_seed} | {vehicle} | {len(group)} | "
                f"{_median(group, 'feedback_pos_p95_m'):.4f} | "
                f"{_median(group, 'feedback_vel_p95_mps'):.4f} | "
                f"{_median(group, 'callback_gap_p95_ms'):.2f} | "
                f"{_median(group, 'odom_age_p95_ms'):.2f} | "
                f"{_median(group, 'guard_fraction'):.4f} | "
                f"{_median(group, 'divergence_1cm_s'):.2f} / "
                f"{_median(group, 'divergence_10cm_s'):.2f} / "
                f"{_median(group, 'divergence_1m_s'):.2f} |"
            )

    if unstable:
        headline = (
            "STOP: same-hardware cross-process determinism failed; flight attribution is invalid."
        )
    elif incomplete:
        headline = (
            "PENDING: cross-process determinism needs five runs on each of two machines per map."
        )
    elif len(attributions) >= 2 and attributions.get(516) == "numerical/rare-event" \
            and attributions.get(539) not in {None, "numerical/rare-event", "unresolved"}:
        headline = "Geometry interaction: separation is specific to map 539."
    elif attributions and len(set(attributions.values())) == 1:
        headline = f"Attribution: {next(iter(attributions.values()))}."
    else:
        headline = (
            "Attribution unresolved: map-specific intervals do not support one stable mechanism."
        )

    return "\n".join([
        "# JAX-to-UAV discrepancy attribution", "", f"**{headline}**", "",
        "## Already falsified", "",
        "- Node infidelity: measured-state replay matched published commands to 5.06 mm mean, "
        "8.62 mm p95, with no drift (4.76 to 5.37 mm); this is the 0.868 m/s × ~6 ms "
        "odometry sampling granularity.",
        "- Tracking lag: best alignment was 0–1 steps and flights travelled 6% farther than "
        "their offline twins.",
        "- Curl/mirror symmetry: theta=0 wound more (+3.84, +2.07, +5.04 turns versus "
        "theta=15's -0.66) and starved m2 harder.",
        "- Safety shield: command displacement was 0.3 mm.", "",
        "The surviving premise is a rare-event metric: memory_time is 5.5 s, a tour emerges "
        "over about 160 s, and a 400 s run contains only 2.5 such opportunities. The same pure "
        "JAX config scored 3 cycles on Jeff and 0 on the laptop. Any flight-only mechanism "
        "must first exceed that vehicle-free numerical split.", "",
        "Evidence scope: 3 SO3 flights exist in the current geometry; 24 across all campaigns, "
        "of which 2 scored one cycle. Every recorded flight is theta_15.", "",
        "## Cross-process determinism", "",
        "| Map | Hardware | Runs | Path hashes | Verdict |", "|---:|---|---:|---:|---|",
        *(determinism_lines or ["| — | — | 0 | 0 | pending |"]), "",
        "## Outcome distributions", "",
        "| Map | Condition | Tour count |", "|---:|---|---|",
        *(distribution_lines or ["| — | — | pending |"]), "",
        "## Ideal ROS versus SO3 UAV", "",
        "| Map | SO3−ideal mean tours (95% bootstrap) | Ideal zero | SO3 zero | "
        "SO3−ideal restricted first tour (s) | Attribution |",
        "|---:|---:|---:|---:|---:|---|",
        *(comparison_lines or ["| — | n/a | n/a | n/a | n/a | pending |"]), "",
        "## First 500 feedback steps", "",
        "| Map | Vehicle | Runs | Position p95 (m) | Velocity p95 (m/s) | Callback p95 (ms) "
        "| Odom age p95 (ms) | Guard fraction | Divergence 1cm / 10cm / 1m (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        *(feedback_lines or ["| — | — | 0 | n/a | n/a | n/a | n/a | n/a | n/a |"]), "",
        "Primary endpoint: tour count. First-tour failures are censored at 400 s. Maps are "
        "analyzed separately; no controller parameter is changed by this diagnostic.", "",
    ])


def discrepancy_report(input_path: Path, output: Path) -> None:
    """Read diagnostic cells and write the causal attribution report."""
    with input_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_discrepancy_report(rows), encoding="utf-8")


def _scaled(params, penalty_scale: float, boundary_scale: float, lam_max: float | None):
    """Apply a penalty rescale and temperature cap to controller params."""
    workspace = params.workspace
    if penalty_scale != 1.0:
        workspace = replace(
            workspace,
            obstacle_cost=workspace.obstacle_cost * penalty_scale,
            out_of_map_cost=workspace.out_of_map_cost * penalty_scale,
            boundary_weight=workspace.boundary_weight * penalty_scale,
        )
    if boundary_scale != 1.0:
        workspace = replace(
            workspace, boundary_weight=workspace.boundary_weight * boundary_scale
        )
    mppi = params.mppi
    if lam_max is not None:
        mppi = replace(mppi, temperature_max=lam_max)
    return replace(params, workspace=workspace, mppi=mppi)


def ess_sweep(run_directory: Path, config_path: str, output: Path, steps: int,
              seed: int, device: str) -> None:
    """Report where the adaptive temperature converges under each candidate setting."""
    config, arrays = _grid_config(run_directory, config_path)
    initial = jnp.asarray(arrays["snap_state"][0], dtype=jnp.float32)
    combinations = [
        ("shipped", 1.0, 1.0, None),
        ("lam_max_1e3", 1.0, 1.0, 1e3),
        ("lam_max_1e4", 1.0, 1.0, 1e4),
        ("lam_max_1e5", 1.0, 1.0, 1e5),
        ("penalty_0.1", 0.1, 1.0, None),
        ("penalty_0.01", 0.01, 1.0, None),
        ("penalty_0.1_lam1e3", 0.1, 1.0, 1e3),
        ("penalty_0.01_lam1e3", 0.01, 1.0, 1e3),
        ("boundary_0.1_lam1e3", 1.0, 0.1, 1e3),
        ("boundary_0.01_lam1e3", 1.0, 0.01, 1e3),
    ]
    step = jax.jit(single_step)
    for name, penalty, boundary, lam_max in combinations:
        params = _scaled(config.controller, penalty, boundary, lam_max)
        carry = initialize_single(
            params, initial,
            jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32),
            jax.random.key(seed),
        )
        fractions, temperature = [], 0.0
        for _ in range(steps):
            carry, result = step(params, carry)
            weights = np.asarray(result.weights)
            fractions.append(100.0 / (np.sum(weights * weights) * weights.size))
            temperature = float(carry.temperature)
        settled = float(np.mean(fractions[-min(200, steps // 3):]))
        cap = params.mppi.temperature_max
        row = {
            "arm": name, "penalty_scale": penalty, "boundary_scale": boundary,
            "lam_max": cap, "steps": steps, "seed": seed,
            "lambda_final": temperature,
            "lambda_pinned": int(abs(temperature - cap) / cap < 1e-6),
            "ess_percent_median": float(np.median(fractions)),
            "ess_percent_settled": settled,
            "ess_target_percent": 100.0 * params.mppi.ess_target,
            "samples": params.mppi.samples, "device": device,
        }
        append_csv(output, row, ESS_FIELDS)
        print(f"{name:<22} lambda={temperature:>9.1f} "
              f"{'PINNED' if row['lambda_pinned'] else '     '} "
              f"ESS_settled={settled:5.2f}%", flush=True)


def cost_decomposition(run_directory: Path, config_path: str, output: Path,
                       device: str) -> None:
    """Split one step's per-rollout cost into terms, by cross-rollout spread."""
    config, arrays = _grid_config(run_directory, config_path)
    params = config.controller
    for index, recorded in enumerate(arrays["snapshot_steps"]):
        carry = restore_snapshot(arrays, index)
        epsilon, _ = sample_epsilon(carry.key, params)
        costs, _, positions = _rollouts(
            params, carry.state, carry.controls, epsilon, carry.temperature
        )
        limits = params.workspace
        gap = jnp.minimum(
            jnp.minimum(positions[..., 0] - limits.x_limits[0],
                        limits.x_limits[1] - positions[..., 0]),
            jnp.minimum(positions[..., 1] - limits.y_limits[0],
                        limits.y_limits[1] - positions[..., 1]),
        )
        encroach = jnp.maximum(limits.boundary_margin - gap, 0.0)
        outside = (
            (positions[..., 0] < limits.x_limits[0]) | (positions[..., 0] > limits.x_limits[1])
            | (positions[..., 1] < limits.y_limits[0]) | (positions[..., 1] > limits.y_limits[1])
        )
        terms = {
            "grid_obstacle": np.asarray(_grid_cost(positions, params)).sum(1),
            "boundary_margin": np.asarray(limits.boundary_weight * encroach * encroach).sum(1),
            "out_of_map": np.asarray(outside * limits.out_of_map_cost).sum(1),
        }
        total = np.asarray(costs)
        terms["flow_and_control"] = total - sum(terms.values())
        total_spread = float(np.percentile(total, 99) - total.min()) or 1.0
        for term, values in terms.items():
            spread = float(np.percentile(values, 99) - values.min())
            append_csv(output, {
                "step": int(recorded), "term": term, "mean": float(values.mean()),
                "std": float(values.std()), "spread_p99_min": spread,
                "share_of_total_spread": spread / total_spread, "device": device,
            }, COST_FIELDS)
        print(f"step {int(recorded):>6}  " + "  ".join(
            f"{t}={float(np.percentile(v, 99) - v.min()):.3g}" for t, v in terms.items()
        ), flush=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["ess", "costs", "discrepancy-jax", "discrepancy-ros", "discrepancy-report"],
    )
    parser.add_argument("--run-dir", type=Path,
                        help="Run directory holding figure_data.npz or arrays.npz")
    parser.add_argument("--config", default="configs/uav_profile.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--condition", choices=["exact", "zero", "ulp", "measured", "stepwise"]
    )
    parser.add_argument("--map-seed", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--preflight-steps", type=int, default=200)
    parser.add_argument("--hardware", default=socket.gethostname())
    parser.add_argument("--sign-mask", type=int, default=0)
    parser.add_argument("--ulps", type=int, default=0)
    parser.add_argument("--residual-run", type=Path)
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--vehicle", choices=["ideal", "so3"])
    parser.add_argument("--input", type=Path)
    arguments = parser.parse_args()
    default = Path("results/uav") / f"diagnostic_{arguments.mode}.csv"
    output = arguments.output or default
    if arguments.mode == "ess":
        if arguments.run_dir is None:
            parser.error("--run-dir is required")
        ess_sweep(arguments.run_dir, arguments.config, output, arguments.steps,
                  arguments.seed, arguments.device)
    elif arguments.mode == "costs":
        if arguments.run_dir is None:
            parser.error("--run-dir is required")
        cost_decomposition(arguments.run_dir, arguments.config, output, arguments.device)
    elif arguments.mode == "discrepancy-jax":
        if arguments.run_dir is None or arguments.condition is None or arguments.map_seed is None:
            parser.error("--run-dir, --condition, and --map-seed are required")
        discrepancy_jax(
            run_directory=arguments.run_dir, config_path=arguments.config, output=output,
            condition=arguments.condition, steps=arguments.steps, seed=arguments.seed,
            preflight_steps=arguments.preflight_steps, device=arguments.device,
            map_seed=arguments.map_seed, repeat=arguments.repeat, hardware=arguments.hardware,
            sign_mask=arguments.sign_mask, ulps=arguments.ulps,
            residual_run=arguments.residual_run, canonical_path=arguments.canonical,
        )
    elif arguments.mode == "discrepancy-ros":
        if arguments.run_dir is None or arguments.vehicle is None:
            parser.error("--run-dir and --vehicle are required")
        discrepancy_ros(
            run_directory=arguments.run_dir, config_path=arguments.config, output=output,
            vehicle=arguments.vehicle, repeat=arguments.repeat, hardware=arguments.hardware,
            canonical_path=arguments.canonical,
        )
    else:
        if arguments.input is None:
            parser.error("--input is required")
        discrepancy_report(arguments.input, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

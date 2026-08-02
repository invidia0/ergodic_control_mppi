"""Re-run a recorded UAV trial as an ideal offline trial and score it identically.

The point of the pairing is that the only difference between the two rows is the vehicle:
same map grid, same start state, same seed, same timestep, same density, same controller
configuration. Any degradation the summary shows is therefore attributable to flying it.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.summary import append_summary, compute_row
from ergodic_control_mppi.simulation import run_simulation


def pair_run(run_directory: Path, device: str = "auto", summary: Path | None = None) -> dict:
    """Run the ideal twin of a recorded trial and append its summary row.

    Args:
        run_directory: A recorder output directory holding ``manifest.json`` and
            ``arrays.npz``.
        device: Requested JAX device selection.
        summary: Summary CSV to append to; defaults to the recorder's own.

    Returns:
        The appended row.
    """
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(run_directory / "arrays.npz", allow_pickle=False)
    steps = int(manifest["steps"])
    if steps <= 0:
        raise ValueError(f"{run_directory} recorded no steps; nothing to pair")

    config = load_config(_resolve_config(manifest))
    grid = np.asarray(arrays["grid"], dtype=np.float32)
    # Plan against the inflated grid, judge contact against the physical map -- the same
    # split the recorder uses, so both rows in a pair are scored identically.
    occupancy = (
        np.asarray(arrays["occupancy"]) > 0 if "occupancy" in arrays.files else grid > 0
    )
    origin = tuple(float(value) for value in np.asarray(arrays["grid_origin"]))
    resolution = float(arrays["grid_resolution"])
    # Same obstacles the UAV actually flew against, as a runtime grid rather than circles.
    workspace = replace(
        config.controller.workspace,
        grid=jnp.asarray(grid),
        grid_origin=jnp.asarray(origin, dtype=jnp.float32),
        grid_resolution=resolution,
    )
    config = replace(
        config,
        run=replace(config.run, steps=steps, seed=int(manifest["seed"])),
        controller=replace(config.controller, workspace=workspace),
    )

    started = time.perf_counter()
    result = run_simulation(
        config,
        device=device,
        initial_state=np.asarray(arrays["initial_state"]),
        preflight_steps=int(manifest.get("preflight_steps", 0)),
    )
    wall_seconds = time.perf_counter() - started

    delta_t = float(manifest["delta_t"])
    positions = result.paths[:, 0, :2]
    row = compute_row(
        identity={
            "run_id": manifest["run_id"],
            "profile": manifest["profile"],
            "mode": "ideal",
            "seed": manifest["seed"],
            "map_seed": manifest["map_seed"],
            "map_fill": manifest["map_fill"],
            "steps": steps,
            "compile_s": "",
            "run_hash": "",
            "config_hash": manifest["config_hash"],
            "git_sha": manifest["git_sha"],
            "seed_controller": manifest["seed"],
            "jax_version": jax.__version__,
            "ros_distro": "",
            "device": result.device,
            "steps_to_threshold": "",
        },
        positions=positions,
        target_grid=np.asarray(arrays["target_grid"]),
        x_limits=tuple(float(v) for v in np.asarray(config.controller.workspace.x_limits)),
        y_limits=tuple(float(v) for v in np.asarray(config.controller.workspace.y_limits)),
        reachable_mask=np.asarray(arrays["reachable_mask"]),
        gmm_means=np.asarray(config.controller.gmm.means),
        gmm_inverses=np.asarray(config.controller.gmm.covariance_inverse),
        delta_t=delta_t,
        occupancy=occupancy,
        grid_origin=origin,
        grid_resolution=resolution,
        robot_radius=float(manifest["robot_radius"]),
        # An ideal run has no guard and tracks its own setpoints exactly, so the guard and
        # tracking columns are structurally zero rather than measured.
        guard_states=np.full(positions.shape[0], "pass"),
        guard_period=delta_t,
        speeds=np.linalg.norm(result.paths[:, 0, 2:4], axis=1),
        actual_times=np.arange(positions.shape[0]) * delta_t,
        commanded_times=np.arange(positions.shape[0]) * delta_t,
        commanded=positions,
        step_ms=np.full(steps, wall_seconds * 1e3 / steps),
        deadline_ms=float(manifest["deadline_ms"]),
        wall_seconds=wall_seconds,
        odometry_seconds=steps * delta_t,
        control_seconds=wall_seconds,
    )
    append_summary(summary or run_directory.parent / "summary.csv", row)
    np.savez_compressed(
        run_directory / "ideal_diagnostics.npz",
        ess_fraction=result.ess_fractions,
        temperature=result.temperatures,
    )
    return row


def _resolve_config(manifest: dict) -> Path:
    """Find the run's config, whether recorded in a container or on this host.

    The manifest's absolute path is the one inside the container, so prefer it only if it
    exists here and fall back to the workspace-relative form.
    """
    for candidate in (manifest.get("config"), manifest.get("config_relative")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        f"cannot find the run's config: tried {manifest.get('config')!r} and "
        f"{manifest.get('config_relative')!r} from the current directory"
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--summary", type=Path, default=None)
    arguments = parser.parse_args()
    row = pair_run(arguments.run_dir, arguments.device, arguments.summary)
    print(
        f"paired {row['run_id']}: occupancy_mse={row['occupancy_mse']:.3e} "
        f"fourier={row['fourier_ergodic']:.4f} in_mode={row['in_mode_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()

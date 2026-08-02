"""Re-fly a recorded trial offline, keeping the planner internals the figures need.

A recorded flight stores what the vehicle *did*; the paper figures also want what the
planner was *considering* -- the rollout cloud, its weights, and the mean plan. Those are
too large to stream (``(K, T, 2)`` per step is ~700 kB at K=250, T=350, so ~14 GB over a
20 000-step run), so this walks the same closed loop with ``single_step`` and stores the
small carry at intervals. ``mppi.replay.replay_step`` expands any snapshot back into the
exact cloud that step used, because the carry holds the PRNG key it drew from.

This is the offline twin of the flight, not the flight itself: the vehicle's tracking error
and guard are absent, so use it for planner-internal figures and the recorded bag for the
video of the actual flight.

    uv run python -m ergodic_control_mppi.experiments.uav_figure_run \
        --run-dir results/uav/tuned_h5 --every 50
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.replay import snapshot_arrays
from ergodic_control_mppi.mppi.single import initialize_single, single_step
from ergodic_control_mppi.simulation import select_device


def figure_run(
    run_directory: Path, every: int, steps: int | None = None, device: str = "auto"
) -> Path:
    """Walk the loop with snapshots and write ``figure_data.npz`` beside the run.

    Args:
        run_directory: A recorder output directory to mirror.
        every: Snapshot interval in control steps.
        steps: Override the recorded step count, for a shorter figure run.
        device: JAX device selection.

    Returns:
        The path written.
    """
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(run_directory / "arrays.npz", allow_pickle=False)
    config = load_config(_config_path(manifest))
    total = int(steps or manifest["steps"])

    workspace = replace(
        config.controller.workspace,
        grid=jnp.asarray(np.asarray(arrays["grid"], dtype=np.float32)),
        grid_origin=jnp.asarray(
            [float(v) for v in np.asarray(arrays["grid_origin"])], dtype=jnp.float32
        ),
        grid_resolution=float(arrays["grid_resolution"]),
    )
    params = jax.device_put(
        replace(config.controller, workspace=workspace), select_device(device)
    )

    carry = initialize_single(
        params,
        jnp.asarray(np.asarray(arrays["initial_state"]), dtype=jnp.float32),
        jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32),
        jax.random.key(int(manifest["seed"])),
    )
    step = jax.jit(single_step)

    snapshots, positions, snapshot_steps = [], [], []
    for index in range(total):
        if index % every == 0:
            snapshots.append(carry)
            snapshot_steps.append(index)
        carry, _ = step(params, carry)
        positions.append(np.asarray(carry.state[:2]))

    output = run_directory / "figure_data.npz"
    np.savez_compressed(
        output,
        path=np.asarray(positions, dtype=np.float32),
        snapshot_steps=np.asarray(snapshot_steps, dtype=np.int32),
        grid=np.asarray(arrays["grid"]),
        occupancy=np.asarray(arrays["occupancy"]),
        grid_origin=np.asarray(arrays["grid_origin"]),
        grid_resolution=np.asarray(arrays["grid_resolution"]),
        target_grid=np.asarray(arrays["target_grid"]),
        reachable_mask=np.asarray(arrays["reachable_mask"]),
        gmm_means=np.asarray(config.controller.gmm.means),
        gmm_inverses=np.asarray(config.controller.gmm.covariance_inverse),
        delta_t=np.asarray(config.controller.model.delta_t),
        **snapshot_arrays(snapshots),
    )
    return output


def _config_path(manifest: dict) -> Path:
    """Resolve the run's config on this host, container path first."""
    for candidate in (manifest.get("config"), manifest.get("config_relative")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f"cannot find the config recorded in {manifest.get('run_id')!r}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--every", type=int, default=50, help="Snapshot interval in steps")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    arguments = parser.parse_args()
    output = figure_run(
        arguments.run_dir, arguments.every, arguments.steps, arguments.device
    )
    data = np.load(output, allow_pickle=False)
    print(
        f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB): "
        f"{data['path'].shape[0]} path samples, "
        f"{data['snapshot_steps'].size} replayable snapshots"
    )


if __name__ == "__main__":
    main()

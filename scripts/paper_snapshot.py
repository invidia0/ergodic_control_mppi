"""Fly the shipped profile on one campaign pillar map in pure JAX and draw it.

The ROS 2 + Gazebo flight is the deployment; this is the same controller on the same map
without the airframe, the tracker or the DDS layer in the way. It is the right artefact for
a figure of the *coverage behaviour*: the campaign was flown here, so a snapshot from here
is a picture of the system the ablation measured rather than of a re-run under a different
executor.

Seeds are flown as one batched call, which is also how the campaign flew them. The lane
count is a numerical branch -- a fixed width with different companions is bit-identical, a
different width is not -- so ``--seeds`` changes the numbers as well as the count. Pick the
width once and keep it.

    uv run python scripts/paper_snapshot.py --obs-num 25 --map-seed 516 --seeds 43,44,45
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.uav_pillar_tuning import _grid_config, score_run
from ergodic_control_mppi.mppi.single import run_batch, stack_params
from ergodic_control_mppi.plotting.deployment import trajectory_snapshot
from ergodic_control_mppi.simulation import controller_key, select_device

# Matches the campaign driver: the warm-up steps whose transient is not part of the run.
PREFLIGHT_STEPS = 200
REPORTED = ("fourier_ergodic", "occupancy_mse", "all_modes_reached", "mode_cycles",
            "mode_dwell_median_s", "in_mode_fraction", "speed_mps")


def _wrap_pdf(png: Path) -> Path:
    """Put the cropped raster into a PDF page, rather than re-rendering as vector.

    Saving this scene with a ``.pdf`` path does work, and produces a 56 MB file: the pillars
    and trail are a single scatter of roughly a quarter of a million points, and vector
    output stores each one as its own path. It is also necessarily uncropped, because
    `_crop_transparent` measures an alpha channel that a vector page does not have, so the
    scene would sit in the wide margin the crop exists to remove.

    Nothing is lost by rasterising. The render is already 2748 px on its long edge, which is
    over 800 dpi across a single column, and the content is a point cloud rather than line
    art -- there is no geometry a vector container would keep sharper. PDF has no alpha, so
    the transparent border is flattened onto white, which is what it sits on in the paper.
    """
    from PIL import Image

    with Image.open(png) as raster:
        page = Image.new("RGB", raster.size, "white")
        page.paste(raster, mask=raster.split()[-1] if raster.mode == "RGBA" else None)
        output = png.with_suffix(".pdf")
        # 600 dpi keeps the page a sane physical size; `width=\linewidth` rescales anyway.
        page.save(output, "PDF", resolution=600.0)
    return output


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-num", type=int, default=25)
    parser.add_argument("--map-seed", type=int, default=516)
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/report/snapshots"))
    # Bare by default: these are scene renders for a figure, not diagnostic plots, and the
    # ticks are what force the camera off the long-side axis (see --azimuth).
    parser.add_argument("--axes", action="store_true",
                        help="keep ticks, labels and axis lines")
    # -90 looks straight down -y, putting the workspace's long side across the frame. The
    # module default is -60 because at -90 mplot3d collapses the y and z ticks onto one
    # screen direction -- which costs nothing once the axes are gone.
    parser.add_argument("--azimuth", type=float, default=-90.0)
    parser.add_argument("--elevation", type=float, default=38.0)
    parser.add_argument("--vehicle-span", type=float, default=1.5,
                        help="drawn tip-to-tip width in metres; not to scale")
    # The trail shares one scatter with the pillar cloud so mplot3d depth-sorts them per
    # point; it therefore has to stay legible against `turbo_r` without competing with it.
    parser.add_argument("--trail-colour", default="#8A93A6",
                        help="path colour; lighter reads as a trace, darker as an object")
    parser.add_argument("--pillar-cmap", default="turbo_r",
                        help="matplotlib name, or a Scientific Colour Map (batlow, acton, "
                             "oslo, devon)")
    parser.add_argument("--pillar-alpha", type=float, default=1.0)
    # The target mixture on the floor. Greys keep it from competing with a
    # coloured pillar ramp for the reader's attention.
    parser.add_argument("--density-cmap", default="Blues")
    parser.add_argument("--vehicle-colour", default="#111111",
                        help="drone glyph colour, e.g. tab:red or tab:blue")
    parser.add_argument("--pillar-style", choices=("points", "cylinders"),
                        default="points")
    parser.add_argument("--suffix", default="", help="tag appended to the output stem")
    # Replay reads the flown paths from here, which is not necessarily where the renders go:
    # a sweep of looks writes to its own directory but must still find the one trajectory.
    parser.add_argument("--paths", type=Path, default=None,
                        help="directory holding the saved .npy paths (default: --output)")
    parser.add_argument("--pdf", action="store_true",
                        help="also wrap the cropped render in a PDF for LaTeX")
    # Re-render the saved paths instead of re-flying them. The flight is deterministic, so
    # this changes nothing about the picture -- it just does not need the GPU, which
    # matters while a campaign is using it.
    parser.add_argument("--replay", action="store_true",
                        help="render from the saved .npy paths rather than flying again")
    arguments = parser.parse_args()

    run_directory = Path(
        f"results/uav/density_{arguments.obs_num}/maps/map_{arguments.map_seed}")
    config, manifest, arrays = _grid_config(run_directory)
    seeds = [int(s) for s in arguments.seeds.split(",")]
    device = select_device(arguments.device)

    arguments.output.mkdir(parents=True, exist_ok=True)
    stems = [f"{arguments.obs_num}p_{arguments.map_seed}_s{s}" for s in seeds]

    source = arguments.paths or arguments.output
    if arguments.replay:
        paths = np.stack([np.load(source / f"{stem}.npy") for stem in stems])
        result = None
    else:
        stacked = stack_params([jax.device_put(config.controller, device)] * len(seeds))
        keys = jnp.stack([controller_key(s) for s in seeds])
        initial = jnp.asarray(np.asarray(arrays["initial_state"]), dtype=jnp.float32)
        controls = jnp.zeros((config.controller.mppi.horizon, 3), dtype=jnp.float32)

        print(f"{len(seeds)} lanes x {arguments.steps} steps on "
              f"{arguments.obs_num}p/{arguments.map_seed}, compiling...", flush=True)
        result = jax.jit(run_batch, static_argnames=("steps", "preflight_steps"))(
            stacked, initial, controls, keys,
            steps=arguments.steps, preflight_steps=PREFLIGHT_STEPS,
        )
        jax.block_until_ready(result.path)
        paths = np.asarray(result.path)

    rows = []
    for index, seed in enumerate(seeds):
        stem = stems[index]
        if result is None:
            # Replaying: the scores were written alongside the paths by the flight that
            # produced them, so they are read back rather than recomputed from a
            # trajectory whose velocities were never saved.
            row = {"seed": seed}
        else:
            row = score_run(
                config, arrays, manifest, seed, arguments.steps,
                positions=paths[index, :, :2], velocities=paths[index, :, 2:4],
                ess_fractions=np.asarray(result.ess_fraction)[index],
                temperatures=np.asarray(result.temperature)[index],
                wall=float("nan"), device=device.platform,
            )
            row["seed"] = seed
            np.save(source / f"{stem}.npy", paths[index, :, :2])
        rows.append(row)
        written = trajectory_snapshot(
            paths[index, :, :2], run_directory,
            arguments.output / f"{stem}{arguments.suffix}.png",
            title=f"seed {seed}" if arguments.axes else None,
            bare=not arguments.axes,
            elevation=arguments.elevation, azimuth=arguments.azimuth,
            # The campaign's map directories hold no sibling profile YAML, so the density
            # contour has to be handed the mixture the run was actually flown against.
            gmm=config.controller.gmm, vehicle_span=arguments.vehicle_span,
            trail_colour=arguments.trail_colour,
            pillar_cmap=arguments.pillar_cmap,
            density_cmap=arguments.density_cmap,
            vehicle_colour=arguments.vehicle_colour,
            pillar_alpha=arguments.pillar_alpha,
            pillar_style=arguments.pillar_style,
        )
        if arguments.pdf:
            print(f"           wrote {_wrap_pdf(written)}", flush=True)
        scores = "  ".join(f"{k}={float(row[k]):.4g}" for k in REPORTED if k in row)
        print(f"  seed {seed}: {scores}\n           wrote {written}", flush=True)

    if result is not None:
        (arguments.output / "scores.json").write_text(
            json.dumps(rows, indent=2, default=float), encoding="utf-8")
        print(f"wrote {arguments.output / 'scores.json'}")


if __name__ == "__main__":
    main()

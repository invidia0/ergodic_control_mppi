"""The two gate experiments that decide the shape of the remaining campaign.

``theta`` -- two adequately-powered campaigns disagree about the curl. On map 518
(153 occupied cells) theta = 15 won: 30 cycles against 12 at theta = 0, and theta = 0 missed
all-modes in 5 of 18 seeds. On map 516 (882 occupied cells) theta = 0 won, monotonically,
rho = -0.687 at p = 1.35e-09. The campaigns also differ in seed count, theta grid and
possibly the temperature regime, so the flip is real but unattributed. This runs one theta
grid, one seed set, one alpha, one runner across *both* maps. If theta = 0 wins on both, the
prior verdict was an artifact and the question closes; if the flip survives, it is real and
worth a density series.

``alpha`` -- a 48-cell probe found alpha = 1.0 triples the tour count. But the control-cost
coefficient is lambda*(1-alpha) and lambda is adapted online by the ESS controller, so alpha
sets a *coupling*, not a weight: a large control cost spreads the sample costs, the ESS loop
raises lambda to compensate, and that raises the control cost again. At alpha <= 0.9 the
runaway is complete -- lambda pinned at cap, ESS at the 1/K floor. If raising ``lam_max``
rescues alpha = 0.95 then the headline is "the cap was too low", not "remove the control
cost". These are not independent axes and must be swept together.

Both stages run at 20 000 steps and resume by identity so a queued job can be interrupted.

``--batch`` fuses each comparison into one ``run_batch`` call, ~6x faster because the loop
is launch-latency bound rather than throughput bound. It is a *different numerical branch*
than the sequential path -- see :func:`groups` and the ``execution`` field.
"""

import argparse
import csv
import socket
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.uav_ablation import _apply
from ergodic_control_mppi.experiments.uav_pillar_tuning import (
    PREFLIGHT_STEPS,
    _grid_config,
    run_cell,
    score_run,
)
from ergodic_control_mppi.mppi.single import run_batch, stack_params
from ergodic_control_mppi.simulation import controller_key, select_device

# The probe wrote every diagnostic *except* the settled temperature, which is the one number
# the alpha stage exists to read. Carried explicitly here so the lambda coupling can be
# measured rather than inferred from cap fractions.
#
# ``device`` and ``hardware`` are part of a run's identity, not metadata: the closed loop is
# chaotic at float32 scale, CPU and GPU backends are each deterministic but disagree with
# each other, and two different GPUs take different numerical branches. Rows from different
# hardware must never be pooled into one paired comparison, so the tag has to be in the file.
FIELDS = [
    "stage", "map_seed", "variant", "theta", "alpha", "lam_max", "seed", "steps",
    "ess_settled_median", "temperature_settled_median", "temperature_cap_fraction",
    "all_modes_reached", "first_all_modes_s", "mode_visits", "mode_cycles",
    "mode_dwell_median_s", "in_mode_fraction", "occupancy_mse", "fourier_ergodic",
    "collisions", "min_clearance_m", "path_length_m", "wall_seconds",
    "device", "hardware", "execution", "jax_version",
]

# Map 518 is the prior campaign's environment, 516 the pillar campaign's. Any run directory
# on that map serves -- only its archived grid, target and start state are used.
MAPS = {
    518: Path("results/uav/paper01"),
    516: Path("results/uav/pillar_25/flight_theta_15_516_s52"),
    539: Path("results/uav/pillar_25/flight_theta_15_539_s46"),
}


def groups(stage: str, seeds: range):
    """Yield ``(label, [lane, ...])`` where each group holds one *complete* comparison.

    A lane is ``(map_seed, variant, overrides, seed)``: ``overrides`` goes straight to
    ``_apply``, so a new axis is a new dict key rather than a new column.

    Grouping is not an efficiency detail. Batched lanes are a different numerical branch and
    the lane count changes reduction order, so a comparison split across two calls is split
    across two branches. Each group below therefore holds every cell its own comparison
    needs, at or under 48 lanes -- where the measured throughput curve flattens.
    """
    if stage == "theta":
        # At alpha = 0.95, the operating point both campaigns used: a historical
        # contradiction has to be resolved where it lives, not at a new operating point.
        # 539 was added after the alpha finding turned out to be map-specific: two maps
        # with opposite signs is the same evidence base alpha had before a third map killed
        # it, so theta gets the third map before anything is written about it.
        for map_seed in (518, 516, 539):
            yield f"map{map_seed}", [
                (map_seed, f"theta{theta:g}", {"theta": theta}, seed)
                for theta in (0.0, 15.0, 45.0) for seed in seeds
            ]
    elif stage == "alpha":
        # theta = 15, the flown arm. Grouped by lam_max so each call holds a whole alpha
        # comparison rather than splitting one across branches.
        for lam_max in (1e3, 1e4, 1e5):
            yield f"lam{lam_max:g}", [
                (516, f"a{alpha:g}", {"theta": 15.0, "alpha": alpha, "lam_max": lam_max},
                 seed)
                for alpha in (0.95, 1.0) for seed in seeds
            ]
    elif stage == "interaction":
        # The theta verdict was measured at alpha = 0.95, where the vehicle flies at half
        # its commanded speed. alpha = 1.0 raises it 60%, and transit cost is exactly what
        # the curl trades against -- so the theta optimum may well move. All four cells share
        # one call because the interaction, not either arm, is the quantity of interest.
        yield "theta_x_alpha", [
            (516, f"t{theta:g}_a{alpha:g}", {"theta": theta, "alpha": alpha}, seed)
            for theta in (0.0, 15.0) for alpha in (0.95, 1.0) for seed in seeds
        ]
    elif stage == "speed":
        # Is alpha the knob, or only the speed it unlocks? reference_speed normalises the
        # flow, so raising it asks for the same speed alpha = 1.0 achieves. Reproducing
        # 1.40 m/s at alpha = 0.95 needs a reference near 2.9 -- above the shield's 2.0 m/s
        # cap, which would make alpha the only *flyable* route to that speed.
        yield "refspeed", [
            (516, f"v{speed:g}", {"theta": 15.0, "reference_speed": speed}, seed)
            for speed in (1.8, 2.5, 3.0) for seed in seeds
        ]
    elif stage == "smoothing":
        # Measured at the settled operating point, not the old one. Read achieved speed as
        # the diagnostic: if smoothing helps while speed holds near 1.40 the win is variance
        # reduction; if speed drops, the low-pass is braking it again.
        #
        # One group per window, unlike every other stage. ``smooth_window`` is a *static*
        # field -- it selects the convolution shape -- so levels have different pytree
        # structures and cannot share a batched call. They could not share a compile even
        # sequentially, so batching costs nothing extra here; the groups are kept at equal
        # lane count and identical seeds so the only difference between them is the window.
        for window in (1, 5, 9):
            yield f"smooth_w{window}", [
                (516, f"w{window}",
                 {"theta": 15.0, "alpha": 1.0, "smooth_window": window}, seed)
                for seed in seeds
            ]
    elif stage == "generalization":
        # alpha on a map it was never tuned on. One group per map keeps each comparison
        # inside a single branch.
        for map_seed in (539, 518):
            yield f"gen{map_seed}", [
                (map_seed, f"a{alpha:g}", {"theta": 15.0, "alpha": alpha}, seed)
                for alpha in (0.95, 1.0) for seed in seeds
            ]
    else:
        raise ValueError(f"unknown stage: {stage}")


def lane_config(base, overrides: dict):
    """Apply one lane's overrides to the shared base config."""
    return _apply(base, dict(overrides))


def _append(output: Path, row: dict) -> None:
    """Append one scored row, writing the header on first use.

    The header is checked against ``FIELDS`` every time. ``DictWriter`` emits values in
    ``FIELDS`` order regardless of what the file's header says, so adding a column and
    appending to an existing archive silently shifts every subsequent row by one and
    corrupts the file while looking like it worked. That happened once; this refuses.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
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
        if not (output.exists() and output.stat().st_size):
            writer.writeheader()
        writer.writerow(row)


def _report(lane, row, steps: int) -> None:
    """Print one finished cell."""
    map_seed, variant, _, seed = lane
    print(
        f"map={map_seed} {variant} seed={seed} "
        f"tours={int(row['all_modes_reached'])}+{row['mode_cycles']:.0f} "
        f"speed={row['path_length_m'] / (steps * 0.02):.2f} "
        f"lam={row['temperature_settled_median']:.1f} "
        f"ess={row['ess_settled_median']:.3f}",
        flush=True,
    )


def identity(lane, stage: str, steps: int, hardware: str, execution: str) -> tuple:
    """Resume key for one cell.

    ``hardware`` and ``execution`` are in the key because both select a numerical branch:
    a row from another machine, or from a batch of a different width, cannot satisfy this
    cell. Silently reusing across branches is the error this guards against.
    """
    map_seed, variant, _, seed = lane
    return (stage, str(map_seed), variant, str(seed), str(steps), hardware, execution)


def run_group(label: str, lanes: list, configs: dict, arguments) -> list[tuple]:
    """Run one whole group as a single fused batched call and score its lanes.

    Every lane is an independent closed loop; ``run_batch`` fuses them into one scan. The
    group shares one numerical branch, which is why :func:`groups` keeps each comparison
    inside a single call.
    """
    map_seed = lanes[0][0]
    if map_seed not in configs:
        configs[map_seed] = _grid_config(MAPS[map_seed])
    base, manifest, arrays = configs[map_seed]

    device = select_device(arguments.device)
    lane_configs = [lane_config(base, overrides) for _, _, overrides, _ in lanes]
    stacked = stack_params([jax.device_put(c.controller, device) for c in lane_configs])
    keys = jnp.stack([controller_key(seed) for *_, seed in lanes])
    initial = jnp.asarray(np.asarray(arrays["initial_state"]), dtype=jnp.float32)
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
    for index, lane in enumerate(lanes):
        _, variant, overrides, seed = lane
        row = score_run(
            lane_configs[index], arrays, manifest, seed, arguments.steps,
            positions=paths[index, :, :2], velocities=paths[index, :, 2:4],
            ess_fractions=ess[index], temperatures=temperatures[index],
            wall=wall / len(lanes), device=device.platform,
        )
        controller = lane_configs[index].controller
        row.update({
            "stage": arguments.stage, "variant": variant,
            # Read back from the realised config, not the override dict, so a lane that did
            # not set an axis records what it actually ran with rather than a blank.
            "theta": round(float(np.degrees(np.arctan2(
                float(controller.stein.rotation[1, 0]),
                float(controller.stein.rotation[0, 0])))), 6),
            "alpha": float(controller.mppi.alpha),
            "lam_max": float(controller.mppi.temperature_max),
            "hardware": arguments.hardware, "execution": f"batch{len(lanes)}",
        })
        rows.append((lane, row))
    return rows


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["theta", "alpha", "interaction",
                                          "speed", "smoothing", "generalization"])
    parser.add_argument("--output", type=Path, default=Path("results/uav/gate.csv"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--first-seed", type=int, default=43)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--device", default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--batch", action="store_true",
                        help="Fuse each comparison group into one call: ~6x faster, and a "
                             "different numerical branch than the sequential path.")
    parser.add_argument("--hardware", default=socket.gethostname(),
                        help="Machine tag. Rows from different hardware are different "
                             "numerical branches and must not be pooled.")
    arguments = parser.parse_args()
    seeds = range(arguments.first_seed, arguments.first_seed + arguments.seeds)

    done = set()
    if arguments.output.exists():
        with arguments.output.open(encoding="utf-8", newline="") as stream:
            done = {
                (r["stage"], r["map_seed"], r.get("variant", ""), r["seed"], r["steps"],
                 r.get("hardware", ""), r.get("execution", "sequential"))
                for r in csv.DictReader(stream)
            }

    configs: dict[int, tuple] = {}
    for label, lanes in groups(arguments.stage, seeds):
        execution = f"batch{len(lanes)}" if arguments.batch else "sequential"
        keys = [identity(lane, arguments.stage, arguments.steps, arguments.hardware,
                         execution) for lane in lanes]
        present = [key in done for key in keys]
        if all(present):
            print(f"SKIP [{label}] {len(lanes)} lanes already complete", flush=True)
            continue
        if arguments.batch:
            if any(present):
                # Resume granularity is the group, not the cell: re-running part of a batch
                # would give the new rows a different lane count, so they would sit on a
                # different branch than their own comparison partners. Refuse to mix.
                raise SystemExit(
                    f"[{label}] is partially present ({sum(present)}/{len(lanes)}). A "
                    "batched group must be whole -- delete its rows and re-run the group."
                )
            for lane, row in run_group(label, lanes, configs, arguments):
                _append(arguments.output, row)
                _report(lane, row, arguments.steps)
            continue
        for lane, is_present in zip(lanes, present):
            if is_present:
                print(f"SKIP {lane[1]} seed={lane[3]}", flush=True)
                continue
            map_seed, variant, overrides, seed = lane
            if map_seed not in configs:
                configs[map_seed] = _grid_config(MAPS[map_seed])
            base, manifest, arrays = configs[map_seed]
            config = lane_config(base, overrides)
            row = run_cell(config, arrays, manifest, seed, arguments.steps, {},
                           arguments.device)
            controller = config.controller
            row.update({
                "stage": arguments.stage, "variant": variant,
                "theta": round(float(np.degrees(np.arctan2(
                    float(controller.stein.rotation[1, 0]),
                    float(controller.stein.rotation[0, 0])))), 6),
                "alpha": float(controller.mppi.alpha),
                "lam_max": float(controller.mppi.temperature_max),
                "hardware": arguments.hardware, "execution": "sequential",
            })
            _append(arguments.output, row)
            _report(lane, row, arguments.steps)
    print(f"{arguments.stage} done", flush=True)


if __name__ == "__main__":
    main()

# Ergodic Control MPPI

JAX implementation of flow-matching Model Predictive Path Integral control for
single-robot ergodic coverage of a Gaussian-mixture target density.

## Implementation

At every control step, the controller samples `mppi.K` noisy control sequences
over `mppi.T` steps, integrates the fixed six-state double integrator, and
scores obstacle, map-boundary, MPPI control, and reference-field tracking costs.
The weighted update becomes the next receding-horizon control sequence.

The package separates numerical code from orchestration:

| Path | Responsibility |
|---|---|
| `ergodic_control_mppi/config.py` | One-pass YAML loading and validation |
| `ergodic_control_mppi/parameters.py` | Immutable JAX parameter trees and typed experiment variants |
| `ergodic_control_mppi/models/double_integrator.py` | Batch-compatible 6-state/3-control dynamics |
| `ergodic_control_mppi/mppi/core.py` | Sampling, rollout costs, reference-field tracking, and MPPI update |
| `ergodic_control_mppi/mppi/field.py` | Analytic GMM score, KDE repulsion, service gating, and scalar potential |
| `ergodic_control_mppi/mppi/single.py` | Single-robot closed-loop scan |
| `ergodic_control_mppi/simulation.py` | Device selection, initialization, dispatch, and NumPy results |
| `ergodic_control_mppi/metrics/` | Ergodicity and coordination metrics |
| `ergodic_control_mppi/experiments/` | Experiment runners, baselines, analyses, and reports |
| `ergodic_control_mppi/plotting/` | Simulation and publication figures |

`run_simulation(...)` always returns paths with shape `(steps, 1, 6)` (a
trivial robot axis kept for metric/plot compatibility). Internally,
`run_single(...)` uses `(steps, 6)`. Obstacles have shape `(num_obstacles, 3)`
and may be empty.

## Reference potential field

At every control step, rollout evaluation states are the current position
followed by the first `T - 1` sampled positions. Their temporal increments are
scored with the flow-tracking objective
`sum(-dt * h(z_k) @ delta_z_k + 0.5 * ||delta_z_k||^2)`. The reference velocity
`h(z_k)` is evaluated once on the horizon-wise median of those states and
broadcast across rollouts. Before its speed gauge, the field is the gradient of
the explicit scalar potential in `ergodic_control_mppi/mppi/field.py`: an
analytic target-density score plus KDE repulsion from the fading executed trail
and from the plan itself. Density and recent per-mode service schedule the
tracked speed. The returned surrogate remains the median of all `T` future
sampled positions.

MPPI temperature adapts toward `mppi.ess_target`, and the control-cost
coefficient is recomputed from the current temperature and `mppi.alpha` at
every step.

## Installation

The base installation is CPU-capable and depends on plain `jax`:

```bash
uv sync --python 3.12
```

Optional environments are:

```bash
uv sync --python 3.12 --extra cuda13  # NVIDIA CUDA 13 JAX wheels
```

This follows the official JAX split between plain CPU `jax` and accelerator
extras such as [`jax[cuda13]`](https://docs.jax.dev/en/latest/installation.html).

## Simulation

Run from the repository root:

```bash
uv run python scripts/main.py
uv run python scripts/main.py --config configs/mppi_params.yaml --device cpu --no-plot
```

The CLI accepts `--device auto|cpu|gpu`. `auto` uses a GPU when JAX exposes one
and otherwise falls back to CPU. Controller imports do not query devices,
print, log, or import plotting.

### ROS 2 Jazzy scene

With `DISPLAY` and `XAUTHORITY` exported for the host XWayland session, run:

```bash
docker compose -f docker/ros2/compose.yaml up --build scene
```

This opens one RViz window with the Perlin map, configured target density,
native SO3 drone, and live trail. For headless use, run the image with
`ros2 launch ergodic_control_mppi_ros scene.launch.py rviz:=false`; the launch
also accepts `config:=PATH`.

The model dimensions are fixed and are not configuration keys:

- state `(6,)`: `[px, py, vx, vy, yaw, yaw_rate]`
- control `(3,)`: `[ax, ay, angular_acceleration]`

Active MPPI keys are `mppi.T`, `mppi.K`, `mppi.lambda`, `mppi.alpha`,
`mppi.exploration`, `mppi.smooth_window`, `mppi.ess_target`, `mppi.lam_min`,
`mppi.lam_max`, `mppi.memory_length`, and `mppi.noise.sigma`. Reference-field
keys are `reference.weight_track`, `reference.reference_speed`,
`reference.memory_time`, `reference.memory_balance`, `reference.memory_gain`,
`reference.fill_resolution`, `reference.fine_bandwidth`, `reference.plan_gain`,
`reference.transit_speedup`, `reference.dwell_slowdown`,
`reference.service_floor`, `reference.service_time`,
`reference.deficit_ceiling`, and `reference.release_ratio`.

`mppi.memory_length` defaults to `ceil(3 * reference.memory_time /
model.delta_t)`. `reference.fine_bandwidth` defaults to
`2 * reference.fill_resolution ** 2`; either derived value can be overridden
explicitly.

## Research commands

Experiment YAML lives in `configs/experiments/`. Destructive runners refuse to
replace CSV output unless `--overwrite` is supplied.

```bash
uv run python -m ergodic_control_mppi.experiments.literature --config configs/experiments/literature_comparison.yaml --overwrite
```

Trial CSV rows preserve the established scalar fields, including
`team_ergodic_error`, `pairwise_overlap`, `safety_metric`,
`redundancy_metric`, `R_pair`, `D_min_pair`, and `runtime_ms`. Existing CSVs
are not regenerated automatically.

The UAV research runners expose their campaign, residual-audit, and baseline
options through:

```bash
uv run python scripts/final_ablation.py --help
uv run python scripts/theory_audit.py --help
uv run python -m ergodic_control_mppi.experiments.baselines --help
```

These runners write an adjacent `.manifest.json` containing resolved inputs,
source hashes, and execution metadata. Resume requires matching provenance;
incompatible outputs require a fresh output path or `--overwrite`.
`final_ablation.py plan` reports the cell count without running simulations.
The horizon alternatives are 75, 100, 250, 350, and 500 steps around the
150-step baseline. The theory audit supports a fixed start with
`--inits 4 --start-index 0` (indices 0 through 3).

The frozen T150 bundle contains its configuration, copied maps, audit variants,
and input hashes under `results/uav/T150/`. Its sequential campaign driver
checks stage artifacts and records logs and completion receipts:

```bash
uv run python scripts/run_t150_revision.py plan --bundle results/uav/T150
uv run python scripts/run_t150_revision.py run --bundle results/uav/T150
```

The driver waits for competing GPU compute jobs; `--wait-for-pid PID` also
waits for a specified process before starting. Laptop timing and SITL
validation run separately. The timing module's `--endtoend` measurement uses
synchronized controller calls and transfer of the applied control to NumPy;
`--steps` sets untimed warmup and `--repeats` sets measured calls. Timing
outputs have provenance manifests and require `--overwrite` for replacement.
`scripts/report_figures.py` renders paired ablation effects with 10,000
hierarchical bootstrap replicates and displays timing measurements in a table.
The completed local T150 timing session is stored in
`results/campaign/timing/T150/timing.json`; it used 200 repeats on the
RTX PRO 500 laptop GPU while connected to mains. Its median fused-step and
synchronized applied-control times are 2.365 ms and 2.649 ms, respectively.

## ROS 2 UAV deployment

The same controller flies a fixed-altitude single UAV on the SO3 quadrotor simulator, with
a map adapter, an independent safety guard, and a recorder that pairs every flight with an
ideal offline run on the identical grid, start state and seed. See
[`ros2/ergodic_control_mppi_ros/README.md`](ros2/ergodic_control_mppi_ros/README.md) for the
build, topic map, launch arguments, safety budget, and outputs.

```bash
docker compose -f docker/ros2/compose.yaml build uav
docker compose -f docker/ros2/compose.yaml run --rm uav \
    ros2 launch ergodic_control_mppi_ros uav.launch.py \
        config:=/workspace/configs/uav_profile.yaml run_id:=smoke steps:=200 rviz:=false
```

`configs/uav_profile.yaml` is the deployment configuration, with `T=150` and
`K=250`; `configs/uav_profile_T150.yaml` contains the same frozen profile.
`configs/mppi_params.yaml` is the default offline simulation configuration.
The local selection snapshot is stored under
`results/selection/T350_20260906_local/`, with its scope and file hashes in
`archive.json`. The T150 GPU validation campaign has not yet been completed.

## Validation

```bash
uv run python -m compileall ergodic_control_mppi scripts tests
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests -v
uv lock --check
```

The ROS package has its own tests, which need the container:

```bash
docker compose -f docker/ros2/compose.yaml run --rm uav \
    bash -lc 'cd /ros_ws && colcon test --packages-select ergodic_control_mppi_ros \
              && colcon test-result --verbose'
```

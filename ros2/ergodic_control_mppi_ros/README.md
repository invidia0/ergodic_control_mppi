# ergodic_control_mppi_ros

Fixed-altitude single-UAV deployment of the JAX ergodic MPPI controller on the SO3
quadrotor simulator. The controller itself is unchanged: this package supplies the map
adapter, the online driver, the safety guard, and the recorder, and drives the same
`single_step` the offline runs use.

## Prerequisites

Docker, plus the NVIDIA container toolkit for GPU runs. The CPU path works without it —
the image installs the CUDA plugin, which falls back to CPU when no GPU is present — but
a CPU controller will not hold the 50 Hz deadline at the shipped configuration, so it
stays in hover (see Troubleshooting).

Two GPU facts worth knowing, both learned the hard way:

- **Only the controller may use the GPU.** Every node that imports `ergodic_control_mppi`
  pulls in JAX, and each JAX process preallocates a fraction of VRAM, so a second one
  starves the controller into an allocator death spiral. `uav.launch.py` pins the map
  adapter, guard, recorder and density visualizer to `JAX_PLATFORMS=cpu` for that reason.
- **Preallocate, do not allocate on demand.** With `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  the MPPI step showed ~500 ms allocator stalls: p50 stayed at 5 ms while p99 blew past
  400 ms. The image sets `XLA_PYTHON_CLIENT_MEM_FRACTION=.75` instead.

## Build and run

```bash
docker compose -f docker/ros2/compose.yaml build uav

# 200-step headless smoke run
docker compose -f docker/ros2/compose.yaml run --rm uav \
    ros2 launch ergodic_control_mppi_ros uav.launch.py run_id:=smoke steps:=200 rviz:=false
```

The `uav` service bind-mounts the repository at `/workspace` and the package source at
`/ros_ws/src/ergodic_control_mppi_ros`. The Python package is imported straight from
the mounted source via `PYTHONPATH` (never installed into site-packages, so there is no
stale second copy) and the ROS workspace is built with `--symlink-install`, so **Python edits are live** — only C++
changes (`so3_control`, `mockamap`, the simulator) need `--build`.

Run the package's own tests:

```bash
docker compose -f docker/ros2/compose.yaml run --rm uav \
    bash -lc 'cd /ros_ws && colcon test --packages-select ergodic_control_mppi_ros \
              && colcon test-result --verbose'
```

## Topic map

| Topic | Type | QoS | Publisher | Subscribers |
| --- | --- | --- | --- | --- |
| `/mock_map` | `sensor_msgs/PointCloud2` | transient local, depth 1 | `mockamap` | map adapter |
| `/ergodic/map_visual` | `sensor_msgs/PointCloud2` | transient local, depth 1 | map adapter | RViz |
| `/ergodic/safety_grid` | `nav_msgs/OccupancyGrid` | transient local, depth 1 | map adapter | controller, shield, recorder |
| `/ergodic/map_grid` | `nav_msgs/OccupancyGrid` | transient local, depth 1 | map adapter | recorder |
| `/sim/odom` | `nav_msgs/Odometry` | reliable, depth 1 | simulator | controller, shield, recorder |
| `/ergodic/cmd_raw` | `quadrotor_msgs/PositionCommand` | reliable, depth 1 | controller, 50 Hz | shield, recorder |
| `/ergodic/safety_path` | `nav_msgs/Path` | reliable, depth 1 | controller, 50 Hz | shield, recorder |
| `/ergodic/plan` | `nav_msgs/Path` | reliable, depth 1 | controller, 5 Hz | RViz, recorder |
| `/position_cmd` | `quadrotor_msgs/PositionCommand` | reliable, depth 10 | **safety shield only** | `so3_control`, recorder |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | depth 10 | controller, shield | recorder |

**`/position_cmd` has exactly one publisher: the safety shield.** Nothing else may write to
it. The controller publishes proposals on `/ergodic/cmd_raw`; only what the guard accepts
reaches the vehicle.

The two grids are not interchangeable. `/ergodic/safety_grid` is inflated by the full
1.14 m default budget and is what the planner and the guard reason about; `/ergodic/map_grid` is the
raw physical map and is what `collisions` and `min_clearance_m` are measured against. Scoring
contact against the inflated grid would report a margin breach as a crash.

Two further interlocks back that up:

- The map adapter publishes `/ergodic/safety_grid` **only** if the arming check passes. No
  grid means the controller never activates and the guard holds hover — the absence of the
  topic is the arming interlock.
- `so3_control` holds the current position if `/position_cmd` goes quiet for
  `command_timeout` (0.10 s), so losing the shield stops the vehicle instead of leaving the
  last velocity feedforward applied.

## Launch arguments

`uav.launch.py`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `config` | `/workspace/configs/mppi_params.yaml` | Controller YAML. Use `configs/uav_profile.yaml` for flight. |
| `run_id` | `run` | Output directory name under `output_root`. |
| `output_root` | `/workspace/results/uav` | Where runs and `summary.csv` are written. |
| `profile` | `baseline` | Arm label recorded in the summary row. |
| `steps` | `200` | Control steps to record; the recorder ends the launch at this count. |
| `seed` | `-1` | Controller seed; `-1` uses the config's. |
| `overwrite` | `false` | Allow replacing an existing `run_id`. |
| `device` | `auto` | JAX device selection. |
| `rviz` | `false` | Headless by default. |
| `bag` | `false` | Also run `ros2 bag record` for replay. |
| `map_seed` / `map_fill` | `518` / `0.002` | Perlin map. See "Choosing the map" below. |
| `map_source` | `perlin3d` | `perlin3d` or the preregistered `random_forest` pillars. |
| `obs_num` | `45` | Pillar count; ignored by Perlin. |
| `pillar_min_radius` / `pillar_max_radius` | `0.3` / `0.6` | True pillar radii in metres. |
| `pillar_min_height` / `pillar_max_height` | `2.0` / `3.0` | Pillar height range in metres. |
| `pillar_min_distance` | `1.2` | Minimum pillar-centre spacing in metres. |
| `altitude` | `0.75` | Flight altitude, metres. |
| `start_x` / `start_y` | `-15.57` / `0.42` | Arming position; must be free and connected. |
| `deadline_ms` | `16.0` | Per-step real-time budget. |
| `preflight_steps` | `0` | Stationary planning iterations retained before flight. Pillar tuning uses 200. |
| `visual_height` | `0.04` | Visualization-only obstacle cap at the target-density plane. |

Safety arguments, which set the inflation budget **and** the guard together:

| Argument | Default | Contribution to inflation |
| --- | --- | --- |
| `robot_radius` | `0.30` m | 0.30 — hummingbird is ≈0.55 m tip to tip |
| `clearance` | `0.15` m | 0.15 — discretionary margin for odometry and map error |
| (computed) | — | 0.11 — `0.5·√2·resolution`, cell diagonal plus voxel quantization |
| `tracking_allowance` | `0.05` m | 0.05 — calibrated from measured `pos_p95_m`, see below |
| `max_speed`, `brake_accel`, `reaction_time` | `2.0`, `6.0`, `0.10` | 0.53 — `v·t + v²/(2a)` |
| | | **total 1.14 m ≈ 8 cells at 0.15 m** |

> `max_speed` and `brake_accel` feed both the guard and the map inflation. They are single
> launch arguments for that reason — changing one in only one place would let the planner
> command a path the guard cannot brake out of. Do not duplicate them.

The generic Perlin defaults remain the eight-cell budget above. The successor pillar
tuning runner passes `clearance:=0.05`, giving 1.04 m or seven cells, and records the full
budget in every manifest. `scripts/run_pillar_campaign.sh` pins the archived eight-cell
protocol explicitly.

## Seven-cell pillar tuning

```bash
scripts/run_pillar_tuning.sh --dry-run
scripts/run_pillar_tuning.sh campaign
scripts/run_pillar_tuning.sh prepare
scripts/run_pillar_tuning.sh screen
```

`campaign` runs the resumable gated sequence; each stage can also be launched explicitly.
Every controller cell first runs for 10,000 steps and is archived in `run_cap.csv`. Cells
without a complete three-mode visit–dwell–exit tour stop there; survivors rerun
deterministically to 20,000 steps so `mode_cycles` measures repeat loops. Poll Jeff with
`ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_pillar_tuning.sh`.

Note the budget mixes two kinds of term. The first four are geometry and do not depend on
speed at all; only the stopping distance does, and it grows as `v²`. A static occupancy grid
carries no velocity, so it has to assume the worst case — heading straight at the obstacle at
the cap — and then applies that penalty isotropically, sideways and behind included. That is
why doubling `max_speed` roughly doubles the total budget (1.14 m → 2.19 m at 4 m/s) even
though 0.61 m of it is pure geometry. Splitting the planning grid from a geometry-only
collision grid would remove the coupling, but not before `brake_accel` is measured: with
geometry-only inflation the non-body margin is 0.31 m against a 0.53 m stopping distance, so
a full-speed loss of command would clip the obstacle.

One of the five terms is still an assumption rather than a measurement:

- **`brake_accel`** — kill the controller mid-run and fit the achieved deceleration from
  `/sim/odom`. If it is below 6.0 m/s², lower this argument and re-check arming.

`tracking_allowance` used to be the other one. It is now measured: `pos_p95_m` is 0.015–0.024 m
across the 8000-step runs, so 0.05 m is 2x the observed error. Re-read `pos_p95_m` and redo
this if the vehicle, the gains, or `reference_speed` change.

## Choosing the map

Perlin fill interacts with the inflation: at `fill:=0.1` a 1.29 m dilation closes the whole
workspace and the adapter refuses to arm. The shipped defaults (`map_seed:=518`,
`map_fill:=0.002`, `start_x:=-15.57`, `start_y:=0.42`) come from the sweep below and give
80.8% free space, fully connected, with all three modes reachable.

Two findings worth knowing before re-deriving them:

- **Arming is rare, and lowering the fill alone will not buy it.** As fill falls, perlin
  keeps its highest-noise voxels, and those are fixed by the seed. Under seed 511 one such
  blob sits on the mode at (-12, -4) at *every* fill down to 0.0005, so that seed can never
  arm with this density. Across 511-520 at `map_fill:=0.002` only **two** seeds arm:

  | seed | free | verdict |
  | --- | --- | --- |
  | 511 | 76.5% | mode 1 inside an obstacle |
  | 512 | 84.0% | mode 2 inside an obstacle |
  | 513 | 78.5% | mode 1 inside an obstacle |
  | **514** | 81.8% | **arms** — 97 occupied cells |
  | 515 | 83.9% | mode inside an obstacle |
  | 516 | 86.2% | mode inside an obstacle |
  | 517 | 75.5% | mode inside an obstacle |
  | **518** | 80.8% | **arms** — 153 occupied cells (shipped default) |
  | 519 | 82.7% | mode inside an obstacle |
  | 520 | 78.5% | start cell blocked |

  Free space is a poor predictor: 516 is the emptiest map in the sweep and still fails.
  What decides it is whether a high-noise voxel happens to sit within 1.29 m of a mode
  centre. 518 ships rather than 514 because it carries more obstacle at the same free
  fraction.
- **Sweep with a separate DDS domain per probe.** `/mock_map` is transient-local and every
  `compose run` joins the same bridge network, so a back-to-back sweep can read the previous
  run's map and mis-attribute the result to the current seed. Use
  `ROS_DOMAIN_ID=<n> docker compose run ...` with a distinct `n`.
- **Mode centres can land exactly on a cell boundary.** Mode 2 sits at y = -4, which is
  6.0 m from the grid origin, and 6.0 / 0.15 = 40.0 *exactly*. `OccupancyGrid.info.resolution`
  is float32, so a float64 arming check floors that to row 40 while the controller and guard
  index row 39 — different cells, one free and one blocked. The map adapter rounds the
  resolution to float32 before using it so both sides agree; without that, arming can pass on
  a map where the controller cannot reach the mode.
- **The start must be chosen too.** (-16, 0) is inside the inflated map. On a blocked start
  the adapter prints the nearest free position to use.
- **The map is chosen against the inflation radius, not just the fill.** Modes that are
  clear at 1.29 m get swallowed as the radius grows, so raising `max_speed` needs a fresh
  map check. Measured on the shipped density:

  | inflation | cells | seeds that arm in 511-520 |
  | --- | --- | --- |
  | 1.29 m (`max_speed` 2.0, `tracking_allowance` 0.20) | 9 | 514, **518** |
  | 1.14 m (`max_speed` 2.0, `tracking_allowance` 0.05 — shipped) | 8 | superset of the above |
  | 1.51 m (`max_speed` 3.0) | 11 | not yet derived |
  | 2.19 m (`max_speed` 4.0) | 15 | not yet derived |

  The 1.29 m row is the one actually swept; 1.14 m is strictly less inflation, so every seed
  that armed at 1.29 m still arms there. The higher-speed rows must be swept before any
  higher cap is used — with only 2 of 10 seeds arming at 1.29 m, a wider inflation may need
  a sweep well beyond 520 to find anything at all.

Re-derive with one DDS domain per probe:

```bash
N=0
for SEED in $(seq 511 520); do
    N=$((N + 1))
    ROS_DOMAIN_ID=$N timeout 90 docker compose -f docker/ros2/compose.yaml run --rm uav \
        ros2 launch ergodic_control_mppi_ros uav.launch.py \
            run_id:=arm_$SEED steps:=1 map_seed:=$SEED map_fill:=0.002 rviz:=false
done
```

Take a seed logging `armed:`. A refusal names which of the three causes it is —
start blocked, modes inside obstacles, or a fragmented workspace — and each has a different
fix, so read the second FATAL line rather than just lowering the fill reflexively.

### Pillar campaign

`scripts/run_pillar_campaign.sh` sweeps seeds 511–610 with 45 vertical columns and no
tilted ring obstacles. A map qualifies only when the inflated free space connects the start
to every mode and at least two of the three direct mode-to-mode segments are blocked. The
first three qualifying seeds are fixed before controller execution; the median-free-space
map supplies five bagged SO3 flights. Exact generator settings are stored in every run
manifest, while the public summary CSV schema remains unchanged.

## Outputs

```
results/uav/
  summary.csv          one row per run, schema frozen in deploy/summary.py
  <run_id>/
    manifest.json      config, seeds, grid geometry, versions, git SHA
    arrays.npz         odometry, raw and accepted commands, grid, target, timings, guard
    bag/               only when bag:=true
```

`summary.csv` columns, in order:

```
run_id,profile,mode,seed,map_seed,map_fill,steps,occupancy_mse,fourier_ergodic,
steps_to_threshold,mode_visits,mode_switches,mode_revisits,mode_dwell_median_s,
mode_dwell_total_s,mode_transitions,mode_cycles,first_all_modes_s,in_mode_fraction,
collisions,min_clearance_m,guard_interventions,guard_fraction,guard_duration_s,
max_speed_mps,pos_rmse_m,pos_p95_m,vel_rmse_mps,vel_p95_mps,compile_s,step_p50_ms,
step_p95_ms,step_p99_ms,step_max_ms,deadline_miss_fraction,achieved_rate_hz,
wall_seconds,real_time_factor,run_hash,config_hash,git_sha,seed_controller,jax_version,
ros_distro,device
```

`mode` is `uav` or `ideal`. Pair a flight with its offline twin — same grid, start, seed,
timestep, density and controller config — then render the acceptance report:

```bash
uv run python -m ergodic_control_mppi.experiments.uav_pair --run-dir results/uav/smoke
uv run python -m ergodic_control_mppi.experiments.uav_report \
    --summary results/uav/summary.csv --output results/uav/report.md
```

The report evaluates every acceptance criterion to PASS/FAIL with the number it was judged
on, and applies the screen shortlist rule (zero collisions, guard under 1%, p99 under
16 ms; lowest median occupancy error wins, but the inherited baseline is retained unless
beaten by more than one of its IQRs).

To screen many settings at once, `scripts/run_uav_screen.sh` flies each arm across seeds,
pairs it, and writes the report. It is resumable: an existing `run_id` directory is skipped.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `refusing to arm: ... blocked or disconnected` | Inflation closed the free space | Lower `map_fill`, or sweep `map_seed` |
| `latching hover, no commands will be published` | Startup p99 over `deadline_ms` | Reduce `mppi.T` / `mppi.K` / `stein.memory_time`, or use a GPU |
| `guard engaged: commanded speed over limit` continuously | `stein.reference_speed` above `max_speed` | Lower it to ≤ 2.0 (`configs/uav_profile.yaml` already does) |
| `guard engaged: safety path enters a blocked cell` often | Planner cutting corners it cannot brake out of | Raise `clearance`, or lower `stein.reference_speed` |
| Launch never exits | Recorder never reached `steps` | Check the controller armed; without a grid it publishes nothing |

## Replay

```bash
ros2 launch ergodic_control_mppi_ros uav.launch.py run_id:=paper01 bag:=true rviz:=false
ros2 launch ergodic_control_mppi_ros replay.launch.py bag:=results/uav/paper01/bag
```

`replay.launch.py` starts only bag playback and RViz — no simulator, no controller, no
guard — so a figure made from it is always a view of recorded data.

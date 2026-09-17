# Ergodic MPPI for MULLET drones

The `mullet-deploy` branch of the ergodic MPPI controller: only what the drone runs and
tests. Research code, experiments, figures and simulators live on `main`.

The controller is service-gated potential-gradient MPPI for single-robot ergodic coverage
of a Gaussian-mixture target density, in JAX. On the drone it is one MULLET *mission
framework*:

```text
paddock (operator)  --LoadMission-->  ergodic_mission  --MissionCommand-->  mullet_core  -->  PX4
                    <--MissionStatus--  (this repo)     <--heartbeat, local position--
```

`mullet_core` owns arming, Offboard, takeoff, land and hold, and forwards mission commands
to PX4 only in `MISSION`. This node never talks to `/fmu/in/*`; every refusal ends in
`mullet_core`'s own hold.

## Layout

| Path | Responsibility |
|---|---|
| `ergodic_control_mppi/config.py` | One-pass YAML loading and validation; `gmm_params` |
| `ergodic_control_mppi/parameters.py` | Immutable JAX parameter trees |
| `ergodic_control_mppi/models/double_integrator.py` | Dynamics: state `(6,)`, control `(3,)` |
| `ergodic_control_mppi/mppi/core.py` | Sampling, rollout costs, reference-field tracking, MPPI update |
| `ergodic_control_mppi/mppi/field.py` | GMM score, KDE repulsion, service gate, scalar potential |
| `ergodic_control_mppi/mppi/single.py` | Closed loop: `single_step`, `measured_step`, `run_single` |
| `ergodic_control_mppi/deploy/grid.py` | Inflation budget, reachability and path-blocking queries |
| `ergodic_control_mppi/deploy/mission.py` | Spec compiler, frames, voxels, start gate, command shaping, dry run |
| `ergodic_control_mppi/deploy/store.py` | The drone's mission library |
| `ergodic_control_mppi/simulation.py` | Device selection and offline runs used by the tests |
| `configs/uav_profile.yaml` | Tuned base profile every mission specialises |
| `ros2/ergodic_mission/` | The ROS 2 node |
| `ros2/mullet-interfaces/` | Submodule: the MULLET wire contract (`MissionCommand`, `MissionStatus`, `LoadMission`, `ListMissions`) |
| `docker/mission/` | Image, compose file and `.env.example` |

State `(2d + 2,) = [p(d), v(d), yaw, yaw_rate]`, control `(d + 1,) = [a(d), yaw_accel]`, both in
PX4 local NED; `d = 2` in the planar modes and `d = 3` in the 3D modes.

## Mission spec `ergodic/3`

The paddock's Mission Builder draws this document and sends it through `LoadMission`. The drone
validates it and refuses it with an operator-readable reason; unknown and missing keys are refused.
`tests/fixtures/paddock_3d_spec.json` is a spec the Paddock wrote (its `ergodicDraft` test holds
the same file); the unit suite compiles it in every mode, and the SITL/HIL harness flies it.
`ergodic/1` was planar only; `ergodic/2` added the 3D modes, the ceiling, obstacle heights and
blob altitudes; `ergodic/3` adds blob pitch and roll and `max_vertical_speed_mps`.

```json
{
  "schema": "ergodic/3",
  "mission_id": "field-a",
  "command_mode": "trajectory",
  "duration_s": 300,
  "max_altitude_m": 12,
  "vehicle": {"radius_m": 0.45, "clearance_m": 0.15, "tracking_allowance_m": 0.2,
              "max_speed_mps": 1.5, "max_accel_mps2": 3.0, "brake_accel_mps2": 3.0,
              "reaction_time_s": 0.1, "max_vertical_speed_mps": 0.5},
  "area": [[44.6301, 10.9471], [44.6305, 10.9480], [44.6299, 10.9486]],
  "obstacles": [
    {"type": "circle", "center": [44.6302, 10.9478], "radius_m": 1.5, "height_m": 20},
    {"type": "polygon", "points": [[44.63, 10.9475], [44.6301, 10.9476], [44.63, 10.9477]], "height_m": 2}
  ],
  "density": [
    {"mean": [44.6303, 10.9479, 6], "sigma_major_m": 6, "sigma_minor_m": 2, "sigma_vertical_m": 1,
     "bearing_deg": 30, "pitch_deg": 0, "roll_deg": 0, "weight": 2}
  ]
}
```

- Points are WGS84 `[lat, lon]` degrees; a density `mean` adds its altitude in metres. Sizes
  are metres; bearings are degrees clockwise from north; density weights are relative.
- A density component's body axes (major, minor, up) are turned by `R_yaw(bearing) R_pitch
  R_roll`, as the Paddock draws the ellipsoid: positive `pitch_deg` lifts the major axis,
  positive `roll_deg` the minor axis.
- Altitudes (`max_altitude_m`, obstacle `height_m`, mean altitudes) are metres above the EKF
  origin, which is about the takeoff ground. `max_altitude_m` is the mission ceiling: a start
  above it is refused, and crossing it in flight holds the drone (`FAULT`).
- `command_mode` picks what PX4 tracks and the controller's dimension:
  - `trajectory` (PX4 tracks the planned position with the planned velocity and the MPPI
    acceleration as feedforward; recommended), `velocity` (PX4 tracks the planned velocity with
    the MPPI acceleration as feedforward) and `acceleration` (PX4 tracks the MPPI acceleration)
    are **planar**: the drone flies at the altitude `mullet_core` holds, every obstacle blocks
    every altitude, and each density component contributes its horizontal Gaussian.
  - `trajectory_3d`, `velocity_3d` and `acceleration_3d` fly **x, y and z**
    (`MissionCommand.TRAJECTORY_3D` / `VELOCITY_3D` / `ACCELERATION_3D`). The workspace is a voxel grid: an obstacle blocks only up to its
    `height_m`, so the drone may fly over it; the ground and the ceiling are blocked; each
    density component is a 3D Gaussian (a planar mission flies its horizontal marginal).
- The area polygon is the geofence: everything outside it is an obstacle. It must not cross
  itself, and it may be non-convex: a notch between two modes is blocked space like any obstacle,
  so every rollout pays for crossing it and the controller goes around (`tests/test_nonconvex.py`
  flies a 30 m V notch for 60 s without entering a margin).
- Every obstacle, the area edge, the ground and the ceiling are grown by the stopping-distance
  budget of `deploy/grid.py:inflation_radius`, computed from the `vehicle` block. In 3D the
  ceiling must leave room above that margin twice.
- `max_speed_mps` scales the speed schedule so its corridor peak is the limit; the node also
  clamps every velocity command to it. `max_accel_mps2` is the model's per-axis limit.
  `max_vertical_speed_mps` (3D modes) is enforced on the commands only: the planner's schedule
  is isotropic, so it may plan a faster climb that the drone then flies slower. Keep it within
  `mullet_core`'s `max_vz_ned_mps`, and `max_speed_mps` within `max_vx/vy_ned_mps`, so the
  core clamps stay a backstop.
- Planar grids are 0.15 m and refused above 2 M cells (about 200 m x 200 m). Voxels are 0.5 m
  and refused above the same count (about 100 m x 100 m x 30 m).
- Every density mode must sit in free space and be connected to the heaviest mode. The drone
  may be anywhere when the mission loads; it must be in that connected free space to start.

**Frames.** At load the drone projects every point into its PX4 local NED frame around the
EKF origin (`VehicleLocalPosition.ref_lat/ref_lon`) with PX4's own azimuthal-equidistant map
projection, and the controller runs in NED directly (in 3D, z is down: a mode at 6 m sits at
z = -6). If PX4 moves its EKF origin after the load, the mission is refused or aborted and must
be loaded again.

## The node

`ergodic_mission` takes one parameter that matters, `drone_id`, and derives every other name
from that drone's `/mullet/heartbeat`:

| Direction | Name | Type |
|---|---|---|
| sub | `/mullet/heartbeat` | `mullet_interfaces/DroneHeartbeat` |
| sub | `<px4_fmu_prefix>/fmu/out/vehicle_local_position` | `px4_msgs/VehicleLocalPosition` |
| pub | `<ros_namespace>/command` | `mullet_interfaces/MissionCommand` |
| pub | `/mullet/mission_status` (2 Hz) | `mullet_interfaces/MissionStatus` |
| srv | `<ros_namespace>/mission/ergodic/load` | `mullet_interfaces/LoadMission` |
| srv | `<ros_namespace>/mission/ergodic/list` | `mullet_interfaces/ListMissions` |
| client | heartbeat `hold_service_path` | `std_srvs/Trigger` |

States: `SELF_TEST → EMPTY → PREPARING → READY → RUNNING ⇄ PAUSED → DONE`, or `FAULT`.

Nothing is `READY` unless every layer below passes, and each result is reported in
`MissionStatus.preflight`:

1. **Build.** The image build runs the full unit suite and the node's tests; red means no image.
2. **Boot.** The node runs the fast safety-relevant modules (config, closed loop, grid,
   mission) on the CPU in a subprocess. A failure refuses every load.
3. **Load.** `spec` (validation, projection, every mode free and connected to the heaviest), then
   `device` (compile and time 200 steps on the GPU; if the 99th percentile misses
   `deadline_ms`, 16 ms by default, the same on the CPU; if neither holds, `FAULT`), then
   `dry_run` (fly the compiled mission for `preflight_seconds`, 20 s, in the model from the
   heaviest mode: no safety margin entered, never outside the area, speed in bounds, all
   values finite).
4. **Start.** On every `MISSION` entry: fresh local position (horizontal and vertical), EKF
   origin unchanged, at or below the ceiling, and the drone in a free cell connected to the
   density modes. Otherwise it calls `hold` and goes `FAULT` ("bring the drone inside the
   area...").

In flight, every step: a local position older than 0.1 s publishes nothing (mullet_core
holds after 0.25 s); an EKF origin change, an altitude above the ceiling, or a plan whose next
second comes within the drone's clearance of an obstacle, calls `hold` and goes `FAULT`;
reaching `duration_s` calls `hold` and goes `DONE`. That plan check uses only the hard core of
the margin (robot radius, clearance and cell size, rounded to whole cells): the controller
plans against the whole margin, and its soft obstacle cost may dip into the rest (tracking
allowance and stopping distance), which is what that buffer is for. On the SIH bench the
reference skims the margin and the drone tracks it within 0.2 m, so checking the whole margin
faulted missions that were nowhere near an obstacle.
Leaving `MISSION` pauses the mission with its coverage memory kept, and `start_mission`
resumes it.

**Mission library.** Every spec `LoadMission` accepts is written to `missions_dir`
(`/workspace/missions`, the `docker/mission/missions` volume) as
`<UTC stamp>_<mission_id>.json`, and `ListMissions` returns the newest 50. The paddock shows
them as previous missions to inspect or reload. Nothing is pruned.

Commands: velocity is clamped to `max_speed_mps` (on its 3D norm in the 3D modes), and every acceleration (the feedforward in
velocity mode) is capped along the measured velocity at `SPEED_CAP_GAIN * (max_speed - speed)`,
so the speed settles at the limit and the cap brakes above it. In the 3D modes vz is also
clipped to `max_vertical_speed_mps` and the vertical acceleration gets the same cap. Each PX4 measurement is fed to
the controller once; between measurements it steps on its own prediction.

**Trajectory modes.** PX4 does not follow a command instantly: on the SIH bench
(`tests/sitl/lag_test.py`, fitted with `tests/sitl/fit_lag.py`) horizontal velocity answers after
about 0.2 s, settles with a 0.6 s time constant at about 80 % of the command, and the altitude
sags during horizontal accelerations. The velocity and acceleration modes replan from every
measurement, so that error is never corrected and the drone drifts into the margins the plan
kept clear of. The trajectory modes plan from their own reference instead, the state the
previous step planned, and send its position with velocity and acceleration feedforward, so
PX4's position loop pulls the drone onto the plan. The reference re-anchors on the measurement
only when the drone strays past `tracking_allowance_m`; `mullet_core` bounds how far the
position may lead the drone (`max_xy_position_lead_m`, `max_z_position_lead_m`).

## Build and deploy

Clone with the submodule (`git clone --recursive`, or `git submodule update --init`).

On the drone (Jetson Orin Nano, JetPack 6.2.2), next to MULLET's own container:

```bash
rsync -az --delete --exclude-from=.dockerignore ./ orin:ergodic_control_mppi_v2/
ssh orin 'cd ergodic_control_mppi_v2/docker/mission && cp -n .env.example .env && docker compose up -d --build'
ssh orin 'docker logs -f ergodic-mission'
```

Set `DRONE_ID` and `ROS_DOMAIN_ID` in `docker/mission/.env` to match `mullet_core`. On the Orin
the image builds on MULLET's base (`mullet-ros2-base`, arm64 only), which pins `px4_msgs` and
carries NVIDIA's user-space forward-compatibility libraries: through them JAX's CUDA 13 build
runs on the JetPack 6 driver, with the NVIDIA runtime. On amd64 the Dockerfile builds the same
`px4_msgs` branch itself.

Measured on the Orin Nano at 15 W (`T=150`, `K=250`, 50 Hz): the controller step alone is GPU
p50 7.1 ms / p99 13.6 ms, CPU p50 45 ms / p99 69 ms. A whole flight tick (observation upload,
step, one packed download, plan guard, command shaping) is GPU p50 10.4 ms / p99 15.2 ms
against the 20 ms period; each host-device copy costs about 2.5 ms on the Jetson, which is why
`flight_step` returns everything in one array. The GPU holds the deadline; the CPU fallback does
not, so a drone without a working GPU refuses missions (`FAULT`) rather than flying late.
At MAXN_SUPER (Super firmware, GPU 1020 MHz) the preflight gate's step p99 is 14.6 ms against the
16 ms deadline (15.6-15.8 ms at 15 W). 100 Hz (`T=300`, 8 ms deadline) misses on both devices,
even at MAXN_SUPER (GPU p99 21.5 ms).

### SITL and HIL

The simulator matches the drone: ROS 2 Jazzy and PX4 1.17 (`px4_msgs` `release/1.17`, what the
Pixhawks run). Run `mullet_sitl_sim` headless, run a mission node for one simulated drone, and fly
it with the end-to-end harness (`tests/sitl/sitl_e2e.py`: takeoff; a mission away from the drone
loads, is listed in the library and refuses to start; then `trajectory_3d`, `trajectory` and
`velocity_3d` over the Paddock field scene through preflight, start, a 15 s rate check, hold, resume and DONE; then land). The
3D run needs a `mullet_core` that serves `MissionCommand.VELOCITY_3D`:

```bash
../MULLET/mullet_sitl_sim/run.sh --headless --config config/paddock.yaml --domain 42
# HIL: the node on the Orin, for simulated drone uav_1 (the GPU holds the real 16 ms gate)
ssh orin 'docker run -d --name ergodic-mission-hil --network host --runtime nvidia --memory 4g \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 \
  -e ROS_DOMAIN_ID=42 -e DRONE_ID=uav_1 ergodic-mission:latest'
docker run --rm --network host -e ROS_DOMAIN_ID=42 \
  ergodic-mission:latest python /workspace/tests/sitl/sitl_e2e.py --drone uav_1
```

On a laptop GPU the node misses the 16 ms gate and refuses missions; for a flow-only run start it
with `--ros-args -p drone_id:=uav_1 -p deadline_ms:=150.0`.

### HIL on the bench: Orin + Pixhawk in SIH

Indoors, PX4's simulation-in-hardware stands in for the world: the Pixhawk runs its real
firmware and EKF on simulated sensors (GPS included) and physics, and talks to the Orin over the
real TELEM2 uXRCE-DDS link, so `mullet_core`, the agent and this node all run exactly as they fly.
The obstacles and blobs exist only in the spec.

1. **Props off.** SIH does not drive the motors, but treat the bench as armed hardware.
2. **Pixhawk, in QGroundControl.** Check PX4 is 1.17 and the board lists the SIH airframe.
   *Parameters → Tools → Save to file* (the backup). Then set `SYS_AUTOSTART = 1100` (SIH
   Quadcopter X), `SYS_HITL = 2`, and `SIH_LOC_LAT0` / `SIH_LOC_LON0` / `SIH_LOC_H0` to the lab.
   Leave `UXRCE_DDS_DOM_ID` and the TELEM2 port and baud as they are. Reboot; QGC must show a
   3D fix and the vehicle at that position. If arming is refused for lack of RC, set
   `COM_RC_IN_MODE = 4` (no RC) for the bench session.
3. **Orin.** `mullet-core/docker`: `docker compose up -d --build` (mullet_core with the 3D modes
   and the serial agent). In `config/mullet_core_params.yaml` keep `max_vx/vy_ned_mps` at or
   above the spec's `max_speed_mps` and `max_vz_ned_mps` at or above `max_vertical_speed_mps`
   (the field scene: 1.0 and 0.5), then `docker compose restart mullet_core`. `docker/mission`:
   `docker compose up -d --build` at MAXN_SUPER (`sudo nvpmodel -m 2`).
4. **Optional live view.** Run the Paddock gateway on the laptop on the same `ROS_DOMAIN_ID`.
5. **Fly the harness** on the Orin (the node's drone id is `uav_0` on a single-vehicle drone):

   ```bash
   docker run --rm --network host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -v "$PWD/hil-specs:/out" \
     ergodic-mission:latest python /workspace/tests/sitl/sitl_e2e.py --drone uav_0 \
     --modes trajectory_3d trajectory velocity_3d --save-spec /out
   ```

   `RESULT: PASS` needs, per mode: preflight `READY` on the GPU, about 50 Hz of commands,
   altitude change under the 8 m ceiling in 3D, a steady hold while paused, no pillar touched
   (short pillars only below their top), `DONE`, and a landing.
6. **From the Paddock.** Open a spec from `hil-specs/` (or draw a similar scene), deploy it to
   `uav_0`, wait for `READY`, start, watch, hold and land.
7. **Restore the Pixhawk:** *Parameters → Tools → Load from file* with the backup, then reboot.

## Pulling algorithm updates from `main`

This branch deletes the research tree, so merge controller changes by path and re-run the
suite:

```bash
git checkout main -- ergodic_control_mppi/mppi ergodic_control_mppi/models \
    ergodic_control_mppi/parameters.py ergodic_control_mppi/config.py
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests
```

## Validation

```bash
uv run python -m compileall ergodic_control_mppi tests
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests -v
uv lock --check
```

The node's own tests need ROS 2; they run inside the image build.

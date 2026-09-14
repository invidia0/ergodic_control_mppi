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
| `ergodic_control_mppi/deploy/mission.py` | Spec compiler, frames, start gate, command shaping, dry run |
| `ergodic_control_mppi/simulation.py` | Device selection and offline runs used by the tests |
| `configs/uav_profile.yaml` | Tuned base profile every mission specialises |
| `ros2/ergodic_mission/` | The ROS 2 node |
| `ros2/mullet-interfaces/` | Submodule: the MULLET wire contract (`MissionCommand`, `MissionStatus`, `LoadMission`) |
| `docker/mission/` | Image, compose file and `.env.example` |

State `(6,) = [px, py, vx, vy, yaw, yaw_rate]`, control `(3,) = [ax, ay, yaw_accel]`, both in
PX4 local NED.

## Mission spec `ergodic/1`

The paddock sends one JSON document per mission. The drone validates it and refuses it
with an operator-readable reason; unknown keys are refused.

```json
{
  "schema": "ergodic/1",
  "mission_id": "field-a",
  "command_mode": "velocity",
  "duration_s": 300,
  "vehicle": {"radius_m": 0.45, "clearance_m": 0.15, "tracking_allowance_m": 0.2,
              "max_speed_mps": 1.5, "max_accel_mps2": 3.0, "brake_accel_mps2": 3.0,
              "reaction_time_s": 0.1},
  "area": [[44.6301, 10.9471], [44.6305, 10.9480], [44.6299, 10.9486]],
  "obstacles": [
    {"type": "circle", "center": [44.6302, 10.9478], "radius_m": 1.5},
    {"type": "polygon", "points": [[44.63, 10.9475], [44.6301, 10.9476], [44.63, 10.9477]]}
  ],
  "density": [
    {"mean": [44.6303, 10.9479], "sigma_major_m": 6, "sigma_minor_m": 2, "bearing_deg": 30, "weight": 2}
  ]
}
```

- Points are WGS84 `[lat, lon]` degrees; sizes are metres; bearings are degrees clockwise
  from north; density weights are relative.
- `command_mode` is `velocity` (PX4 tracks the planned velocity with the MPPI acceleration
  as feedforward) or `acceleration` (PX4 tracks the MPPI acceleration).
- The area polygon is the geofence: everything outside it is an obstacle. Obstacles are 2D
  footprints that block every altitude; the drone flies at the altitude it had on entering
  `MISSION`.
- Every obstacle and the area edge are grown by the stopping-distance budget of
  `deploy/grid.py:inflation_radius`, computed from the `vehicle` block.
- `max_speed_mps` scales the speed schedule so its corridor peak is the limit; the node also
  clamps every velocity command to it. `max_accel_mps2` is the model's per-axis limit.
- The grid is 0.15 m; areas above 2 M cells (about 200 m x 200 m) are refused.

**Frames.** At load the drone projects every point into its PX4 local NED frame around the
EKF origin (`VehicleLocalPosition.ref_lat/ref_lon`) with PX4's own azimuthal-equidistant map
projection, and the controller runs in NED directly. If PX4 moves its EKF origin after the
load, the mission is refused or aborted and must be loaded again.

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
| client | heartbeat `hold_service_path` | `std_srvs/Trigger` |

States: `SELF_TEST → EMPTY → PREPARING → READY → RUNNING ⇄ PAUSED → DONE`, or `FAULT`.

Nothing is `READY` unless every layer below passes, and each result is reported in
`MissionStatus.preflight`:

1. **Build.** The image build runs the full unit suite and the node's tests; red means no image.
2. **Boot.** The node runs the fast safety-relevant modules (config, closed loop, grid,
   mission) on the CPU in a subprocess. A failure refuses every load.
3. **Load.** `spec` (validation, projection, reachability of the drone and every mode), then
   `device` (compile and time 200 steps on the GPU; if the 99th percentile misses
   `deadline_ms`, 16 ms by default, the same on the CPU; if neither holds, `FAULT`), then
   `dry_run` (fly the compiled mission for `preflight_seconds`, 20 s, in the model: no safety
   margin entered, never outside the area, speed in bounds, all values finite).
4. **Start.** On every `MISSION` entry: fresh local position, EKF origin unchanged, drone in
   a free cell. Otherwise it calls `hold` and goes `FAULT`.

In flight, every step: a local position older than 0.1 s publishes nothing (mullet_core
holds after 0.25 s); an EKF origin change, or a plan whose next second enters a safety
margin, calls `hold` and goes `FAULT`; reaching `duration_s` calls `hold` and goes `DONE`.
Leaving `MISSION` pauses the mission with its coverage memory kept, and `start_mission`
resumes it.

## Build and deploy

Clone with the submodule (`git clone --recursive`, or `git submodule update --init`).

On the drone (Jetson Orin Nano, JetPack 6.2.2), next to MULLET's own container:

```bash
rsync -az --delete --exclude-from=.dockerignore ./ orin:ergodic_control_mppi_v2/
ssh orin 'cd ergodic_control_mppi_v2/docker/mission && cp -n .env.example .env && docker compose up -d --build'
ssh orin 'docker logs -f ergodic-mission'
```

Set `DRONE_ID` and `ROS_DOMAIN_ID` in `docker/mission/.env` to match `mullet_core`. The image
uses the NVIDIA runtime; on the Orin, JAX's CUDA 13 build runs on the JetPack 6 driver
through NVIDIA's user-space forward-compatibility libraries.

Measured on the Orin Nano at 15 W (`T=150`, `K=250`, 50 Hz): GPU p50 7.1 ms / p99 13.6 ms;
CPU p50 45 ms / p99 69 ms. The GPU holds the deadline; the CPU fallback does not, so a drone
without a working GPU refuses missions (`FAULT`) rather than flying late. 100 Hz (`T=300`)
misses on both.

For SITL on an amd64 host, build against the simulator's PX4 messages:

```bash
cd docker/mission && PX4_MSGS_REF=release/1.16 DRONE_ID=uav_1 ROS_DOMAIN_ID=42 docker compose up --build
```

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

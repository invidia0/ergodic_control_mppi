# Repository Guidelines

## Purpose

This is the `mullet-deploy` branch: the ergodic MPPI controller as a MULLET mission
framework, and nothing else. Research code, experiments, figures and simulators live on
`main`; do not bring them back here. Controller changes come from `main` by path checkout
(see `README.md`) and must pass the full suite.

The drone runs one ROS 2 node, `ros2/ergodic_mission/ergodic_mission/mission_node.py`,
built into `docker/mission/`. It never publishes to PX4: it streams
`mullet_interfaces/MissionCommand` to `mullet_core`, which forwards it only in `MISSION`.

## Package architecture and data flow

- `ergodic_control_mppi/config.py` reads and validates the base YAML into an immutable
  `AppConfig`; unknown keys raise. `gmm_params` precomputes mixture terms for both YAML and
  mission specs.
- `ergodic_control_mppi/parameters.py` defines the JAX-compatible parameter trees.
- `ergodic_control_mppi/models/double_integrator.py` owns the dynamics: state `(6,)`,
  control `(3,)`.
- `ergodic_control_mppi/mppi/field.py` owns the GMM score, KDE repulsion, service gate and
  scalar potential; `mppi/core.py` owns sampling, rollout costs and `mppi_step`;
  `mppi/single.py` owns the closed loop (`single_step`, `measured_step`, `run_single`).
- `ergodic_control_mppi/deploy/grid.py` owns the inflation budget and the reachability and
  path-blocking queries every safety check uses.
- `ergodic_control_mppi/deploy/mission.py` owns the `ergodic/1` spec: WGS84-to-NED
  projection, rasterization, `compile_mission`, and the flight rules the node applies
  (`start_failure`, `command_vectors`, `dry_run_failure`). Keep node logic here when it can
  be tested without ROS.
- `ergodic_control_mppi/simulation.py` owns device selection and the offline run the tests use.

Flow on the drone: `LoadMission` JSON → `compile_mission` → preflight (device gate, model dry
run) → `READY` → `measured_step` per tick while the heartbeat says `MISSION` →
`command_vectors` → `MissionCommand`.

## Shape and frame contracts

- Planar only: state `[px, py, vx, vy, yaw, yaw_rate]`, control `[ax, ay, yaw_accel]`,
  horizon `(T, 3)`, sampled controls `(K, T, 3)`.
- Everything the controller sees is PX4 local NED (x north, y east). Specs are WGS84 and are
  projected once, at load, around the EKF origin with PX4's map projection. Never add a
  second frame.
- Occupancy grid `(H, W)` indexed `[row = y cell, column = x cell]`; origin is the lower NED
  corner. Resolution and origin are float32-exact so every check picks the controller's cells.
- Commands carry NaN z: `mullet_core` holds the altitude.

## Safety contracts

- Nothing becomes `READY` unless the build suite, the boot self-test, and the load preflight
  (spec, device deadline gate, dry run) all pass. Do not add a way to skip any of them.
- Every refusal in flight ends in `mullet_core`'s `hold`; the node never commands a stop
  itself. A stale local position publishes nothing.
- The boot self-test (`SELF_TEST` in the node) must stay fast and CPU-only; heavy numerical
  suites run at image build.

## Coding style

Apply YAGNI and KISS: reuse existing code and the standard library, add only what the current
requirement needs. Keep functions at the narrowest scope that serves their callers. Every
reusable or non-obvious function has a compact Google-style docstring. Keep parameter
dataclasses frozen and JAX-tree-compatible; keep core functions side-effect free under
`jax.jit`, `jax.vmap` and `jax.lax.scan`; keep `jax_enable_x64` off. Device discovery belongs
in `simulation.py` and in the node's preflight only.

## Validation

Use `unittest`; do not add a test framework. Every nontrivial numerical, safety or validation
change needs the smallest runnable regression test.

```bash
uv run python -m compileall ergodic_control_mppi tests
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests -v
uv lock --check
```

The node's tests import ROS 2 and run inside the image build (`docker/mission/Dockerfile`).

## Working conventions

- Report progress on in-progress work as it happens.
- Say where anything runs (laptop, Jetson `orin`, SITL) and with which configuration.
- Keep `README.md` in sync with the tree; describe only what exists now.

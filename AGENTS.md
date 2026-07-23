# Repository Guidelines

## Purpose and supported paths

This repository implements flow-matching MPPI for ergodic coverage with one
shared JAX numerical core and two closed-loop orchestrators:

- `ergodic_control_mppi.mppi.single.run_single` for one robot.
- `ergodic_control_mppi.mppi.multi.run_multi` for synchronous decentralized
  teams. Each robot exchanges its median predicted position trajectory and
  executed-position history; cross-robot Stein interactions use only the
  repulsive kernel-gradient term.

`ergodic_control_mppi.simulation.run_simulation` owns configuration-to-runtime
dispatch, state initialization, device placement, and conversion to NumPy.
`scripts/main.py` is the user-facing simulation command. The value of
`robots.num_robots` in `configs/mppi_params.yaml` selects single or multi mode.

## Package architecture and data flow

- `ergodic_control_mppi/config.py` reads and validates one YAML document into
  an immutable `AppConfig`.
- `ergodic_control_mppi/parameters.py` defines JAX-compatible controller
  parameters and typed experiment variants. Runtime histories, surrogates,
  keys, and controls are loop state, not parameters.
- `ergodic_control_mppi/models/double_integrator.py` owns the fixed 6-state,
  3-control dynamics and batch-compatible clamping.
- `ergodic_control_mppi/mppi/stein.py` owns analytic GMM density/score and RBF
  Stein interactions.
- `ergodic_control_mppi/mppi/core.py` owns functional sampling, rollouts,
  costs, adaptive bandwidth, importance weighting, and `mppi_step`.
- `ergodic_control_mppi/mppi/{single,multi}.py` own the JAX closed-loop scans.
- `ergodic_control_mppi/metrics/` owns experiment-independent metric inputs
  and metric calculation.
- `ergodic_control_mppi/experiments/` owns scenario construction, typed
  controller variants, trial execution, CSV output, BO, ablations,
  sensitivity, and literature comparisons.
- `ergodic_control_mppi/plotting/` owns interactive simulation and publication
  plotting. Controller modules never import plotting.

Runtime flow is YAML -> `load_config` -> `AppConfig` -> `run_simulation` ->
`run_single` or `run_multi` -> normalized `SimulationResult.paths` with shape
`(N, R, 6)`. Experiment runners use the same simulation path and pass a
`TrialData` value to metrics before writing stable flat CSV rows.

## Shape and numerical contracts

- State: `(6,)`, ordered `[px, py, vx, vy, yaw, yaw_rate]`.
- Control: `(3,)`, ordered `[ax, ay, angular_acceleration]`.
- Nominal control horizon: `(T, 3)`; robot batch: `(R, T, 3)`.
- Sampled controls/positions: `(K, T, 3)` and `(K, T, 2)`.
- Single path: `(N, 6)` internally; multi path: `(N, R, 6)`.
- Public simulation and metric paths: `(N, R, 6)`.
- Obstacles: `(num_obstacles, 3)` as `(x, y, radius)`, including `(0, 3)`.
- Shared multi-robot particles per robot: `((R-1) * (T + H), 2)`.

Keep parameter dataclasses frozen and JAX-tree-compatible. Only
shape-controlling integers may be static JAX fields. Use `dataclasses.replace`
for nested variants. Keep core functions side-effect free under `jax.jit`,
`jax.vmap`, and `jax.lax.scan`. Preserve
`jax.config.update("jax_enable_x64", False)`. CPU-only execution is mandatory;
device discovery belongs only in `simulation.py`.

## Configuration and experiment contracts

`configs/` contains YAML only. The fixed model dimensions are not YAML knobs.
Active controller keys are those validated by `load_config`; do not document
or silently accept removed keys such as `mppi.dim_x`, `mppi.dim_u`, or
`stein.weight_pdf`.

Experiment CSV field names are public contracts. Existing CSVs remain
readable, runs do not regenerate old numerical results, and destructive
runners must refuse to replace outputs unless `--overwrite` is supplied.
Optuna is optional and imported only by BO code. Literature baselines reuse
the package dynamics equations.

## Supported commands

Run from the repository root:

```bash
uv run python scripts/main.py [--config PATH] [--device auto|cpu|gpu] [--no-plot]
uv run python -m ergodic_control_mppi.experiments.bo --help
uv run python -m ergodic_control_mppi.experiments.ablations --help
uv run python -m ergodic_control_mppi.experiments.sensitivity --help
uv run python -m ergodic_control_mppi.experiments.literature --help
uv run python -m compileall ergodic_control_mppi scripts tests
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests -v
uv lock --check
```

## Refactor boundaries

- Preserve active YAML semantics, array shapes, CPU fallback, and experiment
  CSV schemas except for explicitly removed inactive fields.
- Preserve the implemented Stein/MPPI objective. Do not retune it or claim a
  theoretical audit against an unavailable paper.
- Correct adaptive-temperature control-cost coupling, initial multi-robot
  surrogates, BO error headers, and equivalent analytic derivatives.
- Do not add legacy import shims, a performance benchmark, or a latency gate.
- Documentation may mention only files, commands, parameters, and outputs that
  exist in the current tree and are wired into runtime.

## Validation

Use `unittest`; do not add a test framework. Every nontrivial numerical or
validation change needs the smallest runnable regression test. Plot tests use
a noninteractive backend and temporary output paths.

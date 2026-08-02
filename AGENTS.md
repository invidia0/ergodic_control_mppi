# Repository Guidelines

## Purpose and supported paths

This repository implements flow-matching MPPI for single-robot ergodic
coverage with one shared JAX numerical core and one closed-loop orchestrator:

- `ergodic_control_mppi.mppi.single.run_single` for the one controlled robot.

`ergodic_control_mppi.simulation.run_simulation` owns configuration-to-runtime
dispatch, state initialization, device placement, and conversion to NumPy.
`scripts/main.py` is the user-facing simulation command.

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
- `ergodic_control_mppi/mppi/single.py` owns the JAX closed-loop scan.
- `ergodic_control_mppi/metrics/` owns experiment-independent metric inputs
  and metric calculation.
- `ergodic_control_mppi/experiments/` owns scenario construction, typed
  controller variants, trial execution, CSV output, BO, ablations,
  sensitivity, and literature comparisons.
- `ergodic_control_mppi/plotting/` owns interactive simulation and publication
  plotting. Controller modules never import plotting.

Runtime flow is YAML -> `load_config` -> `AppConfig` -> `run_simulation` ->
`run_single` -> normalized `SimulationResult.paths` with shape `(N, 1, 6)`.
Experiment runners use the same simulation path and pass a `TrialData` value
to metrics before writing stable flat CSV rows.

## Coding style and helper placement

Apply YAGNI and KISS: reuse existing code and the standard library, add only
what the current requirement needs, and prefer the simplest clear
implementation. Factor code for modularity, obvious usage, and maintainability
without speculative abstractions, configuration, dependencies, or verbose
boilerplate. Keep scripts concise and limited to orchestration and CLI concerns
when reusable behavior belongs in the package.

Keep functions at the narrowest scope that serves their actual callers:

- Keep a function script-local when only that file uses it.
- Put functions shared within one package folder in a folder-local utility
  module.
- Put functions used across different package areas in a repository-level
  utility module.

Do not promote helpers before reuse requires it, and reuse an existing
appropriate module before creating another utility module. Every reusable,
public, or otherwise non-self-explanatory function must have a Google-style
docstring. Include only applicable sections such as `Args`, `Returns`, and
`Raises`; do not add verbose docstrings that merely restate an obvious
signature.

## Shape and numerical contracts

- State: `(6,)`, ordered `[px, py, vx, vy, yaw, yaw_rate]`.
- Control: `(3,)`, ordered `[ax, ay, angular_acceleration]`.
- Nominal control horizon: `(T, 3)`.
- Sampled controls/positions: `(K, T, 3)` and `(K, T, 2)`.
- Single path: `(N, 6)` internally.
- Public simulation and metric paths: `(N, 1, 6)` (trivial robot axis kept for
  metric/plot compatibility).
- Obstacles: `(num_obstacles, 3)` as `(x, y, radius)`, including `(0, 3)`.

Keep parameter dataclasses frozen and JAX-tree-compatible. Only
shape-controlling integers may be static JAX fields. Use `dataclasses.replace`
for nested variants. Keep core functions side-effect free under `jax.jit`,
`jax.vmap`, and `jax.lax.scan`. Preserve
`jax.config.update("jax_enable_x64", False)`. CPU fallback and testing are
mandatory; optional GPU execution is supported. Device discovery belongs only
in `simulation.py`.

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

# "check Jeff" = poll the ablation campaign on the RTX 5090 box that runs it.
# Read-only (greps remote scratch/*.log), safe against a live campaign.
ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_campaign.sh
ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_pillar_tuning.sh
```

## Refactor boundaries

- Preserve active YAML semantics, array shapes, CPU fallback, and experiment
  CSV schemas except for explicitly removed inactive fields.
- Preserve the implemented Stein/MPPI objective. Do not retune it or claim a
  theoretical audit against an unavailable paper.
- Correct adaptive-temperature control-cost coupling, BO error headers, and
  equivalent analytic derivatives.
- Do not add legacy import shims, a performance benchmark, or a latency gate.
- Documentation may mention only files, commands, parameters, and outputs that
  exist in the current tree and are wired into runtime.

## Validation

Use `unittest`; do not add a test framework. Every nontrivial numerical or
validation change needs the smallest runnable regression test. Plot tests use
a noninteractive backend and temporary output paths.

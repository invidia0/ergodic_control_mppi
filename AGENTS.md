# Repository Guidelines

## Purpose and supported paths

This repository implements service-gated potential-gradient MPPI for single-robot ergodic
coverage against a potential-gradient reference field, with one shared JAX
numerical core and one closed-loop orchestrator:

- `ergodic_control_mppi.mppi.single.run_single` for the one controlled robot.

`ergodic_control_mppi.simulation.run_simulation` owns configuration-to-runtime
dispatch, state initialization, device placement, and conversion to NumPy.
`scripts/main.py` is the user-facing simulation command.

## Package architecture and data flow

- `ergodic_control_mppi/config.py` reads and validates one YAML document into
  an immutable `AppConfig`. Unknown keys raise.
- `ergodic_control_mppi/parameters.py` defines JAX-compatible controller
  parameters and typed experiment variants. Runtime histories, surrogates,
  keys, and controls are loop state, not parameters.
- `ergodic_control_mppi/models/double_integrator.py` owns the `d`-aware
  dynamics: state `(2d+2,)`, control `(d+1,)`.
- `ergodic_control_mppi/mppi/field.py` owns the analytic GMM score, the KDE
  repulsion from the fading trail and from the plan, the service gate, and the
  scalar potential every term of the reference field is the gradient of.
- `ergodic_control_mppi/mppi/core.py` owns functional sampling, rollouts,
  costs, importance weighting, and `mppi_step`.
- `ergodic_control_mppi/mppi/single.py` owns the JAX closed-loop scan.
- `ergodic_control_mppi/mppi/replay.py` owns measured-state replay of a
  recorded flight.
- `ergodic_control_mppi/deploy/` owns occupancy-grid adapters shared by ROS 2
  and offline UAV maps.
- `ergodic_control_mppi/metrics/` owns experiment-independent metric inputs
  and metric calculation (`ergodicity`, `discrepancy`, `modes`, `coordination`).
- `ergodic_control_mppi/experiments/` owns scenario construction, typed
  controller variants, trial execution, CSV output, UAV campaigns, literature
  comparison, baselines, `dimension`, `bunny`, and `perlin`.
- `ergodic_control_mppi/plotting/` owns interactive simulation and publication
  plotting, including `dimension.py`. Controller modules never import plotting.

Runtime flow is YAML -> `load_config` -> `AppConfig` -> `run_simulation` ->
`run_single` -> normalized `SimulationResult.paths` with shape `(N, 1, 6)`.
Experiment runners use the same simulation path and pass a `TrialData` value
to metrics before writing stable flat CSV rows. Volumetric studies call
`run_single` through `experiments.dimension.lift`.

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
public, or otherwise non-self-explanatory function must have a compact
Google-style docstring: one-line summary, then only applicable `Args`,
`Returns`, and `Raises`. Do not write design-history essays in public APIs.

## Shape and numerical contracts

- Position dimension `d = gmm.means.shape[-1]`. Every YAML run is planar
  (`d = 2`); `d > 2` exists only through `experiments.dimension.lift`, and the
  planar branch must stay bit-identical.
- State: `(2d + 2,)`, ordered `[p(d), v(d), yaw, yaw_rate]`; planar
  `(6,)` = `[px, py, vx, vy, yaw, yaw_rate]`.
- Control: `(d + 1,)`, ordered `[a(d), angular_acceleration]`; planar `(3,)`.
- Nominal control horizon: `(T, d + 1)`.
- Sampled controls/positions: `(K, T, d + 1)` and `(K, T, d)`.
- Single path: `(N, 2d + 2)` internally.
- Public simulation and metric paths: `(N, 1, 6)` (trivial robot axis kept for
  metric/plot compatibility).
- Obstacles: `(num_obstacles, 3)` as `(x, y, radius)`, including `(0, 3)`, or
  `(num_obstacles, 4)` with the pillar's top height appended (`d >= 3` only).
- Occupancy grid: `(H, W)`, or `(Z, H, W)` for a volume, indexed last axis
  first; `grid_origin` has one entry per grid axis in position order.

Keep parameter dataclasses frozen and JAX-tree-compatible. Only
shape-controlling integers may be static JAX fields. Use `dataclasses.replace`
for nested variants. Keep core functions side-effect free under `jax.jit`,
`jax.vmap`, and `jax.lax.scan`. Preserve
`jax.config.update("jax_enable_x64", False)`. CPU fallback and testing are
mandatory; optional GPU execution is supported. Device discovery belongs only
in `simulation.py`.

## Configuration and experiment contracts

`configs/` contains YAML only. The fixed model dimensions are not YAML knobs.
Active controller keys are those validated by `load_config`. Unknown keys
raise; do not document or silently accept extra knobs.

Experiment CSV field names are public contracts. Existing CSVs remain
readable, runs do not regenerate old numerical results, and destructive
runners must refuse to replace outputs unless `--overwrite` is supplied.
Literature baselines reuse the package dynamics equations.

## Working conventions

- Report progress on in-progress task work as it happens, not only at the end,
  so the user can see what is currently being worked on.
- Audit and report which experiments/scripts are being run — where (local vs.
  remote box), with what config, and to what output path — so the user always
  knows what is currently running or was just run.
- Keep `README.md` synchronized with the actual, current repository state.
  Describe only what exists now.

## Agent rule reconstruction

`.agents/`, `.cursor/`, and `.vscode/` are not tracked. Recreate them as:

- `.agents/readme-auditor.md`: after CLI, path, config-key, or package-layout
  changes, edit `README.md` so every documented path and command exists in the
  tree. No history notes. Process guidance stays in this file.
- `.cursor/rules/latex-compile.mdc`: after `.tex` edits, compile from the
  manuscript directory with `pdflatex` / `bibtex` (not the repository root, not
  `latexmk`).

## Supported commands

Run from the repository root:

```bash
uv run python scripts/main.py [--config PATH] [--device auto|cpu|gpu] [--no-plot]
uv run python -m ergodic_control_mppi.experiments.literature --help
uv run python -m ergodic_control_mppi.experiments.baselines --help
uv run python -m ergodic_control_mppi.experiments.ablation --help
uv run python -m ergodic_control_mppi.experiments.uav_ablation --help
uv run python -m ergodic_control_mppi.experiments.dimension {clutter3d,scaling} --help
uv run python -m ergodic_control_mppi.experiments.bunny {run,figure} --help
uv run python -m ergodic_control_mppi.experiments.perlin {run,figure} --help
uv run python -m compileall ergodic_control_mppi scripts tests
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests -v
uv lock --check

# Poll the ablation campaign on the RTX 5090 box. Read-only.
ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_campaign.sh
ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_pillar_tuning.sh
```

## Refactor boundaries

- Preserve active YAML semantics, array shapes, CPU fallback, and experiment
  CSV schemas except for explicitly removed inactive fields.
- Preserve the implemented reference-field/MPPI objective. Do not retune it.
- Correct adaptive-temperature control-cost coupling and equivalent analytic
  derivatives.
- Do not add import shims, a performance benchmark, or a latency gate.
- Documentation may mention only files, commands, parameters, and outputs that
  exist in the current tree and are wired into runtime.

## Validation

Use `unittest`; do not add a test framework. Every nontrivial numerical or
validation change needs the smallest runnable regression test. Plot tests use
a noninteractive backend and temporary output paths.

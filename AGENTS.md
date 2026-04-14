# Repository Guidelines

## Purpose

This repository implements flow-matching MPPI for ergodic coverage with:

- a single-robot path
- a decentralized multi-robot path aligned with the DARS 2026 paper in [`DARS2026_paper.pdf`](/home/mmantovani/Projects/ergodic_control_mppi/DARS2026_paper.pdf)

The main runnable entrypoint is [`scripts/main.py`](/home/mmantovani/Projects/ergodic_control_mppi/scripts/main.py). Mode selection is driven by `robots.num_robots` in [`configs/mppi_params.yaml`](/home/mmantovani/Projects/ergodic_control_mppi/configs/mppi_params.yaml).

## Architecture

- [`configs/params_loader.py`](/home/mmantovani/Projects/ergodic_control_mppi/configs/params_loader.py) is the configuration entrypoint. It validates YAML and constructs immutable runtime dataclasses.
- [`models/double_integrator.py`](/home/mmantovani/Projects/ergodic_control_mppi/models/double_integrator.py) owns only the system dynamics and control clamping.
- [`mppi/core.py`](/home/mmantovani/Projects/ergodic_control_mppi/mppi/core.py) contains the functional MPPI implementation, rollout scoring, adaptive Stein bandwidth handling, and the flow-matching cost integration.
- [`mppi/stein.py`](/home/mmantovani/Projects/ergodic_control_mppi/mppi/stein.py) is the source of truth for the target density, score function, kernel, and Stein interaction operators.
- [`scripts/main.py`](/home/mmantovani/Projects/ergodic_control_mppi/scripts/main.py) contains the closed-loop simulation paths, device selection, and plotting for both single and multi-robot runs.

## Working Rules

- Keep JAX parameter containers immutable and tree-compatible. Prefer `dataclasses.replace(...)` when changing nested runtime parameters.
- Preserve the existing shape contracts. Core examples:
  - state: `(dim_x,)`
  - control horizon: `(T, dim_u)`
  - sampled trajectories: `(K, T, ...)`
  - robot batches: `(R, ...)`
  - obstacle arrays: `(num_obstacles, 3)`
- Keep `mppi/core.py` and `mppi/stein.py` side-effect free inside `jax.jit`, `jax.vmap`, and `jax.lax.scan` paths.
- Preserve `jax.config.update("jax_enable_x64", False)` unless the task is explicitly about numeric precision.
- Do not assume CUDA is available. Runtime behavior must continue to work on CPU-only machines.
- Preserve the script-friendly import pattern in [`scripts/main.py`](/home/mmantovani/Projects/ergodic_control_mppi/scripts/main.py) unless entrypoints are being reorganized consistently across the repo.

## Validation

There is no automated test suite in the current repository snapshot.

Use the smallest relevant validation step for a change:

```bash
uv run python -m compileall configs models mppi scripts
```

For runtime smoke testing, run:

```bash
uv run python scripts/main.py
```

## Documentation Rules

- `README.md` and `AGENTS.md` must only mention files, commands, and parameters that exist and are wired into the current runtime.
- When documenting the multi-robot controller, describe it in implementation-first terms and then map it to the paper theory.
- Do not present inactive config keys as supported public interface.

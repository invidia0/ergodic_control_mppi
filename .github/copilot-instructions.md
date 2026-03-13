# Project Guidelines

## Code Style
- Follow the existing research-code style: small functional modules, immutable dataclasses for parameter bundles, and explicit `np` / `jnp` naming.
- Prefer minimal, targeted changes over broad refactors. Keep scripts straightforward unless the task is specifically to reorganize them.
- Preserve existing shape conventions and dtype assumptions in JAX code instead of hiding them behind extra abstractions.

## Architecture
- `configs/params_loader.py` is the central configuration entry point. It loads `configs/mppi_params.yaml` into immutable JAX dataclasses used across the repo.
- `models/` contains dynamics only. `models/double_integrator.py` defines the state transition and control clamping for the 6D double-integrator model.
- `mppi/core.py` holds the functional MPPI implementation and trajectory rollout logic. Keep code here side-effect free and JAX-friendly.
- `mppi/stein.py` contains the ergodic objective and Stein gradient utilities. Treat it as the source of truth for GMM-based target density behavior.
- `mppi/mppi_controller.py` is a thin stateful wrapper around the functional core and handles CPU/GPU selection.
- `scripts/` contains experiment entry points, diagnostics, parameter sweeps, evaluation, and plotting. Many scripts also write results back into `scripts/` subfolders.

## Build and Test
- Install dependencies with `uv sync`. If `uv` is unavailable, create a virtual environment and run `pip install .`.
- Run Python scripts from the repository root. Multiple scripts assume `configs/mppi_params.yaml` is reachable via a repo-root relative path and use `os.getcwd()` when saving outputs.
- There is no automated test suite in the repository today. For low-risk validation, run `uv run python -m compileall configs models mppi scripts`.
- After behavior changes, prefer the smallest relevant smoke test instead of full sweeps. Typical entry points are `uv run python scripts/diagnose.py` and the specific script you changed.
- Avoid launching long parameter sweeps or regenerating large result sets unless the task explicitly requires it.

## Conventions
- Preserve `jax.config.update("jax_enable_x64", False)` unless the task is explicitly about numeric precision. Changing it affects both runtime and numerical behavior.
- Keep JAX parameter containers immutable and tree-compatible with `@jax.tree_util.register_dataclass`. When experiments need modified nested parameters, prefer `dataclasses.replace`.
- Maintain the existing array-shape contracts in core logic, such as state `(dim_x,)`, controls `(T, dim_u)`, obstacle arrays `(num_obstacles, 3)`, and GMM/Stein tensors with explicit leading mode axes.
- Avoid adding side effects inside `jax.jit` or `jax.lax.scan` paths in `mppi/core.py` and sweep helpers.
- Do not remove the `sys.path.insert(0, str(Path(__file__).parent.parent))` pattern from scripts unless you convert script entry points consistently across the whole repo.
- Be careful with experiment artifacts under `scripts/test_*`, `scripts/plots`, and `scripts/video_material`. Do not overwrite existing result files unless the task explicitly calls for regeneration.
- Do not assume CUDA is available. Runtime code should continue to work on CPU-only machines.
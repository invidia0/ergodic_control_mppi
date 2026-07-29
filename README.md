# Ergodic Control MPPI

JAX implementation of flow-matching Model Predictive Path Integral control for
single-robot ergodic coverage of a Gaussian-mixture target density.

## Implementation

At every control step, the controller samples `mppi.K` noisy control sequences
over `mppi.T` steps, integrates the fixed six-state double integrator, and
scores obstacle, map-boundary, MPPI control, and Stein flow-matching costs. The
weighted update becomes the next receding-horizon control sequence.

The package separates numerical code from orchestration:

| Path | Responsibility |
|---|---|
| `ergodic_control_mppi/config.py` | One-pass YAML loading and validation |
| `ergodic_control_mppi/parameters.py` | Immutable JAX parameter trees and typed experiment variants |
| `ergodic_control_mppi/models/double_integrator.py` | Batch-compatible 6-state/3-control dynamics |
| `ergodic_control_mppi/mppi/core.py` | Sampling, rollout costs, Stein integration, and MPPI update |
| `ergodic_control_mppi/mppi/stein.py` | Analytic GMM score, RBF gradient, and Stein interactions |
| `ergodic_control_mppi/mppi/single.py` | Single-robot closed-loop scan |
| `ergodic_control_mppi/simulation.py` | Device selection, initialization, dispatch, and NumPy results |
| `ergodic_control_mppi/metrics/` | Ergodicity and coordination metrics |
| `ergodic_control_mppi/experiments/` | Literature comparisons |
| `ergodic_control_mppi/plotting/` | Simulation and publication figures |

`run_simulation(...)` always returns paths with shape `(steps, 1, 6)` (a
trivial robot axis kept for metric/plot compatibility). Internally,
`run_single(...)` uses `(steps, 6)`. Obstacles have shape `(num_obstacles, 3)`
and may be empty.

## Stein flow-matching theory mapping

At every control step, rollout evaluation states are the current position
followed by the first `T - 1` sampled positions. Their temporal increments are
scored with the Section III-C objective
`sum(-dt * h(z_k) @ delta_z_k + 0.5 * ||delta_z_k||^2)`. The Stein flow
`h(z_k)` is evaluated once on the horizon-wise median compression of the
rollout evaluation states and broadcast across rollouts. The returned
surrogate remains the median of all `T` future sampled positions.

The Stein target-density score and kernel gradient use a median-heuristic
bandwidth with a configured floor. This is the implementation's mapping to the
flow-matching theory; no unavailable paper artifact is required or claimed to
have been re-audited.

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

The model dimensions are fixed and are not configuration keys:

- state `(6,)`: `[px, py, vx, vy, yaw, yaw_rate]`
- control `(3,)`: `[ax, ay, angular_acceleration]`

Active tuning keys are `mppi.T`, `mppi.K`, `mppi.lambda`, `mppi.alpha`,
`mppi.exploration`, `mppi.ess_target`, `mppi.lam_min`, `mppi.lam_max`,
`mppi.memory_length`, `mppi.noise.sigma`, `stein.weight_stein`,
`stein.ell_self`, `stein.repulsion_weight`, `stein.repulsion_bandwidth`, and
`stein.theta`. `stein.theta` accepts the inclusive range `[0, 90]` degrees.

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

## Validation

```bash
uv run python -m compileall ergodic_control_mppi scripts tests
JAX_PLATFORMS=cpu uv run python -m unittest discover -s tests -v
uv lock --check
```

# Ergodic Control MPPI

JAX implementation of flow-matching Model Predictive Path Integral control for ergodic coverage with a Gaussian mixture target density. The repository supports both:

- single-robot ergodic exploration
- decentralized multi-robot ergodic exploration following the formulation in [`DARS2026_paper.pdf`](/home/mmantovani/Projects/ergodic_control_mppi/DARS2026_paper.pdf)

The runtime path is selected through `robots.num_robots` in [`configs/mppi_params.yaml`](/home/mmantovani/Projects/ergodic_control_mppi/configs/mppi_params.yaml).

## Overview

At each control step, MPPI samples `K` noisy control sequences over a horizon `T`, rolls them through the double-integrator dynamics, scores them, and updates the nominal control using importance weights.

The rollout cost combines:

- obstacle and out-of-map penalties
- the standard MPPI control-cost cross term
- a flow-matching ergodic term built from Stein interactions with the target density
- an optional target log-density term

The controller runs on CPU and also uses CUDA when JAX can acquire a CUDA device.

## Theory to Code

### Single robot

The single-robot controller follows the paper's flow-matching MPPI idea:

- sampled rollouts induce a predicted spatial trajectory surrogate
- the surrogate is used to build a Stein target flow toward the GMM density
- the robot is also repelled from its own predicted waypoints and recent executed positions
- the resulting flow is added as an MPPI rollout cost

In code, the surrogate is the temporal median of sampled spatial rollouts (returned as `trajectory_surrogate`), and not the arithmetic mean trajectory used in the paper derivation. The rest of the mechanism is the same: the surrogate defines the self particle set and the flow cost biases sampling toward ergodic coverage.

### Multi robot

The multi-robot path in [`scripts/main.py`](/home/mmantovani/Projects/ergodic_control_mppi/scripts/main.py) is decentralized:

- each robot runs its own MPPI loop
- each robot shares a predicted spatial trajectory surrogate with the others
- each robot appends other robots' shared trajectories and recent histories into `cross_particles`
- the self term keeps the single-robot ergodic attraction to the target density
- the cross term keeps only the kernel-gradient repulsion from other robots, matching the paper's decision to drop the foreign score contribution

This maps directly to the paper's multi-robot formulation:

- shared predicted trajectories induce coordination
- repulsion acts across the whole prediction horizon, so robots avoid future overlap rather than only current positions
- recent executed positions are included to reduce revisitation
- the multi-robot objective is encoded entirely in the rollout cost, so the MPPI optimization loop itself does not change

### Adaptive parameters

Two adaptive rules from the paper are present in the implementation:

- self-interaction bandwidth is updated online with the median heuristic, floored by `stein.ell_self` (previously `stein.h`)
- MPPI temperature `lambda` is adapted with an ESS target and clamped by `mppi.lam_min` and `mppi.lam_max`

The cross-robot bandwidth `stein.ell_x` remains fixed. Its square root is the approximate interaction distance scale in workspace units.

## Repository Layout

| Path | Purpose |
|------|---------|
| [`configs/mppi_params.yaml`](/home/mmantovani/Projects/ergodic_control_mppi/configs/mppi_params.yaml) | Main configuration file for simulation, MPPI, map, density, and Stein parameters |
| [`configs/params_loader.py`](/home/mmantovani/Projects/ergodic_control_mppi/configs/params_loader.py) | Strict YAML loader and validator that builds the runtime parameter dataclasses |
| [`models/double_integrator.py`](/home/mmantovani/Projects/ergodic_control_mppi/models/double_integrator.py) | 6D double-integrator dynamics and control clamping |
| [`mppi/core.py`](/home/mmantovani/Projects/ergodic_control_mppi/mppi/core.py) | Functional MPPI rollout, weighting, and Stein-flow cost integration |
| [`mppi/stein.py`](/home/mmantovani/Projects/ergodic_control_mppi/mppi/stein.py) | GMM log-density, score, kernel, and Stein interaction operators |
| [`scripts/main.py`](/home/mmantovani/Projects/ergodic_control_mppi/scripts/main.py) | Single and multi-robot simulation entrypoint plus visualization |
| [`results/`](/home/mmantovani/Projects/ergodic_control_mppi/results) | Saved artifacts from experiments already present in the repository |
| [`DARS2026_paper.pdf`](/home/mmantovani/Projects/ergodic_control_mppi/DARS2026_paper.pdf) | Paper reference for the intended multi-robot formulation |

## Installation

The project uses `uv` for environment management.

```bash
uv sync
```

If you prefer plain `venv` + `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Usage

Run commands from the repository root.

```bash
# Run either the single-robot or multi-robot simulation
# depending on robots.num_robots in configs/mppi_params.yaml
uv run python scripts/main.py

# Syntax check the Python modules
uv run python -m compileall configs models mppi scripts
```

The script loads [`configs/mppi_params.yaml`](/home/mmantovani/Projects/ergodic_control_mppi/configs/mppi_params.yaml), generates random obstacles and initial states, then executes:

- `closed_loop_jit(...)` when `robots.num_robots == 1`
- `multi_robot_closed_loop_jit(...)` when `robots.num_robots > 1`

## Configuration

The main configuration surface is [`configs/mppi_params.yaml`](/home/mmantovani/Projects/ergodic_control_mppi/configs/mppi_params.yaml).

Important keys for switching and tuning modes:

| Key | Meaning |
|-----|---------|
| `robots.num_robots` | `1` for single-robot mode, `>1` for decentralized multi-robot mode |
| `mppi.K` | Number of MPPI rollouts per replanning step |
| `mppi.T` | Receding-horizon length |
| `mppi.lambda` | Initial MPPI temperature |
| `mppi.ess_target` | ESS fraction target for adaptive temperature tuning |
| `mppi.lam_min`, `mppi.lam_max` | Clamp range for adaptive temperature |
| `mppi.history_len` | Length of the executed-position history buffer used in Stein interactions |
| `stein.weight` | Weight of the flow-matching cost |
| `stein.weight_pdf` | Weight of the log-density term |
| `stein.h` | Floor for the adaptive self bandwidth |
| `stein.ell_x` | Fixed cross-robot bandwidth |
| `stein.theta` | Rotation angle defining the curl-augmented geometry |
| `stein.alpha_cross` | Strength of inter-robot repulsion |

The target density is specified under `density`. Obstacles are sampled each run from `map.obstacles`.

## State and Control

| Quantity | Shape | Meaning |
|----------|-------|---------|
| state `x` | `(6,)` | `[px, py, vx, vy, yaw, yaw_rate]` |
| control `u` | `(3,)` | `[ax, ay, alpha]` |

Controls are clamped to the configured acceleration bounds before integration.

## Current implementation notes

- The current multi-robot implementation shares predicted spatial surrogates and history buffers inside the simulation loop in [`scripts/main.py`](/home/mmantovani/Projects/ergodic_control_mppi/scripts/main.py).
- The paper describes pooling mean predicted trajectories. The current code uses the temporal median of sampled spatial rollouts as the exchanged surrogate.

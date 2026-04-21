# Experiment Contract (Phase 0 Freeze)

This file is the source of truth for experiment data exchange across `metrics/`, `experiments/`, `analysis/`, and `plots/`.

## Rollout Log Contract

One rollout log must contain:

- `robot_paths`: array with shape `(steps, robots, state_dim)` where state starts with `(x, y, ...)`
- `target_density_grid`: array with shape `(ny, nx)`, normalized to sum to `1.0`
- `map_x_limits`: `(x_min, x_max)`
- `map_y_limits`: `(y_min, y_max)`
- `obstacle_map`: array with shape `(num_obstacles, 3)` as `(x, y, radius)`
- `safety_radius`: scalar minimum clearance used by safety metric
- `metadata`: free-form scalar metadata (seed, scenario, runtime, parameters)

## Trial Input / Output

Input to metrics:

- robot trajectories over time
- target density grid
- obstacle map
- safety radius
- map limits

Output of one trial:

- one flat dictionary with scalar metrics and scalar metadata
- required metric keys:
  - `team_ergodic_error` (minimize)
  - `pairwise_overlap` (minimize)
  - `safety_metric` (minimize)
  - `redundancy_metric` (minimize)
- optional additional metric keys:
  - `R_pair` (mean close-pair count over time)
  - `D_min_pair` (minimum pairwise spacing over time)

## Sweep Output

Output of one sweep:

- list of flat dictionaries, one row per parameter setting and seed
- persisted incrementally to CSV
- one row must include:
  - scenario id
  - seed
  - controller/sweep parameters
  - scalar metrics
  - runtime

## Swept Parameters (Canonical Names)

- `alpha_cross` (maps to runtime `stein.alpha_cross`)
- `ell_x` (maps to runtime `stein.ell_x`)
- `weight_stein` (maps to runtime `stein.weight_stein`)
- `theta` (maps to runtime Stein rotation angle in degrees)
- `history_window` (maps to runtime `mppi.history_len`)
- `horizon` (maps to runtime `mppi.T`)

## Seed Handling

- Sweep config declares `num_seeds` or explicit `seeds`
- Each trial receives one explicit integer seed
- Seed controls run stochasticity (initial states, random keys)
- Rows always store the exact seed used

## Stability Rule

This contract should remain backward-compatible across phases. New fields can be added, but existing keys and semantics should not be changed silently.

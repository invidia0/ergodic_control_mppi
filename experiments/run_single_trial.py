from __future__ import annotations

import dataclasses
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experiments.scenarios import Scenario
from experiments.trial_types import TrialData
from metrics.aggregate import compute_all_metrics
from scripts.main import closed_loop_jit, multi_robot_closed_loop_jit, _random_state


def _rotation_from_theta(theta_deg: float) -> jnp.ndarray:
    theta = jnp.deg2rad(jnp.asarray(theta_deg, dtype=jnp.float32))
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ],
        dtype=jnp.float32,
    )


def _apply_controller_overrides(params, overrides: dict[str, Any]):
    if not overrides:
        return params

    stein = params.stein
    kwargs = {}

    if "alpha_cross" in overrides:
        kwargs["alpha_cross"] = float(overrides.get("alpha_cross"))
    if "ell_x" in overrides:
        kwargs["ell_x"] = float(overrides.get("ell_x"))
    if "weight_stein" in overrides:
        kwargs["weight_stein"] = float(overrides.get("weight_stein"))
    if "theta" in overrides:
        kwargs["A"] = _rotation_from_theta(float(overrides["theta"]))

    if kwargs:
        stein = dataclasses.replace(stein, **kwargs)
        params = dataclasses.replace(params, stein=stein)

    if "horizon" in overrides:
        horizon = int(overrides["horizon"])
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        params = dataclasses.replace(params, T=horizon)

    if "history_window" in overrides or "history_len" in overrides:
        history_len = int(overrides.get("history_window", overrides.get("history_len")))
        if history_len < 1:
            raise ValueError("history_window/history_len must be >= 1")
        params = dataclasses.replace(
            params,
            history_len=history_len,
            history=jnp.zeros((history_len, 2), dtype=jnp.float32),
        )

    return params


def _run_controller(
    params,
    seed: int,
    team_size: int,
    steps: int,
) -> np.ndarray:
    key = jax.random.PRNGKey(seed)

    if team_size == 1:
        key, subkey = jax.random.split(key)
        x0 = _random_state(subkey, params)
        U0 = jnp.zeros((params.T, params.dim_u), dtype=jnp.float32)
        path, _, _ = closed_loop_jit(params, x0, U0, key, N=steps)
        return np.asarray(path)[:, None, :]  # (steps, 1, state_dim)

    robot_keys = jax.random.split(key, team_size)
    x0_all = jnp.stack([_random_state(robot_keys[i], params) for i in range(team_size)], axis=0)
    U0_all = jnp.zeros((team_size, params.T, params.dim_u), dtype=jnp.float32)
    sim_keys = jax.random.split(jax.random.fold_in(key, 1), team_size)
    paths_all, _, _ = multi_robot_closed_loop_jit(
        params,
        x0_all,
        U0_all,
        sim_keys,
        num_robots=team_size,
        N=steps,
    )
    return np.asarray(paths_all)  # (steps, team_size, state_dim)


def run_single_trial(
    scenario: Scenario,
    controller_config: dict[str, Any],
    seed: int,
    team_size: int | None = None,
    steps: int | None = None,
) -> dict[str, float | int | str]:
    """
    Run one trial and return a flat scalar row.
    """
    team_n = int(team_size if team_size is not None else scenario.run_config.num_robots)
    n_steps = int(steps if steps is not None else scenario.run_config.steps)
    if team_n < 1:
        raise ValueError("team_size must be >= 1")
    if n_steps < 1:
        raise ValueError("steps must be >= 1")

    params = _apply_controller_overrides(scenario.params, controller_config)

    t0 = time.perf_counter()
    robot_paths = _run_controller(params, seed=seed, team_size=team_n, steps=n_steps)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    trial_data = TrialData(
        robot_paths=robot_paths,
        target_density_grid=scenario.target_density_grid,
        map_x_limits=scenario.map_x_limits,
        map_y_limits=scenario.map_y_limits,
        obstacle_map=scenario.obstacle_map,
        safety_radius=scenario.safety_radius,
        metadata={
            "scenario_name": scenario.name,
            "seed": int(seed),
            "team_size": team_n,
            "steps": n_steps,
            "runtime_ms": float(runtime_ms),
        },
    )

    metrics = compute_all_metrics(trial_data)
    row: dict[str, float | int | str] = {
        "scenario_name": scenario.name,
        "seed": int(seed),
        "team_size": team_n,
        "steps": n_steps,
        "runtime_ms": float(runtime_ms),
    }
    for key, value in controller_config.items():
        if isinstance(value, (int, float, str)):
            row[key] = value
    row.update(metrics)
    return row


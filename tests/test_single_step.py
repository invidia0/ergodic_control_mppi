"""Regressions for the shared one-step controller API and the runtime occupancy grid."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.core import stage_cost
from ergodic_control_mppi.mppi.single import (
    initialize_single,
    run_single,
    single_step,
    stationary_step,
)
from ergodic_control_mppi.simulation import controller_key, run_simulation
from tests.helpers import write_small_config


def _small_params(steps: int = 3):
    with tempfile.TemporaryDirectory() as directory:
        config = load_config(write_small_config(Path(directory), steps=steps))
    return config


class SingleStepEquivalenceTest(unittest.TestCase):
    """The scan and an explicit loop over ``single_step`` must agree numerically.

    Not bit-identical: XLA fuses a ``lax.scan`` body differently from eager op-by-op
    execution, which moves the last float32 bits. The tolerance below is roughly three
    orders of magnitude tighter than any real divergence in the update would produce.
    """

    def test_loop_matches_run_single(self):
        steps = 3
        config = _small_params(steps)
        params = config.controller
        state = jnp.array([1.0, -2.0, 0.3, -0.1, 0.5, 0.0], dtype=jnp.float32)
        controls = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
        key = jax.random.PRNGKey(7)

        scanned = run_single(params, state, controls, key, steps=steps)

        carry = initialize_single(params, state, controls, key)
        path = []
        for _ in range(steps):
            carry, result = single_step(params, carry)
            path.append(carry.state)
        looped = jnp.stack(path)

        np.testing.assert_allclose(
            np.asarray(scanned.path), np.asarray(looped), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(scanned.optimal_trajectory),
            np.asarray(result.optimal_trajectory),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(int(carry.step_index), steps)
        self.assertEqual(scanned.ess_fraction.shape, (steps,))
        self.assertEqual(scanned.temperature.shape, (steps,))
        self.assertTrue(np.all(np.isfinite(scanned.ess_fraction)))

    def test_memory_holds_executed_positions_oldest_first(self):
        config = _small_params()
        params = config.controller
        state = jnp.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        carry = initialize_single(
            params, state, jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32), jax.random.PRNGKey(0)
        )
        # Initialization fills the buffer with the start position.
        np.testing.assert_allclose(np.asarray(carry.memory[-1]), np.asarray(state[:2]), atol=0)
        nxt, _ = single_step(params, carry)
        np.testing.assert_allclose(np.asarray(nxt.memory[-1]), np.asarray(nxt.state[:2]), atol=0)
        np.testing.assert_allclose(np.asarray(nxt.memory[:-1]), np.asarray(carry.memory[1:]), atol=0)

    def test_stationary_preflight_retains_planner_carry_only(self):
        config = _small_params()
        params = config.controller
        state = jnp.array([1.0, -2.0, 0.2, -0.1, 0.5, 0.0], dtype=jnp.float32)
        controls = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
        key = jax.random.PRNGKey(4)
        carry = initialize_single(params, state, controls, key)
        held, _ = stationary_step(params, carry, state)

        np.testing.assert_array_equal(np.asarray(held.state), np.asarray(state))
        np.testing.assert_array_equal(
            np.asarray(held.memory),
            np.broadcast_to(np.asarray(state[:2]), held.memory.shape),
        )
        self.assertEqual(int(held.step_index), 0)
        self.assertFalse(np.array_equal(np.asarray(held.key), np.asarray(carry.key)))

        manual = initialize_single(params, state, controls, key)
        manual, _ = stationary_step(params, manual, state)
        manual, _ = stationary_step(params, manual, state)
        manual, _ = single_step(params, manual)
        scanned = run_single(params, state, controls, key, steps=1, preflight_steps=2)
        np.testing.assert_allclose(
            np.asarray(scanned.path[0]), np.asarray(manual.state), rtol=1e-5, atol=1e-6
        )


class OccupancyGridCostTest(unittest.TestCase):
    """An empty grid must change nothing; a populated one must charge blocked cells."""

    def setUp(self):
        self.params = _small_params().controller
        self.states = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )

    def test_empty_grid_is_a_no_op(self):
        self.assertEqual(self.params.workspace.grid.size, 0)
        baseline = stage_cost(self.states, self.params)
        widened = replace(
            self.params,
            workspace=replace(self.params.workspace, grid=jnp.zeros((0, 0), dtype=jnp.float32)),
        )
        np.testing.assert_array_equal(
            np.asarray(baseline), np.asarray(stage_cost(self.states, widened))
        )

    def test_blocked_cell_is_charged_once(self):
        workspace = self.params.workspace
        resolution = 1.0
        origin = jnp.asarray([workspace.x_limits[0], workspace.y_limits[0]], dtype=jnp.float32)
        width = int((float(workspace.x_limits[1]) - float(workspace.x_limits[0])) / resolution)
        height = int((float(workspace.y_limits[1]) - float(workspace.y_limits[0])) / resolution)
        grid = np.zeros((height, width), dtype=np.float32)
        # Block exactly the cell holding (4.0, 2.0), leaving the origin cell free.
        column = int((4.0 - float(workspace.x_limits[0])) / resolution)
        row = int((2.0 - float(workspace.y_limits[0])) / resolution)
        grid[row, column] = 1.0

        gridded = replace(
            self.params,
            workspace=replace(
                workspace,
                grid=jnp.asarray(grid),
                grid_origin=origin,
                grid_resolution=resolution,
            ),
        )
        baseline = np.asarray(stage_cost(self.states, self.params))
        charged = np.asarray(stage_cost(self.states, gridded))
        self.assertAlmostEqual(float(charged[0]), float(baseline[0]), places=5)
        self.assertAlmostEqual(
            float(charged[1]), float(baseline[1]) + float(workspace.obstacle_cost), places=2
        )


class ClosedLoopGridAvoidanceTest(unittest.TestCase):
    """A runtime grid must actually steer the closed loop, not just raise a cost."""

    def test_blocked_region_is_vacated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.yaml"
            data = yaml.safe_load(Path("configs/uav_profile.yaml").read_text(encoding="utf-8"))
            data.update(steps=150, seed=43)
            data["mppi"].update(K=32, T=15)
            data["reference"]["memory_time"] = 1.0
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            config = load_config(path)

        start = np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        resolution = 0.5
        x_limits = (-20.0, 20.0)
        y_limits = (-10.0, 10.0)

        def run(grid):
            workspace = replace(
                config.controller.workspace,
                grid=jnp.asarray(grid),
                grid_origin=jnp.asarray([x_limits[0], y_limits[0]], dtype=jnp.float32),
                grid_resolution=resolution,
            )
            patched = replace(config, controller=replace(config.controller, workspace=workspace))
            return run_simulation(patched, device="cpu", initial_state=start).paths[:, 0, :2]

        free = run(np.zeros((0, 0), dtype=np.float32))
        occupied = lambda path: int(
            ((path[:, 0] >= -10.0) & (path[:, 0] <= -9.0) & (path[:, 1] >= 0.0)).sum()
        )
        self.assertGreater(occupied(free), 0, "the free run must enter the region to be tested")

        width = int((x_limits[1] - x_limits[0]) / resolution)
        height = int((y_limits[1] - y_limits[0]) / resolution)
        grid = np.zeros((height, width), dtype=np.float32)
        grid[
            int((0.0 - y_limits[0]) / resolution) :,
            int((-10.0 - x_limits[0]) / resolution) : int((-9.0 - x_limits[0]) / resolution),
        ] = 1.0
        self.assertEqual(occupied(run(grid)), 0)


class ExplicitInitialStateTest(unittest.TestCase):
    """``run_simulation`` must start exactly where it is told."""

    def test_initial_state_is_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_small_config(Path(directory), steps=2))
        wanted = np.array([2.0, -3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = run_simulation(config, device="cpu", initial_state=wanted)
        np.testing.assert_allclose(result.initial_states[0], wanted, atol=0)

        sampled = run_simulation(config, device="cpu")
        self.assertEqual(sampled.initial_states.shape, (1, 6))
        self.assertEqual(result.ess_fractions.shape, (2,))
        self.assertEqual(result.temperatures.shape, (2,))

    def test_controller_key_is_the_first_stable_split(self):
        expected, _ = jax.random.split(jax.random.PRNGKey(43))
        np.testing.assert_array_equal(np.asarray(controller_key(43)), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()

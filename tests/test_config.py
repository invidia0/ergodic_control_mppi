import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from ergodic_control_mppi.config import load_config
from tests.helpers import write_small_config


class ConfigTest(unittest.TestCase):
    def _mutate(self, callback) -> Path:
        self.temp = tempfile.TemporaryDirectory()
        path = write_small_config(Path(self.temp.name))
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        callback(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.addCleanup(self.temp.cleanup)
        return path

    def test_valid_load_and_deterministic_obstacles(self):
        config_a = load_config("configs/mppi_params.yaml")
        config_b = load_config("configs/mppi_params.yaml")
        self.assertEqual(config_a.controller.stein.self_bandwidth, 1.0)
        np.testing.assert_array_equal(config_a.controller.workspace.obstacles, config_b.controller.workspace.obstacles)

    def test_obstacle_seed_decouples_map_from_run_seed(self):
        """Absent -> old behaviour; present -> the map is pinned across seeds."""
        baseline = load_config(self._mutate(lambda data: data.update(seed=43)))
        moved = load_config(self._mutate(lambda data: data.update(seed=999)))
        pinned = load_config(
            self._mutate(
                lambda data: (
                    data.update(seed=999),
                    data["map"]["obstacles"].update(seed=43),
                )
            )
        )
        # Omitting the key must reproduce the pre-change behaviour exactly.
        np.testing.assert_array_equal(
            baseline.controller.workspace.obstacles, pinned.controller.workspace.obstacles
        )
        self.assertFalse(
            np.array_equal(
                baseline.controller.workspace.obstacles, moved.controller.workspace.obstacles
            )
        )

    def test_zero_obstacles(self):
        config = load_config(self._mutate(lambda data: data["map"]["obstacles"].update(num_obstacles=0)))
        self.assertEqual(config.controller.workspace.obstacles.shape, (0, 3))

    def test_theta_endpoints(self):
        for theta in (0, 90):
            with self.subTest(theta=theta):
                config = load_config(self._mutate(lambda data, theta=theta: data["stein"].update(theta=theta)))
                self.assertTrue(np.all(np.isfinite(config.controller.stein.rotation)))

    def test_non_positive_self_bandwidth_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["stein"].update(ell_self=0)))

    def test_non_integral_shape_value_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["mppi"].update(K=8.5)))

    def test_nonsymmetric_covariance_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["mppi"]["noise"].update(
                sigma=[[1, 2, 0], [0, 1, 0], [0, 0, 1]]
            )))

    def test_non_positive_definite_gmm_covariance_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["density"]["covariances"].__setitem__(
                0, [[1, 2], [2, 1]]
            )))

    def test_memory_time_and_scales_are_derived(self):
        # Assert the derivations, not the tuned values, so tuning the config cannot
        # break this test: decay = exp(-dt/tau), P = ceil(3 tau/dt), h_f = 2 delta_res^2.
        raw = yaml.safe_load(Path("configs/mppi_params.yaml").read_text(encoding="utf-8"))
        tau = raw["stein"]["memory_time"]
        delta_t = raw["model"]["delta_t"]
        config = load_config("configs/mppi_params.yaml")
        stein = config.controller.stein
        self.assertAlmostEqual(stein.memory_decay, np.exp(-delta_t / tau), places=9)
        self.assertEqual(config.controller.mppi.memory_length, math.ceil(3.0 * tau / delta_t))
        self.assertAlmostEqual(stein.fine_bandwidth, 2.0 * raw["stein"]["fill_resolution"] ** 2, places=9)
        # h_c = median trace(Sigma) = 5 for this density, below its 0.25 * min separation cap of 34.
        self.assertAlmostEqual(stein.coarse_bandwidth, 5.0, places=6)
        self.assertGreater(stein.coarse_bandwidth, stein.fine_bandwidth)

    def test_coarse_scale_capped_by_mode_separation(self):
        # min squared separation 2.25 -> cap 0.5625 wins over the 5.0 mode width.
        config = load_config(self._mutate(
            lambda data: data["density"].update(means=[[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
        ))
        self.assertAlmostEqual(config.controller.stein.coarse_bandwidth, 0.5625, places=6)

    def test_retired_memory_knobs_are_rejected(self):
        for retired in ("deficit_gate", "spiral_weight", "memory_mode", "spiral_bandwidth"):
            with self.subTest(key=retired), self.assertRaises(ValueError):
                load_config(self._mutate(
                    lambda data, key=retired: data["stein"].update({key: 1.0})
                ))

    def test_malformed_gmm_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["density"].update(means=[[0, 0, 0]])))


if __name__ == "__main__":
    unittest.main()

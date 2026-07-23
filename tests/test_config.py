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
        np.testing.assert_array_equal(config_a.controller.workspace.obstacles, config_b.controller.workspace.obstacles)

    def test_zero_obstacles(self):
        config = load_config(self._mutate(lambda data: data["map"]["obstacles"].update(num_obstacles=0)))
        self.assertEqual(config.controller.workspace.obstacles.shape, (0, 3))

    def test_theta_endpoints(self):
        for theta in (0, 90):
            with self.subTest(theta=theta):
                config = load_config(self._mutate(lambda data, theta=theta: data["stein"].update(theta=theta)))
                self.assertTrue(np.all(np.isfinite(config.controller.stein.rotation)))

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

    def test_malformed_gmm_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["density"].update(means=[[0, 0, 0]])))


if __name__ == "__main__":
    unittest.main()

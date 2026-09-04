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
        self.assertEqual(config_a.controller.field.fine_bandwidth, 0.94)
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

    def test_withdrawn_stein_knobs_raise(self):
        """Loud, not silently ignored.

        A stale profile that still sets `theta` would otherwise load, fly a different field
        from the one it describes, and be compared against arms it is not comparable with.
        That is the porting hazard this replaces.
        """
        for retired in ("theta", "curl_boost", "ell_self", "attraction", "memory_scales",
                        "coarse_bandwidth", "service_penalty", "plan_repulsion",
                        "flow_iterations", "flow_step", "ensemble_subsample"):
            with self.subTest(key=retired), self.assertRaises(ValueError):
                load_config(self._mutate(
                    lambda data, key=retired: data["reference"].update({key: 1.0})
                ))

    def test_the_stein_section_itself_raises(self):
        def rename(data):
            data["stein"] = data.pop("reference")
        with self.assertRaises(ValueError):
            load_config(self._mutate(rename))

    def test_weight_stein_is_reported_as_renamed(self):
        with self.assertRaises(ValueError) as raised:
            load_config(self._mutate(lambda data: data["reference"].update(weight_stein=1.0)))
        self.assertIn("weight_track", str(raised.exception))

    def test_release_ratio_must_exceed_one(self):
        """sigma* = 1 is release exactly at fair share, which needs an unbounded penalty."""
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["reference"].update(release_ratio=1.0)))
        config = load_config(self._mutate(
            lambda data: data["reference"].update(release_ratio=0.0)))
        self.assertEqual(config.controller.field.release_ratio, 0.0)

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

    def test_memory_and_service_times_are_derived(self):
        # Assert the derivations, not the tuned values, so tuning the config cannot break
        # this test: decay = exp(-dt/tau), P = ceil(3 tau/dt), h = 2 delta_res^2.
        raw = yaml.safe_load(Path("configs/mppi_params.yaml").read_text(encoding="utf-8"))
        tau = raw["reference"]["memory_time"]
        delta_t = raw["model"]["delta_t"]
        config = load_config("configs/mppi_params.yaml")
        field = config.controller.field
        self.assertAlmostEqual(field.memory_decay, np.exp(-delta_t / tau), places=9)
        self.assertEqual(config.controller.mppi.memory_length, math.ceil(3.0 * tau / delta_t))
        # The service window is deliberately independent of the trail: metres of path
        # against a history of visits.
        self.assertAlmostEqual(
            field.service_decay,
            np.exp(-delta_t / raw["reference"].get("service_time", tau)), places=9)

    def test_bandwidth_derives_from_the_fill_resolution(self):
        """h = 2 delta_res^2 puts the kernel peak at the desired track spacing."""
        config = load_config(self._mutate(
            lambda data: (data["reference"].pop("fine_bandwidth", None),
                          data["reference"].update(fill_resolution=0.3))
        ))
        self.assertAlmostEqual(config.controller.field.fine_bandwidth, 0.18, places=9)

    def test_retired_memory_knobs_are_rejected(self):
        for retired in ("deficit_gate", "spiral_weight", "memory_mode", "spiral_bandwidth"):
            with self.subTest(key=retired), self.assertRaises(ValueError):
                load_config(self._mutate(
                    lambda data, key=retired: data["reference"].update({key: 1.0})
                ))

    def test_malformed_gmm_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._mutate(lambda data: data["density"].update(means=[[0, 0, 0]])))


if __name__ == "__main__":
    unittest.main()

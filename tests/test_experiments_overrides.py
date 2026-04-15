import unittest

from configs.params_loader import load_mppi_params
from experiments.run_single_trial import _apply_controller_overrides


class ExperimentsOverridesTest(unittest.TestCase):
    def setUp(self):
        self.params = load_mppi_params("configs/mppi_params.yaml")

    def test_horizon_override_updates_mppi_horizon(self):
        new_horizon = int(self.params.T) + 5
        out = _apply_controller_overrides(self.params, {"horizon": new_horizon})
        self.assertEqual(int(out.T), new_horizon)

    def test_horizon_override_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            _apply_controller_overrides(self.params, {"horizon": 0})

    def test_weight_stein_override_updates_stein_params(self):
        self.assertNotEqual(float(self.params.stein.weight_stein), 123.0)
        out = _apply_controller_overrides(self.params, {"weight_stein": 123.0})
        self.assertEqual(float(out.stein.weight_stein), 123.0)


if __name__ == "__main__":
    unittest.main()

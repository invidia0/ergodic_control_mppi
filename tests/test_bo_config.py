import tempfile
import textwrap
import unittest
from pathlib import Path

from experiments.bo_config import load_bo_config


class BOConfigTest(unittest.TestCase):
    def test_load_bo_config_from_repo_yaml(self):
        cfg = load_bo_config("configs/sweeps/open_multimodal_bo.yaml")
        self.assertGreaterEqual(cfg.n_trials, 1)
        self.assertEqual(cfg.search_seeds, [0, 1])
        self.assertEqual(cfg.reeval_top_n, 10)
        self.assertTrue(cfg.include_baseline)

    def test_duplicate_discrete_values_are_rejected(self):
        content = textwrap.dedent(
            """
            search_seeds: [0]
            reeval_seeds: [0]
            alpha_cross: {low: 1.0, high: 2.0, log: true}
            ell_x: {low: 1.0, high: 2.0, log: true}
            weight_stein: {low: 1.0, high: 2.0, log: true}
            theta: {low: 1.0, high: 2.0, log: false}
            history_window_values: [100, 100]
            horizon_values: [200]
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bo_config(path)


if __name__ == "__main__":
    unittest.main()

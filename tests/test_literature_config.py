import tempfile
import textwrap
import unittest
from pathlib import Path

from experiments.literature_config import load_literature_comparison_config


class LiteratureConfigTest(unittest.TestCase):
    def test_load_repo_literature_config(self):
        cfg = load_literature_comparison_config("configs/sweeps/literature_comparison.yaml")
        self.assertEqual(cfg.team_size, 4)
        self.assertEqual(cfg.steps, 5000)
        self.assertEqual(cfg.seeds, [0, 1, 2])
        self.assertEqual(len(cfg.scenarios), 4)
        self.assertEqual(cfg.methods, ["mppi", "smc", "hedac", "traj_opt", "dec"])

    def test_duplicate_scenario_names_are_rejected(self):
        content = textwrap.dedent(
            """
            seeds: [0]
            scenarios:
              - name: same
                weights: [1.0]
                means: [[0.0, 0.0]]
                covariances: [[[1.0, 0.0], [0.0, 1.0]]]
              - name: same
                weights: [1.0]
                means: [[1.0, 1.0]]
                covariances: [[[1.0, 0.0], [0.0, 1.0]]]
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_literature_comparison_config(path)

    def test_invalid_scenario_weights_are_rejected(self):
        content = textwrap.dedent(
            """
            seeds: [0]
            scenarios:
              - name: bad_weights
                weights: [0.4, 0.4]
                means: [[0.0, 0.0], [1.0, 1.0]]
                covariances:
                  - [[1.0, 0.0], [0.0, 1.0]]
                  - [[1.0, 0.0], [0.0, 1.0]]
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_weights.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_literature_comparison_config(path)


if __name__ == "__main__":
    unittest.main()

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from ergodic_control_mppi.experiments.literature import load_literature_comparison_config
from ergodic_control_mppi.experiments.common import load_scenario
from ergodic_control_mppi.experiments.literature import run_literature_comparison
from ergodic_control_mppi.experiments.literature_methods import (
    make_no_obstacle_scenario,
    run_literature_method,
    sample_initial_states,
)
from tests.helpers import write_small_config


def _write_literature_config(directory: Path, base: Path, methods: str) -> Path:
    output = directory / "literature.yaml"
    output.write_text(textwrap.dedent(f"""
        scenario_name: smoke
        scenario_config_path: {base}
        output_csv_path: {directory / 'rows.csv'}
        summary_csv_path: {directory / 'summary.csv'}
        convergence_csv_path: {directory / 'convergence.csv'}
        plot_output_dir: {directory / 'plots'}
        steps: 3
        seeds: [0]
        d_thresh: 1.0
        methods: [{methods}]
        fourier_order: 2
        desired_speed: 1.0
        tracker_gain: 2.0
        smc_gain: 1.5
        dec_gain: 1.5
        hedac:
          grid_size: 10
          jacobi_iterations: 2
          diffusion_gain: 1.0
          damping: 0.1
          gradient_gain: 2.0
        traj_opt:
          iterations: 1
          learning_rate: 0.05
          control_weight: 0.0001
          smoothness_weight: 0.0001
          bounds_weight: 1.0
        scenarios:
          - name: small
            weights: [1.0]
            means: [[0.0, 0.0]]
            covariances: [[[4.0, 0.0], [0.0, 4.0]]]
    """), encoding="utf-8")
    return output


class LiteratureTest(unittest.TestCase):
    def test_all_methods_return_finite_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = write_small_config(directory, steps=3)
            config = load_literature_comparison_config(
                _write_literature_config(directory, base, "mppi, smc, hedac, traj_opt, dec")
            )
            scenario = make_no_obstacle_scenario(
                load_scenario(str(base)), config.scenarios[0], steps=3
            )
            self.assertEqual(scenario.params.workspace.obstacles.shape, (0, 3))
            initial = sample_initial_states(scenario.params, 1, 0)
            for method in config.methods:
                paths = run_literature_method(method, scenario, initial, steps=3, seed=0, cfg=config)
                self.assertEqual(paths.shape, (3, 1, 6))
                self.assertTrue(np.all(np.isfinite(paths)))

    def test_runner_preserves_required_csv_fields_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = write_small_config(directory, steps=2)
            config_path = _write_literature_config(directory, base, "smc")
            rows, summaries, convergence = run_literature_comparison(str(config_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(convergence)
            self.assertTrue({
                "team_ergodic_error", "pairwise_overlap", "safety_metric",
                "redundancy_metric", "R_pair", "D_min_pair",
            }.issubset(rows[0]))
            with self.assertRaises(FileExistsError):
                run_literature_comparison(str(config_path))


if __name__ == "__main__":
    unittest.main()

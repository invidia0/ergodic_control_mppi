import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from experiments.run_literature_comparison import run_literature_comparison



def _write_small_base_config(tmp_dir: Path) -> Path:
    base_path = Path("configs/mppi_params.yaml")
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg["steps"] = 10
    cfg["robots"]["num_robots"] = 2
    cfg["mppi"]["K"] = 12
    cfg["mppi"]["T"] = 10
    cfg["mppi"]["history_len"] = 12
    cfg["stein"]["weight_stein"] = 150.0

    out = tmp_dir / "small_mppi.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out



def _write_small_literature_config(tmp_dir: Path, base_cfg_path: Path) -> Path:
    content = textwrap.dedent(
        f"""
        scenario_name: unit_literature
        scenario_config_path: {base_cfg_path}

        output_csv_path: {tmp_dir / 'comparison.csv'}
        summary_csv_path: {tmp_dir / 'comparison_summary.csv'}
        convergence_csv_path: {tmp_dir / 'comparison_convergence.csv'}
        plot_output_dir: {tmp_dir / 'plots'}

        team_size: 2
        steps: 10
        seeds: [0]
        d_thresh: 1.0

        methods: [mppi, smc, hedac, traj_opt, dec]

        fourier_order: 2
        desired_speed: 1.0
        tracker_gain: 2.0
        smc_gain: 1.5
        dec_gain: 1.5

        hedac:
          grid_size: 20
          jacobi_iterations: 6
          diffusion_gain: 1.0
          damping: 0.1
          gradient_gain: 4.0

        traj_opt:
          iterations: 3
          learning_rate: 0.05
          control_weight: 1.0e-4
          smoothness_weight: 1.0e-4
          bounds_weight: 1.0

        scenarios:
          - name: unimodal_small
            weights: [1.0]
            means:
              - [0.0, 0.0]
            covariances:
              - [[4.0, 0.0], [0.0, 4.0]]
        """
    )
    out = tmp_dir / "literature_small.yaml"
    out.write_text(content, encoding="utf-8")
    return out


class LiteratureComparisonRunnerTest(unittest.TestCase):
    def test_runner_rows_include_required_scalar_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_cfg = _write_small_base_config(tmp_dir)
            lit_cfg = _write_small_literature_config(tmp_dir, base_cfg)

            rows, summary_rows, conv_rows = run_literature_comparison(str(lit_cfg))

            self.assertEqual(len(rows), 5)
            self.assertEqual(len(summary_rows), 5)
            self.assertGreater(len(conv_rows), 0)

            required = {
                "scenario_name",
                "method_name",
                "seed",
                "team_size",
                "steps",
                "runtime_ms",
                "team_ergodic_error",
                "pairwise_overlap",
                "safety_metric",
                "redundancy_metric",
                "R_pair",
                "D_min_pair",
            }
            for row in rows:
                self.assertTrue(required.issubset(set(row.keys())))
                for k in (
                    "runtime_ms",
                    "team_ergodic_error",
                    "pairwise_overlap",
                    "safety_metric",
                    "redundancy_metric",
                    "R_pair",
                    "D_min_pair",
                ):
                    self.assertIsInstance(float(row[k]), float)


if __name__ == "__main__":
    unittest.main()

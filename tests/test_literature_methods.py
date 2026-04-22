import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiments.literature_config import load_literature_comparison_config
from experiments.literature_methods import (
    make_no_obstacle_scenario,
    run_literature_method,
    sample_initial_states,
)
from experiments.scenarios import load_yaml_scenario



def _write_small_base_config(tmp_dir: Path) -> Path:
    base_path = Path("configs/mppi_params.yaml")
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg["steps"] = 12
    cfg["robots"]["num_robots"] = 2
    cfg["mppi"]["K"] = 16
    cfg["mppi"]["T"] = 12
    cfg["mppi"]["history_len"] = 16
    cfg["stein"]["weight_stein"] = 200.0

    out = tmp_dir / "small_mppi.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out



def _write_small_literature_config(tmp_dir: Path, base_cfg_path: Path) -> Path:
    content = textwrap.dedent(
        f"""
        scenario_name: unit_literature
        scenario_config_path: {base_cfg_path}

        output_csv_path: {tmp_dir / 'out.csv'}
        summary_csv_path: {tmp_dir / 'summary.csv'}
        convergence_csv_path: {tmp_dir / 'convergence.csv'}
        plot_output_dir: {tmp_dir / 'plots'}

        team_size: 2
        steps: 12
        seeds: [0]
        d_thresh: 1.0

        methods: [mppi, smc, hedac, traj_opt, dec]

        fourier_order: 3
        desired_speed: 1.0
        tracker_gain: 2.0
        smc_gain: 1.5
        dec_gain: 1.5

        hedac:
          grid_size: 24
          jacobi_iterations: 8
          diffusion_gain: 1.0
          damping: 0.1
          gradient_gain: 4.0

        traj_opt:
          iterations: 4
          learning_rate: 0.05
          control_weight: 1.0e-4
          smoothness_weight: 5.0e-4
          bounds_weight: 2.0

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


class LiteratureMethodsTest(unittest.TestCase):
    def test_methods_return_valid_team_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_cfg = _write_small_base_config(tmp_dir)
            lit_cfg_path = _write_small_literature_config(tmp_dir, base_cfg)
            lit_cfg = load_literature_comparison_config(lit_cfg_path)

            base_scenario = load_yaml_scenario(str(base_cfg), scenario_name="base")
            scenario = make_no_obstacle_scenario(
                base_scenario,
                lit_cfg.scenarios[0],
                team_size=lit_cfg.team_size,
                steps=lit_cfg.steps,
                grid_shape=tuple(base_scenario.target_density_grid.shape),
            )
            x0_all = sample_initial_states(scenario.params, lit_cfg.team_size, seed=0)

            for method_name in lit_cfg.methods:
                paths = run_literature_method(
                    method_name,
                    scenario,
                    x0_all,
                    steps=lit_cfg.steps,
                    seed=0,
                    cfg=lit_cfg,
                )
                self.assertEqual(paths.shape, (lit_cfg.steps, lit_cfg.team_size, 6))
                self.assertTrue(np.all(np.isfinite(paths)))

                x_min, x_max = scenario.map_x_limits
                y_min, y_max = scenario.map_y_limits
                xy = paths[..., :2]
                in_map = (
                    (xy[..., 0] >= x_min)
                    & (xy[..., 0] <= x_max)
                    & (xy[..., 1] >= y_min)
                    & (xy[..., 1] <= y_max)
                )
                self.assertGreater(float(np.mean(in_map.astype(np.float64))), 0.90)

    def test_no_obstacle_scenario_is_empty_and_runnable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_cfg = _write_small_base_config(tmp_dir)
            lit_cfg_path = _write_small_literature_config(tmp_dir, base_cfg)
            lit_cfg = load_literature_comparison_config(lit_cfg_path)

            base_scenario = load_yaml_scenario(str(base_cfg), scenario_name="base")
            scenario = make_no_obstacle_scenario(
                base_scenario,
                lit_cfg.scenarios[0],
                team_size=lit_cfg.team_size,
                steps=lit_cfg.steps,
                grid_shape=tuple(base_scenario.target_density_grid.shape),
            )

            self.assertEqual(scenario.obstacle_map.shape, (0, 3))
            self.assertEqual(tuple(scenario.params.obstacle_params.xyr.shape), (0, 3))

            x0_all = sample_initial_states(scenario.params, lit_cfg.team_size, seed=1)
            paths = run_literature_method(
                "mppi",
                scenario,
                x0_all,
                steps=6,
                seed=1,
                cfg=lit_cfg,
            )
            self.assertEqual(paths.shape, (6, lit_cfg.team_size, 6))
            self.assertTrue(np.all(np.isfinite(paths)))


if __name__ == "__main__":
    unittest.main()

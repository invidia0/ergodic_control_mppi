"""Regressions for the gated seven-cell pillar tuning campaign."""

import csv
import tempfile
import unittest
from pathlib import Path

from ergodic_control_mppi.experiments.uav_pillar_tuning import (
    APPROACH_ARMS,
    CAP_FIELDS,
    FIELDS,
    SCREEN_ARMS,
    dry_run,
    evaluate,
    run_stage,
    select_split,
    stage_arms,
)


class SelectionTest(unittest.TestCase):
    def test_first_six_split_before_controller_evaluation(self):
        rows = [
            {"map_seed": str(seed), "qualifies": "1", "free_fraction": str(free)}
            for seed, free in zip(range(511, 518), (0.8, 0.6, 0.7, 0.9, 0.5, 0.65, 0.1))
        ]
        result = select_split(rows)
        self.assertEqual(result["development"], [511, 512, 513])
        self.assertEqual(result["holdout"], [514, 515, 516])
        self.assertEqual(result["development_representative"], 513)
        self.assertEqual(result["holdout_representative"], 516)


class ArmTest(unittest.TestCase):
    def test_stage_shapes_and_overrides_are_fixed(self):
        self.assertEqual(
            dry_run(),
            {
                "screen_cap": 30, "screen_full_max": 30,
                "approach_cap": 108, "approach_full_max": 108,
                "holdout_cap": 108, "holdout_full_max": 108,
            },
        )
        approach = stage_arms("approach", "T500", "base")
        self.assertEqual(approach["tau11"]["T"], 500)
        self.assertEqual(approach["tau11"]["lam_max"], 1e5)
        self.assertEqual(approach["tau11"]["memory_time"], 11.0)
        holdout = stage_arms("holdout", "K500", "h2.35")
        self.assertEqual(holdout["winner"]["K"], 500)
        self.assertEqual(holdout["winner"]["fine_bandwidth"], 2.35)
        self.assertEqual(set(stage_arms("screen", "shipped", "base")), set(SCREEN_ARMS))
        self.assertEqual(len(APPROACH_ARMS), 6)

    def test_cap_must_precede_full_run(self):
        with self.assertRaisesRegex(ValueError, "smaller than steps"):
            run_stage(
                Path("missing"), Path("missing.csv"), "screen", "shipped", "base",
                43, 1, 10000, "cpu", set(), Path("cap.csv"), 10000,
            )


class GateTest(unittest.TestCase):
    def _row(self, arm, seed, reached, steps=10000):
        row = dict.fromkeys(CAP_FIELDS, "")
        row.update(
            stage="screen", arm=arm, map_seed=511, seed=seed, steps=steps,
            ess_settled_median=0.3, temperature_cap_fraction=0.0,
            occupancy_mse=1e-6, all_modes_reached=int(reached),
            first_all_modes_s=100.0 if reached else float("nan"),
            mode_cycles=0, collisions=0, accepted=int(reached),
        )
        return row

    def test_screen_requires_a_material_visitation_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self._row("common_cap", seed, False) for seed in range(43, 49)]
            rows += [self._row("T500", seed, seed < 47) for seed in range(43, 49)]
            with (root / "run_cap.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CAP_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            full = [self._row("T500", seed, True, 20000) for seed in range(43, 47)]
            with (root / "offline.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows({field: row[field] for field in FIELDS} for row in full)
            result = evaluate(root)
        self.assertEqual(result["screen_winner"], "T500")
        self.assertTrue(result["screen"]["T500"]["eligible"])
        self.assertFalse(result["screen"]["common_cap"]["eligible"])


if __name__ == "__main__":
    unittest.main()

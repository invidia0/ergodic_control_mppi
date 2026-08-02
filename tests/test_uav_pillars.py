"""Regressions for geometry-only pillar map selection."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ergodic_control_mppi.experiments.uav_pillars import (
    _grid_hash,
    _paired_values,
    parse_probe_log,
    select_maps,
)


class ProbeTest(unittest.TestCase):
    def test_qualification_requires_two_blocked_mode_segments(self):
        row = parse_probe_log(
            "[INFO] armed: 123 occupied cells, free space 67.5%, "
            "all 3 modes reachable from (0, 0), blocked_mode_segments=2",
            511,
        )
        self.assertEqual(row["free_fraction"], 0.675)
        self.assertEqual(row["qualifies"], 1)

    def test_connected_but_easy_map_is_rejected(self):
        row = parse_probe_log(
            "[INFO] armed: 40 occupied cells, free space 90.0%, "
            "all 3 modes reachable from (0, 0), blocked_mode_segments=1",
            512,
        )
        self.assertEqual(row["qualifies"], 0)

    def test_selection_uses_seed_then_median_free_space(self):
        rows = [
            {"map_seed": "514", "qualifies": "1", "free_fraction": "0.60"},
            {"map_seed": "511", "qualifies": "1", "free_fraction": "0.80"},
            {"map_seed": "513", "qualifies": "1", "free_fraction": "0.70"},
            {"map_seed": "512", "qualifies": "0", "free_fraction": "0.65"},
            {"map_seed": "515", "qualifies": "1", "free_fraction": "0.50"},
        ]
        self.assertEqual(select_maps(rows), ([511, 513, 514], 513))

    def test_hash_uses_raw_occupancy_not_inflated_grid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            occupancy = np.asarray([[0, 1], [0, 0]], dtype=np.int8)
            np.savez(first / "arrays.npz", occupancy=occupancy, grid=np.zeros((2, 2)))
            np.savez(second / "arrays.npz", occupancy=occupancy, grid=np.ones((2, 2)))
            self.assertEqual(_grid_hash(first), _grid_hash(second))

    def test_paired_values_match_exact_run_ids(self):
        uav = {"a": {"metric": "2"}, "b": {"metric": "9"}}
        ideal = {"a": {"metric": "4"}, "c": {"metric": "3"}}
        self.assertEqual(_paired_values(uav, ideal, "metric"), [0.5])


if __name__ == "__main__":
    unittest.main()

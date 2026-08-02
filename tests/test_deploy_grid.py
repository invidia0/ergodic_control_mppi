"""Regressions for the deployment occupancy grid: the safety-critical geometry."""

import unittest

import numpy as np

from ergodic_control_mppi.deploy.grid import (
    all_reachable,
    blocked_mode_segments,
    clearance_along,
    entry_cell,
    inflate,
    inflation_radius,
    metric_reachable_mask,
    path_blocked,
    rasterize,
    reachable_from,
    segment_blocked,
    slice_cloud,
    world_to_cell,
)

RESOLUTION = 0.15
X_LIMITS = (-20.0, 20.0)
Y_LIMITS = (-10.0, 10.0)
ORIGIN = (X_LIMITS[0], Y_LIMITS[0])


class InflationBudgetTest(unittest.TestCase):
    """The inflation radius is the safety-critical number; pin it and its terms."""

    def test_shipped_defaults(self):
        radius = inflation_radius(
            robot_radius=0.30,
            clearance=0.15,
            tracking_allowance=0.05,
            max_speed=2.0,
            brake_accel=6.0,
            reaction_time=0.10,
            resolution=RESOLUTION,
        )
        # 0.30 + 0.15 + 0.106 (0.5*sqrt(2)*0.15) + 0.05 + 0.533 (2*0.1 + 4/12)
        self.assertAlmostEqual(radius, 1.1394, places=3)
        self.assertEqual(int(np.ceil(radius / RESOLUTION)), 8)

    def test_pillar_tuning_uses_seven_cells(self):
        radius = inflation_radius(
            robot_radius=0.30,
            clearance=0.05,
            tracking_allowance=0.05,
            max_speed=2.0,
            brake_accel=6.0,
            reaction_time=0.10,
            resolution=RESOLUTION,
        )
        self.assertAlmostEqual(radius, 1.0394, places=3)
        self.assertEqual(int(np.ceil(radius / RESOLUTION)), 7)

    def test_weaker_braking_grows_the_budget(self):
        common = dict(
            robot_radius=0.30,
            clearance=0.15,
            tracking_allowance=0.20,
            max_speed=2.0,
            reaction_time=0.10,
            resolution=RESOLUTION,
        )
        strong = inflation_radius(brake_accel=6.0, **common)
        weak = inflation_radius(brake_accel=3.0, **common)
        self.assertAlmostEqual(weak - strong, 4.0 / 6.0 - 4.0 / 12.0, places=6)

    def test_zero_braking_is_rejected(self):
        with self.assertRaises(ValueError):
            inflation_radius(0.3, 0.15, 0.2, 2.0, 0.0, 0.1, RESOLUTION)


class SliceTest(unittest.TestCase):
    def test_only_the_vertical_band_survives(self):
        points = np.array(
            [
                [1.0, 2.0, 0.75],  # centre of the band
                [3.0, 4.0, 0.95],  # exactly on the upper edge
                [5.0, 6.0, 0.55],  # exactly on the lower edge
                [7.0, 8.0, 1.20],  # above
                [9.0, 9.0, 0.10],  # below
            ]
        )
        kept = slice_cloud(points, 0.75, 0.20)
        self.assertEqual(kept.shape, (3, 2))
        np.testing.assert_allclose(kept[:, 0], [1.0, 3.0, 5.0])

    def test_empty_cloud(self):
        self.assertEqual(slice_cloud(np.zeros((0, 3)), 0.75, 0.2).shape, (0, 2))


class RasterizeTest(unittest.TestCase):
    def test_point_lands_in_the_expected_cell(self):
        grid = rasterize(np.array([[0.0, 0.0]]), X_LIMITS, Y_LIMITS, RESOLUTION)
        self.assertEqual(grid.shape, (134, 267))
        self.assertEqual(grid.sum(), 1)
        row, column = np.argwhere(grid)[0]
        self.assertEqual(column, int(20.0 / RESOLUTION))
        self.assertEqual(row, int(10.0 / RESOLUTION))

    def test_points_outside_the_workspace_are_dropped(self):
        grid = rasterize(np.array([[100.0, 0.0], [0.0, -50.0]]), X_LIMITS, Y_LIMITS, RESOLUTION)
        self.assertEqual(grid.sum(), 0)


class InflateTest(unittest.TestCase):
    def test_disk_shape_and_radius(self):
        grid = np.zeros((21, 21), dtype=bool)
        grid[10, 10] = True
        inflated = inflate(grid, radius=3.0, resolution=1.0)
        # Everything within 3 cells of the centre, nothing at 4.
        self.assertTrue(inflated[10, 13])
        self.assertFalse(inflated[10, 14])
        self.assertTrue(inflated[7, 10])
        self.assertFalse(inflated[10 + 3, 10 + 3])  # distance sqrt(18) > 3

    def test_inflation_never_shrinks_and_clips_at_edges(self):
        grid = np.zeros((5, 5), dtype=bool)
        grid[0, 0] = True
        inflated = inflate(grid, radius=2.0, resolution=1.0)
        self.assertTrue(inflated[grid].all())
        self.assertTrue(inflated[0, 2])
        self.assertFalse(inflated[4, 4])

    def test_zero_radius_is_a_copy(self):
        grid = np.zeros((3, 3), dtype=bool)
        grid[1, 1] = True
        inflated = inflate(grid, radius=0.0, resolution=1.0)
        np.testing.assert_array_equal(inflated, grid)
        inflated[0, 0] = True
        self.assertFalse(grid[0, 0])


class ConnectivityTest(unittest.TestCase):
    def setUp(self):
        # A 10x10 m workspace at 1 m cells, split by a full-height wall at column 5.
        self.grid = np.zeros((10, 10), dtype=bool)
        self.grid[:, 5] = True
        self.origin = (0.0, 0.0)

    def test_walled_off_target_is_rejected(self):
        reachable, flags, _ = all_reachable(
            self.grid, self.origin, 1.0, (1.5, 1.5), np.array([[2.5, 2.5], [8.5, 8.5]])
        )
        self.assertFalse(reachable)
        self.assertTrue(flags[0])
        self.assertFalse(flags[1])

    def test_same_side_targets_are_accepted(self):
        reachable, flags, _ = all_reachable(
            self.grid, self.origin, 1.0, (1.5, 1.5), np.array([[2.5, 2.5], [4.5, 8.5]])
        )
        self.assertTrue(reachable)
        self.assertTrue(flags.all())

    def test_gap_in_the_wall_reconnects(self):
        self.grid[4, 5] = False
        reachable, _, _ = all_reachable(
            self.grid, self.origin, 1.0, (1.5, 1.5), np.array([[8.5, 8.5]])
        )
        self.assertTrue(reachable)

    def test_diagnosis_separates_blocked_from_cut_off(self):
        """A mode inside an obstacle needs a different fix from one merely walled off."""
        grid = self.grid.copy()
        grid[8, 8] = True  # mode 1 sits inside this cell
        _, _, diagnosis = all_reachable(
            grid, self.origin, 1.0, (1.5, 1.5), np.array([[8.5, 8.5], [7.5, 7.5]])
        )
        self.assertFalse(diagnosis["start_blocked"])
        self.assertEqual(diagnosis["blocked_modes"], [0])
        self.assertEqual(diagnosis["disconnected_modes"], [1])
        self.assertGreater(diagnosis["component_cells"], 0)
        self.assertLess(diagnosis["component_cells"], diagnosis["free_cells"])

    def test_diagnosis_reports_a_blocked_start(self):
        _, _, diagnosis = all_reachable(
            self.grid, self.origin, 1.0, (5.5, 5.5), np.array([[2.5, 2.5]])
        )
        self.assertTrue(diagnosis["start_blocked"])
        self.assertEqual(diagnosis["component_cells"], 0)

    def test_start_outside_the_grid_flies_in(self):
        """Starting off the map is legitimate: connectivity is judged from the entry cell."""
        reachable, flags, diagnosis = all_reachable(
            self.grid, self.origin, 1.0, (-6.0, 3.5), np.array([[2.5, 2.5]])
        )
        self.assertTrue(reachable)
        self.assertTrue(flags.all())
        self.assertTrue(diagnosis["start_outside"])
        self.assertFalse(diagnosis["start_blocked"])

    def test_outside_start_still_respects_the_wall(self):
        """Flying in from the left must not grant access to the far side."""
        reachable, flags, _ = all_reachable(
            self.grid, self.origin, 1.0, (-6.0, 3.5), np.array([[8.5, 8.5]])
        )
        self.assertFalse(reachable)
        self.assertFalse(flags[0])

    def test_blocked_start_reaches_nothing(self):
        visited = reachable_from(self.grid, self.origin, 1.0, (5.5, 5.5))
        self.assertFalse(visited.any())

    def test_start_outside_the_grid_seeds_from_its_entry_cell(self):
        visited = reachable_from(self.grid, self.origin, 1.0, (-5.0, 3.5))
        self.assertTrue(visited.any())
        self.assertTrue(visited[3, 0])
        # The wall still separates the two halves.
        self.assertFalse(visited[3, 9])

    def test_fully_blocked_grid_has_no_entry(self):
        self.assertIsNone(entry_cell(np.ones((4, 4), dtype=bool), (0.0, 0.0), 1.0, (-5.0, 1.5)))


class ResolutionPrecisionTest(unittest.TestCase):
    """`OccupancyGrid.info.resolution` is float32, so consumers never see exactly 0.15.

    Checking arming at double precision while the controller indexes at float32 lets the
    two disagree by one cell. That is not hypothetical: it armed a map on which the
    controller saw a target mode buried in an obstacle and never went there.
    """

    def test_double_and_float32_resolution_can_select_different_cells(self):
        exact, stored = 0.15, float(np.float32(0.15))
        self.assertNotEqual(exact, stored)
        # A point far enough out that the accumulated difference crosses a cell boundary.
        point = np.array([12.0, -4.0])
        origin = (-20.0, -10.0)
        a = world_to_cell(point, origin, exact)
        b = world_to_cell(point, origin, stored)
        self.assertFalse(np.array_equal(a, b), "pick a point where the two disagree")

    def test_arming_and_indexing_agree_when_both_use_the_stored_value(self):
        grid = np.zeros((134, 267), dtype=bool)
        stored = float(np.float32(0.15))
        origin = (-20.0, -10.0)
        target = np.array([[12.0, -4.0]])
        row, column = (int(i) for i in world_to_cell(target[0], origin, stored))
        grid[row, column] = True
        # Judged at the stored resolution the mode is blocked, so arming must refuse.
        ok, _, diagnosis = all_reachable(grid, origin, stored, (-16.0, 0.0), target)
        self.assertFalse(ok)
        self.assertEqual(diagnosis["blocked_modes"], [0])


class SegmentTest(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((10, 10), dtype=bool)
        self.grid[5, 5] = True
        self.origin = (0.0, 0.0)

    def test_segment_through_a_blocked_cell(self):
        self.assertTrue(segment_blocked(self.grid, self.origin, 1.0, [4.5, 5.5], [6.5, 5.5]))

    def test_segment_beside_a_blocked_cell(self):
        self.assertFalse(segment_blocked(self.grid, self.origin, 1.0, [4.5, 3.5], [6.5, 3.5]))

    def test_blocked_mode_segments_counts_each_pair_once(self):
        modes = np.array([[4.5, 5.5], [6.5, 5.5], [4.5, 3.5]])
        self.assertEqual(blocked_mode_segments(self.grid, self.origin, 1.0, modes), 1)

    def test_no_tunneling_across_a_thin_obstacle(self):
        """A long jump must not step over the single blocked cell between its ends."""
        self.assertTrue(segment_blocked(self.grid, self.origin, 1.0, [0.5, 5.5], [9.5, 5.5]))

    def test_leaving_the_grid_counts_as_blocked(self):
        self.assertTrue(segment_blocked(self.grid, self.origin, 1.0, [0.5, 0.5], [-5.0, 0.5]))

    def test_empty_path_is_blocked(self):
        self.assertTrue(path_blocked(self.grid, self.origin, 1.0, np.zeros((0, 2))))

    def test_clear_polyline(self):
        path = np.array([[0.5, 0.5], [0.5, 3.5], [3.5, 3.5]])
        self.assertFalse(path_blocked(self.grid, self.origin, 1.0, path))


class ClearanceTest(unittest.TestCase):
    def test_distance_to_the_nearest_occupied_centre(self):
        grid = np.zeros((10, 10), dtype=bool)
        grid[5, 5] = True  # centre at (5.5, 5.5)
        distances = clearance_along(grid, (0.0, 0.0), 1.0, np.array([[5.5, 5.5], [5.5, 8.5]]))
        np.testing.assert_allclose(distances, [0.0, 3.0])

    def test_empty_map_is_infinitely_clear(self):
        distances = clearance_along(
            np.zeros((4, 4), dtype=bool), (0.0, 0.0), 1.0, np.array([[1.0, 1.0]])
        )
        self.assertTrue(np.isinf(distances).all())


class MetricMaskTest(unittest.TestCase):
    def test_mask_excludes_the_far_side_of_a_wall(self):
        grid = np.zeros((20, 20), dtype=bool)
        grid[:, 10] = True
        mask = metric_reachable_mask(
            grid, (0.0, 0.0), 1.0, (2.5, 2.5), (0.0, 20.0), (0.0, 20.0), (16, 16)
        )
        self.assertEqual(mask.shape, (16, 16))
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, -1].any())


if __name__ == "__main__":
    unittest.main()

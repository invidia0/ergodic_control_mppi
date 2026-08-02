"""Regressions for the map adapter's slice/inflate/publish path."""

import unittest

import numpy as np

from ergodic_control_mppi_ros.map_adapter import (
    build_safety_grid,
    clip_visual_points,
    to_message,
)

X_LIMITS = (-10.0, 10.0)
Y_LIMITS = (-5.0, 5.0)
RESOLUTION = 0.5


class BuildSafetyGridTest(unittest.TestCase):
    def test_only_the_flight_band_becomes_occupancy(self):
        points = np.array(
            [
                [0.0, 0.0, 0.75],  # in band
                [4.0, 2.0, 3.50],  # far above, must be ignored
                [-4.0, -2.0, 0.05],  # below, must be ignored
            ]
        )
        occupancy, _ = build_safety_grid(
            points, X_LIMITS, Y_LIMITS, 0.75, 0.20, RESOLUTION, radius=0.0
        )
        self.assertEqual(occupancy.sum(), 1)
        row, column = np.argwhere(occupancy)[0]
        self.assertEqual(column, int(10.0 / RESOLUTION))
        self.assertEqual(row, int(5.0 / RESOLUTION))

    def test_inflation_grows_the_blocked_area(self):
        points = np.array([[0.0, 0.0, 0.75]])
        occupancy, inflated = build_safety_grid(
            points, X_LIMITS, Y_LIMITS, 0.75, 0.20, RESOLUTION, radius=1.29
        )
        self.assertEqual(occupancy.sum(), 1)
        self.assertGreater(inflated.sum(), occupancy.sum())
        # 1.29 m at 0.5 m cells is ceil = 3 cells. The discrete disk dx^2+dy^2 <= 9 has
        # 7 + 2*5 + 2*5 + 2*1 = 29 cells, counting by row offset dy = 0, +-1, +-2, +-3.
        self.assertEqual(inflated.sum(), 29)
        self.assertTrue(inflated[occupancy].all())

    def test_empty_cloud_leaves_the_workspace_free(self):
        _, inflated = build_safety_grid(
            np.zeros((0, 3)), X_LIMITS, Y_LIMITS, 0.75, 0.20, RESOLUTION, radius=1.29
        )
        self.assertFalse(inflated.any())


class MessageTest(unittest.TestCase):
    def test_visual_cloud_is_capped_without_mutating_raw_points(self):
        points = np.array([[1.0, 2.0, -0.1], [3.0, 4.0, 2.5]], dtype=np.float32)
        clipped = clip_visual_points(points, 0.04)
        np.testing.assert_allclose(clipped[:, 2], [-0.1, 0.04])
        np.testing.assert_allclose(points[:, 2], [-0.1, 2.5])

    def test_occupancy_grid_encoding_and_geometry(self):
        grid = np.zeros((4, 6), dtype=bool)
        grid[1, 2] = True
        message = to_message(grid, (-1.0, -2.0), 0.25)

        self.assertEqual(message.header.frame_id, "world")
        self.assertEqual(message.info.width, 6)
        self.assertEqual(message.info.height, 4)
        self.assertAlmostEqual(message.info.resolution, 0.25)
        self.assertAlmostEqual(message.info.origin.position.x, -1.0)
        self.assertAlmostEqual(message.info.origin.position.y, -2.0)
        self.assertAlmostEqual(message.info.origin.orientation.w, 1.0)

        self.assertEqual(len(message.data), 24)
        self.assertEqual(set(message.data), {0, 100})
        # Row-major: the blocked cell is at row 1, column 2.
        self.assertEqual(message.data[1 * 6 + 2], 100)
        self.assertEqual(sum(1 for value in message.data if value == 100), 1)

    def test_round_trip_preserves_the_grid(self):
        rng = np.random.default_rng(0)
        grid = rng.random((7, 5)) > 0.7
        message = to_message(grid, (0.0, 0.0), 0.15)
        restored = np.asarray(message.data, dtype=np.int8).reshape(
            message.info.height, message.info.width
        ) > 0
        np.testing.assert_array_equal(restored, grid)


if __name__ == "__main__":
    unittest.main()

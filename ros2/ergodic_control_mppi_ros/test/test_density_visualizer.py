import unittest

import numpy as np
from visualization_msgs.msg import Marker

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi_ros.density_visualizer import build_density_marker

from helpers import CONFIG


class DensityVisualizerTest(unittest.TestCase):
    def test_marker_matches_workspace_and_density(self):
        config = load_config(CONFIG)
        marker = build_density_marker(config)
        points = np.array([(point.x, point.y, point.z) for point in marker.points])
        colors = np.array([(color.r, color.g, color.b, color.a) for color in marker.colors])

        self.assertEqual(marker.header.frame_id, "world")
        self.assertEqual(marker.type, Marker.TRIANGLE_LIST)
        self.assertEqual(len(marker.points), len(marker.colors))
        self.assertGreater(len(marker.points), 0)
        self.assertEqual(len(marker.points) % 6, 0)
        self.assertTrue(np.allclose(points[:, 2], 0.04))
        self.assertGreaterEqual(points[:, 0].min(), -20.0 - 1e-6)
        self.assertLessEqual(points[:, 0].max(), 20.0 + 1e-6)
        self.assertGreaterEqual(points[:, 1].min(), -10.0 - 1e-6)
        self.assertLessEqual(points[:, 1].max(), 10.0 + 1e-6)
        self.assertTrue(np.all((colors >= 0.0) & (colors <= 1.0)))
        self.assertTrue(np.allclose(colors[:, 3], 1.0))
        self.assertGreater(np.ptp(colors[:, :3], axis=0).max(), 0.5)
        peak = colors[np.argmin(colors[:, 0])]
        self.assertGreater(peak[2], peak[1])
        self.assertGreater(peak[1], peak[0])


if __name__ == "__main__":
    unittest.main()

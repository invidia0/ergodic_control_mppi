"""Publish the configured target density as a flat RViz heatmap."""

import jax.numpy as jnp
from matplotlib import colormaps
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from ergodic_control_mppi.config import AppConfig, load_config
from ergodic_control_mppi.mppi.stein import pdf


def _centers(lower: float, upper: float, resolution: float) -> np.ndarray:
    count = int((upper - lower) // resolution)
    return np.linspace(lower + resolution / 2, upper - resolution / 2, count, dtype=np.float32)


def build_density_marker(config: AppConfig) -> Marker:
    """Build the static target-density marker from validated controller configuration.

    Args:
        config: Validated application configuration.

    Returns:
        A world-frame, unlit ``TRIANGLE_LIST`` marker.
    """
    resolution = config.run.resolution
    workspace = config.controller.workspace
    x = _centers(*np.asarray(workspace.x_limits, dtype=float), resolution)
    y = _centers(*np.asarray(workspace.y_limits, dtype=float), resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    density = np.asarray(pdf(jnp.asarray(xy), config.controller.gmm))
    normalized = density / density.max()
    rgba = colormaps["Blues"](normalized)
    rgba[:, 3] = 1.0
    half = resolution / 2
    offsets = np.array(
        [
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, -half],
            [half, half],
            [-half, half],
        ],
        dtype=np.float32,
    )
    vertices = (xy[:, None, :] + offsets).reshape(-1, 2)
    colors = np.repeat(rgba, 6, axis=0)

    marker = Marker()
    marker.header.frame_id = "world"
    marker.ns = "target_density"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.points = [Point(x=float(px), y=float(py), z=0.04) for px, py in vertices]
    marker.colors = [
        ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))
        for r, g, b, a in colors
    ]
    return marker


class DensityVisualizer(Node):
    """Transient-local publisher for the configured target density."""

    def __init__(self) -> None:
        super().__init__("density_visualizer")
        path = self.declare_parameter("config", "/workspace/configs/mppi_params.yaml").value
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(Marker, "/target_density", qos)
        marker = build_density_marker(load_config(path))
        marker.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(marker)


def main() -> None:
    """Run the density visualizer node."""
    rclpy.init()
    node = DensityVisualizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # The recorder tears the context down under us; rclpy can be mid-take when it
        # goes. Only tolerate it once the context is actually gone -- a RuntimeError while
        # still running is a real fault and must propagate.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        # The recorder ends the run by shutting the context down, so by the time the other
        # nodes unwind it is already closed; shutting down twice raises.
        if rclpy.ok():
            rclpy.shutdown()

"""Turn the static map cloud into the single inflated safety grid used by the deployment.

The grid is published only when the workspace is actually flyable: the arming position and
every target mode centre must be free and mutually connected. A failed check publishes
nothing, which leaves the controller inactive and the shield holding hover -- that absence
is the whole arming interlock, so no extra topic is needed.
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.grid import (
    all_reachable,
    blocked_mode_segments,
    inflate,
    inflation_radius,
    nearest_free,
    rasterize,
    slice_cloud,
)

TRANSIENT_LOCAL = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def build_safety_grid(
    points: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    altitude: float,
    vertical_half_extent: float,
    resolution: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice, rasterize, and inflate a point cloud into a planning grid.

    Returns:
        The raw occupancy and the inflated occupancy, both boolean with shape ``(H, W)``.
    """
    occupancy = rasterize(
        slice_cloud(points, altitude, vertical_half_extent), x_limits, y_limits, resolution
    )
    return occupancy, inflate(occupancy, radius, resolution)


def to_message(grid: np.ndarray, origin: tuple[float, float], resolution: float) -> OccupancyGrid:
    """Pack a boolean grid into a world-frame ``OccupancyGrid`` (0 free, 100 blocked)."""
    message = OccupancyGrid()
    message.header.frame_id = "world"
    message.info.resolution = float(resolution)
    message.info.width = int(grid.shape[1])
    message.info.height = int(grid.shape[0])
    message.info.origin.position.x = float(origin[0])
    message.info.origin.position.y = float(origin[1])
    message.info.origin.orientation.w = 1.0
    message.data = np.where(grid.ravel(), 100, 0).astype(np.int8).tolist()
    return message


def standing_visual_points(points: np.ndarray, base: float) -> np.ndarray:
    """Return the visualization-only cloud as columns standing on the density plane.

    Only the part of each obstacle at or above ``base`` is kept, so the pillars rise out of
    the plane the vehicle flies in and it reads as threading *between* them.

    This replaced a clamp of every z down to a fixed 0.04 m, which flattened each 2-3 m
    pillar into a disk on the floor, 0.71 m below the flight altitude -- in RViz the vehicle
    then appeared to fly over a field of pancakes rather than through a forest.

    Visualization only: the planning occupancy comes from ``build_safety_grid``, which
    slices the *raw* cloud around the flight altitude and is not affected by this.
    """
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return cloud[cloud[:, 2] >= base]


class MapAdapter(Node):
    """Publish the inflated safety grid once the workspace passes the arming check."""

    def __init__(self) -> None:
        super().__init__("map_adapter")
        self.config_path = self.declare_parameter(
            "config", "/workspace/configs/mppi_params.yaml"
        ).value
        self.altitude = self.declare_parameter("altitude", 0.75).value
        self.vertical_half_extent = self.declare_parameter("vertical_half_extent", 0.20).value
        # Round to float32 up front. `OccupancyGrid.info.resolution` is a float32 field, so
        # every consumer reads back 0.15000000596..., not 0.15. Checking arming at full
        # double precision while the controller and guard index at float32 lets the two
        # disagree by one cell -- enough to arm on a map where the controller sees a mode
        # centre buried in an obstacle, which is exactly how mode 2 became unreachable.
        self.resolution = float(
            np.float32(self.declare_parameter("resolution", 0.15).value)
        )
        self.robot_radius = self.declare_parameter("robot_radius", 0.30).value
        self.clearance = self.declare_parameter("clearance", 0.15).value
        self.tracking_allowance = self.declare_parameter("tracking_allowance", 0.20).value
        self.max_speed = self.declare_parameter("max_speed", 2.0).value
        self.brake_accel = self.declare_parameter("brake_accel", 6.0).value
        self.reaction_time = self.declare_parameter("reaction_time", 0.10).value
        # Two scalars rather than one array parameter: launch cannot marshal a list of
        # per-element substitutions, so a list here would silently arrive unset.
        self.start_xy = (
            float(self.declare_parameter("start_x", -16.0).value),
            float(self.declare_parameter("start_y", 0.0).value),
        )

        config = load_config(self.config_path)
        workspace = config.controller.workspace
        self.x_limits = tuple(float(value) for value in np.asarray(workspace.x_limits))
        self.y_limits = tuple(float(value) for value in np.asarray(workspace.y_limits))
        self.origin = (self.x_limits[0], self.y_limits[0])
        self.mode_centers = np.asarray(config.controller.gmm.means, dtype=np.float64)
        self.radius = inflation_radius(
            self.robot_radius,
            self.clearance,
            self.tracking_allowance,
            self.max_speed,
            self.brake_accel,
            self.reaction_time,
            self.resolution,
        )
        self.get_logger().info(
            f"inflation radius {self.radius:.3f} m "
            f"({int(np.ceil(self.radius / self.resolution))} cells at {self.resolution} m)"
        )

        self.publisher = self.create_publisher(
            OccupancyGrid, "/ergodic/safety_grid", TRANSIENT_LOCAL
        )
        # The physical map, uninflated. Planning and the guard use the inflated grid, but a
        # collision claim has to be measured against the obstacles that actually exist --
        # judging contact against a 1.29 m dilation would report a margin breach as a crash.
        self.map_publisher = self.create_publisher(
            OccupancyGrid, "/ergodic/map_grid", TRANSIENT_LOCAL
        )
        self.visual_publisher = self.create_publisher(
            PointCloud2, "/ergodic/map_visual", TRANSIENT_LOCAL
        )
        self.subscription = self.create_subscription(
            PointCloud2, "/mock_map", self.on_cloud, TRANSIENT_LOCAL
        )
        self.armed = False

    def on_cloud(self, message: PointCloud2) -> None:
        """Build, check, and publish the grid. Idempotent: only the first cloud is used."""
        if self.armed:
            return
        points = np.asarray(
            list(point_cloud2.read_points_list(message, field_names=("x", "y", "z"))),
            dtype=np.float64,
        )
        occupancy, inflated = build_safety_grid(
            points,
            self.x_limits,
            self.y_limits,
            self.altitude,
            self.vertical_half_extent,
            self.resolution,
            self.radius,
        )
        start = self.start_xy
        reachable, flags, diagnosis = all_reachable(
            inflated, self.origin, self.resolution, start, self.mode_centers
        )
        free_fraction = float(1.0 - inflated.mean())
        if not reachable:
            self.get_logger().fatal(
                f"refusing to arm after {self.radius:.2f} m inflation: "
                f"free space {free_fraction:.1%} "
                f"({diagnosis['component_cells']} of {diagnosis['free_cells']} free cells "
                f"reachable from start {start}); "
                f"start_blocked={diagnosis['start_blocked']}, "
                f"start_outside={diagnosis['start_outside']}, "
                f"modes inside obstacles={diagnosis['blocked_modes']}, "
                f"modes cut off={diagnosis['disconnected_modes']}"
            )
            if diagnosis["start_blocked"]:
                suggestion = nearest_free(inflated, self.origin, self.resolution, start)
                where = (
                    f" try start_x:={suggestion[0]:.2f} start_y:={suggestion[1]:.2f}"
                    if suggestion
                    else " no free cell exists at this fill"
                )
                self.get_logger().fatal(
                    f"the start position itself is inside the inflated map --{where}, "
                    "or lower map_fill"
                )
            elif diagnosis["blocked_modes"]:
                self.get_logger().fatal(
                    "target modes sit inside inflated obstacles -- lower map_fill, or "
                    "change the density so its modes are in free space"
                )
            else:
                self.get_logger().fatal(
                    "the workspace is fragmented -- lower map_fill so the free space "
                    "reconnects"
                )
            return

        stamp = self.get_clock().now().to_msg()
        raw_out = to_message(occupancy, self.origin, self.resolution)
        raw_out.header.stamp = stamp
        self.map_publisher.publish(raw_out)
        self.visual_publisher.publish(
            point_cloud2.create_cloud_xyz32(
                message.header, standing_visual_points(points, self.altitude).tolist()
            )
        )
        message_out = to_message(inflated, self.origin, self.resolution)
        message_out.header.stamp = stamp
        self.publisher.publish(message_out)
        self.armed = True
        if diagnosis["start_outside"]:
            self.get_logger().info(
                f"start {start} is outside the map; the vehicle flies in and the boundary "
                "term recovers it. Connectivity was judged from its entry cell."
            )
        self.get_logger().info(
            f"armed: {occupancy.sum()} occupied cells, free space {free_fraction:.1%}, "
            f"all {len(flags)} modes reachable from {start}, "
            f"blocked_mode_segments={blocked_mode_segments(inflated, self.origin, self.resolution, self.mode_centers)}"
        )


def main() -> None:
    """Run the map adapter node."""
    rclpy.init()
    node = MapAdapter()
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

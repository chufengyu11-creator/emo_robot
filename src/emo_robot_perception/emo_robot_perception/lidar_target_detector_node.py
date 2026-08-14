#!/usr/bin/env python3

"""Publish prototype target estimates from the X2 chest LiDAR."""

from __future__ import annotations

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from emo_robot_interfaces.msg import LidarTarget
from emo_robot_perception.lidar_target_estimator import (
    LidarTargetEstimate,
    LidarTargetEstimator,
    LidarTargetEstimatorConfig,
)


class LidarTargetDetectorNode(Node):
    """Detect a point-cloud cluster without commanding the robot."""

    def __init__(self) -> None:
        super().__init__("lidar_target_detector")
        self.declare_parameter(
            "pointcloud_topic",
            "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud",
        )
        self.declare_parameter(
            "target_topic",
            "/emo_robot/perception/lidar_target",
        )
        self.declare_parameter("forward_axis", "x")
        self.declare_parameter("lateral_axis", "z")
        self.declare_parameter("vertical_axis", "y")
        self.declare_parameter("forward_min_m", 0.4)
        self.declare_parameter("forward_max_m", 10.0)
        self.declare_parameter("vertical_min_m", -1.62)
        self.declare_parameter("vertical_max_m", 0.38)
        self.declare_parameter("lateral_abs_max_m", 4.0)
        self.declare_parameter("lateral_bin_size_m", 0.10)
        self.declare_parameter("cluster_half_width_m", 0.25)
        self.declare_parameter("min_cluster_points", 50)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("tracking_enabled", True)
        self.declare_parameter("candidate_count", 5)
        self.declare_parameter("track_min_cluster_points", 25)
        self.declare_parameter("max_tracking_forward_jump_m", 1.0)
        self.declare_parameter("max_tracking_lateral_jump_m", 0.8)
        self.declare_parameter("max_tracking_angle_jump_deg", 18.0)
        self.declare_parameter("lost_grace_frames", 5)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter(
            "debug_image_topic",
            "/emo_robot/lidar/target_debug_image",
        )
        self.declare_parameter("debug_image_width", 640)
        self.declare_parameter("debug_image_height", 480)
        self.declare_parameter("debug_forward_range_m", 4.0)
        self.declare_parameter("debug_lateral_range_m", 4.0)

        config = LidarTargetEstimatorConfig(
            forward_axis=str(self.get_parameter("forward_axis").value),
            lateral_axis=str(self.get_parameter("lateral_axis").value),
            vertical_axis=str(self.get_parameter("vertical_axis").value),
            forward_min_m=float(
                self.get_parameter("forward_min_m").value
            ),
            forward_max_m=float(
                self.get_parameter("forward_max_m").value
            ),
            vertical_min_m=float(
                self.get_parameter("vertical_min_m").value
            ),
            vertical_max_m=float(
                self.get_parameter("vertical_max_m").value
            ),
            lateral_abs_max_m=float(
                self.get_parameter("lateral_abs_max_m").value
            ),
            lateral_bin_size_m=float(
                self.get_parameter("lateral_bin_size_m").value
            ),
            cluster_half_width_m=float(
                self.get_parameter("cluster_half_width_m").value
            ),
            min_cluster_points=int(
                self.get_parameter("min_cluster_points").value
            ),
            smoothing_window=int(
                self.get_parameter("smoothing_window").value
            ),
            tracking_enabled=bool(
                self.get_parameter("tracking_enabled").value
            ),
            candidate_count=int(
                self.get_parameter("candidate_count").value
            ),
            track_min_cluster_points=int(
                self.get_parameter("track_min_cluster_points").value
            ),
            max_tracking_forward_jump_m=float(
                self.get_parameter("max_tracking_forward_jump_m").value
            ),
            max_tracking_lateral_jump_m=float(
                self.get_parameter("max_tracking_lateral_jump_m").value
            ),
            max_tracking_angle_jump_deg=float(
                self.get_parameter("max_tracking_angle_jump_deg").value
            ),
            lost_grace_frames=int(
                self.get_parameter("lost_grace_frames").value
            ),
        )
        self._estimator = LidarTargetEstimator(config)
        self._publish_debug = bool(
            self.get_parameter("publish_debug_image").value
        )
        self._debug_width = int(
            self.get_parameter("debug_image_width").value
        )
        self._debug_height = int(
            self.get_parameter("debug_image_height").value
        )
        self._debug_forward_range = float(
            self.get_parameter("debug_forward_range_m").value
        )
        self._debug_lateral_range = float(
            self.get_parameter("debug_lateral_range_m").value
        )
        if self._debug_width <= 0 or self._debug_height <= 0:
            raise ValueError("debug image dimensions must be positive")
        if self._debug_forward_range <= 0.0:
            raise ValueError("debug_forward_range_m must be positive")
        if self._debug_lateral_range <= 0.0:
            raise ValueError("debug_lateral_range_m must be positive")

        self._bridge = CvBridge()
        self._target_pub = self.create_publisher(
            LidarTarget,
            str(self.get_parameter("target_topic").value),
            10,
        )
        self._debug_pub = (
            self.create_publisher(
                Image,
                str(self.get_parameter("debug_image_topic").value),
                10,
            )
            if self._publish_debug
            else None
        )
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self._on_pointcloud,
            qos,
        )
        self._conversion_error_logged = False
        self.get_logger().info(
            "LiDAR target detector ready in observation-only mode; "
            "no velocity commands will be published."
        )

    def _on_pointcloud(self, message: PointCloud2) -> None:
        """Estimate and publish one result for every input cloud."""
        try:
            points = self._read_xyz_points(message)
            estimate = self._estimator.estimate(points)
        except Exception as exc:
            if not self._conversion_error_logged:
                self._conversion_error_logged = True
                self.get_logger().error(
                    f"LiDAR target processing failed: {exc}"
                )
            self._estimator.reset()
            estimate = LidarTargetEstimate()

        self._publish_target(message, estimate)
        self._publish_debug_image(message, estimate)

    def _publish_target(
        self,
        source: PointCloud2,
        estimate: LidarTargetEstimate,
    ) -> None:
        message = LidarTarget()
        message.header = source.header
        message.valid = bool(estimate.valid)
        message.forward_m = float(estimate.forward_m)
        message.lateral_m = float(estimate.lateral_m)
        message.angle_rad = float(estimate.angle_rad)
        message.point_count = int(estimate.point_count)
        self._target_pub.publish(message)

    def _publish_debug_image(
        self,
        source: PointCloud2,
        estimate: LidarTargetEstimate,
    ) -> None:
        if self._debug_pub is None:
            return
        frame = self._render_debug_image(estimate)
        message = self._bridge.cv2_to_imgmsg(frame, "bgr8")
        message.header.stamp = source.header.stamp
        message.header.frame_id = source.header.frame_id
        self._debug_pub.publish(message)

    @staticmethod
    def _read_xyz_points(message: PointCloud2) -> np.ndarray:
        field_names = {field.name for field in message.fields}
        missing = {"x", "y", "z"} - field_names
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"PointCloud2 is missing required fields: {missing_text}"
            )

        raw_points = point_cloud2.read_points(
            message,
            field_names=("x", "y", "z"),
            skip_nans=False,
        )
        if isinstance(raw_points, np.ndarray):
            if raw_points.dtype.names:
                return np.column_stack(
                    (
                        raw_points["x"],
                        raw_points["y"],
                        raw_points["z"],
                    )
                ).astype(np.float32, copy=False)
            points = np.asarray(raw_points, dtype=np.float32)
        else:
            points_list = list(raw_points)
            if not points_list:
                return np.empty((0, 3), dtype=np.float32)
            points = np.asarray(points_list, dtype=np.float32)

        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return points[:, :3].astype(np.float32, copy=False)

    def _render_debug_image(
        self,
        estimate: LidarTargetEstimate,
    ) -> np.ndarray:
        frame = np.full(
            (self._debug_height, self._debug_width, 3),
            (18, 18, 18),
            dtype=np.uint8,
        )
        self._draw_grid(frame)
        self._draw_points(frame, estimate.filtered_points, cluster=False)
        self._draw_points(frame, estimate.cluster_points, cluster=True)

        status = "INVALID"
        color = (0, 0, 255)
        if estimate.valid:
            status = (
                f"TARGET x={estimate.forward_m:.2f}m "
                f"lat={estimate.lateral_m:+.2f}m "
                f"angle={math.degrees(estimate.angle_rad):+.1f}deg "
                f"points={estimate.point_count}"
            )
            color = (0, 255, 0)
            pixel = self._point_to_pixel(
                np.asarray([estimate.forward_m]),
                np.asarray([estimate.lateral_m]),
            )
            cv2.circle(
                frame,
                (int(pixel[0][0]), int(pixel[1][0])),
                8,
                color,
                2,
            )
        cv2.putText(
            frame,
            status,
            (12, self._debug_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        return frame

    def _draw_grid(self, frame: np.ndarray) -> None:
        origin_x = self._debug_width // 2
        origin_y = self._debug_height - 28
        cv2.line(
            frame,
            (origin_x, 24),
            (origin_x, origin_y),
            (90, 90, 90),
            1,
        )
        cv2.line(
            frame,
            (24, origin_y),
            (self._debug_width - 24, origin_y),
            (90, 90, 90),
            1,
        )
        cv2.circle(frame, (origin_x, origin_y), 6, (0, 255, 255), -1)
        cv2.putText(
            frame,
            "LiDAR target prototype",
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _draw_points(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        cluster: bool,
    ) -> None:
        forward, lateral = self._estimator.project_forward_lateral(points)
        if forward.size == 0:
            return
        pixel_x, pixel_y = self._point_to_pixel(forward, lateral)
        inside = (
            (forward >= 0.0)
            & (forward <= self._debug_forward_range)
            & (np.abs(lateral) <= self._debug_lateral_range / 2.0)
            & (pixel_x >= 0)
            & (pixel_x < self._debug_width)
            & (pixel_y >= 0)
            & (pixel_y < self._debug_height)
        )
        if not np.any(inside):
            return
        px = pixel_x[inside]
        py = pixel_y[inside]
        if cluster:
            frame[py, px] = (0, 255, 0)
            return
        ratio = np.clip(
            forward[inside] / self._debug_forward_range,
            0.0,
            1.0,
        )
        frame[py, px] = np.column_stack(
            (
                220.0 * ratio,
                80.0 + 120.0 * ratio,
                255.0 * (1.0 - ratio),
            )
        ).astype(np.uint8)

    def _point_to_pixel(
        self,
        forward: np.ndarray,
        lateral: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        margin_top = 28
        margin_bottom = 28
        margin_x = 24
        usable_height = max(
            1,
            self._debug_height - margin_top - margin_bottom,
        )
        usable_width = max(1, self._debug_width - 2 * margin_x)
        origin_x = self._debug_width // 2
        origin_y = self._debug_height - margin_bottom
        pixel_x = origin_x - (
            lateral / (self._debug_lateral_range / 2.0)
        ) * (usable_width / 2.0)
        pixel_y = origin_y - (
            forward / self._debug_forward_range
        ) * usable_height
        return (
            np.rint(pixel_x).astype(np.int32),
            np.rint(pixel_y).astype(np.int32),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LidarTargetDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

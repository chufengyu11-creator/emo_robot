#!/usr/bin/env python3

"""Subscribe to the X2 chest LiDAR point cloud and/or IMU streams."""

from collections import deque
from typing import Deque

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
from sensor_msgs.msg import Image, Imu, PointCloud2
from sensor_msgs_py import point_cloud2


class LidarSubscriberNode(Node):
    """Observe X2 chest LiDAR streams without commanding the robot."""

    def __init__(self) -> None:
        super().__init__("lidar_subscriber")
        self.declare_parameter("topic_type", "both")
        self.declare_parameter(
            "pointcloud_topic",
            "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud",
        )
        self.declare_parameter(
            "imu_topic",
            "/aima/hal/sensor/lidar_chest_front/imu",
        )
        self.declare_parameter("log_interval_s", 1.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter(
            "debug_image_topic",
            "/emo_robot/lidar/debug_image",
        )
        self.declare_parameter("debug_image_width", 640)
        self.declare_parameter("debug_image_height", 480)
        self.declare_parameter("debug_range_x_m", 4.0)
        self.declare_parameter("debug_lateral_range_m", 3.0)
        self.declare_parameter("debug_min_height_m", -1.0)
        self.declare_parameter("debug_max_height_m", 1.5)
        self.declare_parameter("debug_flip_lateral", False)

        self._topic_type = str(
            self.get_parameter("topic_type").value
        ).lower()
        self._pointcloud_topic = str(
            self.get_parameter("pointcloud_topic").value
        )
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._log_interval_s = float(
            self.get_parameter("log_interval_s").value
        )
        self._publish_debug = bool(
            self.get_parameter("publish_debug_image").value
        )
        self._debug_image_topic = str(
            self.get_parameter("debug_image_topic").value
        )
        self._debug_image_width = int(
            self.get_parameter("debug_image_width").value
        )
        self._debug_image_height = int(
            self.get_parameter("debug_image_height").value
        )
        self._debug_range_x_m = float(
            self.get_parameter("debug_range_x_m").value
        )
        self._debug_lateral_range_m = float(
            self.get_parameter("debug_lateral_range_m").value
        )
        self._debug_min_height_m = float(
            self.get_parameter("debug_min_height_m").value
        )
        self._debug_max_height_m = float(
            self.get_parameter("debug_max_height_m").value
        )
        self._debug_flip_lateral = bool(
            self.get_parameter("debug_flip_lateral").value
        )

        if self._topic_type not in {"pointcloud", "imu", "both"}:
            raise ValueError(
                "topic_type must be 'pointcloud', 'imu', or 'both'"
            )
        if self._log_interval_s <= 0.0:
            raise ValueError("log_interval_s must be greater than zero")
        if self._debug_image_width <= 0 or self._debug_image_height <= 0:
            raise ValueError("debug image dimensions must be positive")
        if (
            self._debug_range_x_m <= 0.0
            or self._debug_lateral_range_m <= 0.0
        ):
            raise ValueError("debug ranges must be greater than zero")
        if self._debug_min_height_m >= self._debug_max_height_m:
            raise ValueError(
                "debug_min_height_m must be less than "
                "debug_max_height_m"
            )

        # Match the QoS used by the official AimDK Python example.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._pointcloud_arrivals: Deque = deque()
        self._imu_arrivals: Deque = deque()
        now = self.get_clock().now()
        self._pointcloud_last_log = now
        self._imu_last_log = now
        self._pointcloud_first_received = False
        self._imu_first_received = False
        self._pointcloud_field_warning_logged = False
        self._pointcloud_debug_error_logged = False

        self._bridge = CvBridge()
        self._debug_pub = (
            self.create_publisher(
                Image,
                self._debug_image_topic,
                10,
            )
            if self._publish_debug
            else None
        )
        if self._debug_pub is not None:
            self.get_logger().info(
                f"Publishing LiDAR debug image: {self._debug_image_topic}"
            )

        if self._topic_type in {"pointcloud", "both"}:
            self._pointcloud_subscription = self.create_subscription(
                PointCloud2,
                self._pointcloud_topic,
                self._on_pointcloud,
                qos,
            )
            self.get_logger().info(
                f"Subscribing to LiDAR point cloud: "
                f"{self._pointcloud_topic}"
            )

        if self._topic_type in {"imu", "both"}:
            self._imu_subscription = self.create_subscription(
                Imu,
                self._imu_topic,
                self._on_imu,
                qos,
            )
            self.get_logger().info(
                f"Subscribing to LiDAR IMU: {self._imu_topic}"
            )

    def _update_rate(self, arrivals: Deque) -> float:
        """Record one arrival and return the count in the latest second."""
        now = self.get_clock().now()
        arrivals.append(now)
        while (
            arrivals
            and (now - arrivals[0]).nanoseconds * 1e-9 > 1.0
        ):
            arrivals.popleft()
        return float(len(arrivals))

    def _should_log(self, last_log) -> bool:
        """Return true when the configured logging period has elapsed."""
        return (
            (self.get_clock().now() - last_log).nanoseconds * 1e-9
            >= self._log_interval_s
        )

    def _on_pointcloud(self, message: PointCloud2) -> None:
        """Report point-cloud metadata and receive rate."""
        receive_rate = self._update_rate(self._pointcloud_arrivals)
        if not self._pointcloud_first_received:
            self._pointcloud_first_received = True
            self.get_logger().info("First LiDAR point cloud received.")

        self._publish_pointcloud_debug_image(message)

        if not self._should_log(self._pointcloud_last_log):
            return
        self._pointcloud_last_log = self.get_clock().now()

        fields = ", ".join(
            f"{field.name}({field.datatype})"
            for field in message.fields
        )
        point_count = int(message.width) * int(message.height)
        self.get_logger().info(
            "LiDAR PointCloud2\n"
            f"  frame_id: {message.header.frame_id}\n"
            f"  dimensions: {message.width} x {message.height}\n"
            f"  points: {point_count}\n"
            f"  point_step: {message.point_step}\n"
            f"  fields: {fields}\n"
            f"  data_size: {len(message.data)} bytes\n"
            f"  receive_rate: {receive_rate:.1f} Hz"
        )

    def _publish_pointcloud_debug_image(self, message: PointCloud2) -> None:
        if self._debug_pub is None:
            return

        field_names = {field.name for field in message.fields}
        missing = {"x", "y", "z"} - field_names
        if missing:
            if not self._pointcloud_field_warning_logged:
                self._pointcloud_field_warning_logged = True
                missing_text = ", ".join(sorted(missing))
                self.get_logger().warning(
                    "PointCloud2 is missing required debug fields: "
                    f"{missing_text}"
                )
            return

        try:
            points = self._read_xyz_points(message)
            debug_frame = self._render_debug_image(points)
            debug_msg = self._bridge.cv2_to_imgmsg(debug_frame, "bgr8")
            debug_msg.header.stamp = message.header.stamp
            debug_msg.header.frame_id = message.header.frame_id
            self._debug_pub.publish(debug_msg)
        except Exception as exc:
            if not self._pointcloud_debug_error_logged:
                self._pointcloud_debug_error_logged = True
                self.get_logger().error(
                    f"LiDAR debug image rendering failed: {exc}"
                )

    @staticmethod
    def _read_xyz_points(message: PointCloud2) -> np.ndarray:
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

    def _render_debug_image(self, points: np.ndarray) -> np.ndarray:
        image = np.full(
            (self._debug_image_height, self._debug_image_width, 3),
            (18, 18, 18),
            dtype=np.uint8,
        )
        self._draw_debug_grid(image)

        filtered = self._filter_debug_points(points)
        if filtered.size:
            self._draw_debug_points(image, filtered)

        self._draw_debug_status(image, points, filtered)
        return image

    def _filter_debug_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        mask = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (x >= 0.0)
            & (x <= self._debug_range_x_m)
            & (y >= self._debug_min_height_m)
            & (y <= self._debug_max_height_m)
            & (np.abs(z) <= self._debug_lateral_range_m / 2.0)
        )
        return points[mask]

    def _point_to_pixel(
        self,
        forward_m: np.ndarray,
        lateral_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        margin_top = 28
        margin_bottom = 28
        margin_x = 24
        usable_height = max(
            1,
            self._debug_image_height - margin_top - margin_bottom,
        )
        usable_width = max(1, self._debug_image_width - 2 * margin_x)
        origin_x = self._debug_image_width // 2
        origin_y = self._debug_image_height - margin_bottom

        lateral = -lateral_m if self._debug_flip_lateral else lateral_m
        pixel_x = origin_x - (
            lateral / (self._debug_lateral_range_m / 2.0)
        ) * (usable_width / 2.0)
        pixel_y = origin_y - (
            forward_m / self._debug_range_x_m
        ) * usable_height

        return (
            np.rint(pixel_x).astype(np.int32),
            np.rint(pixel_y).astype(np.int32),
        )

    def _draw_debug_grid(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        origin_x = width // 2
        origin_y = height - 28
        grid_color = (55, 55, 55)
        axis_color = (90, 90, 90)

        cv2.line(image, (origin_x, 24), (origin_x, origin_y), axis_color, 1)
        cv2.line(image, (24, origin_y), (width - 24, origin_y), axis_color, 1)

        max_distance = int(np.floor(self._debug_range_x_m))
        for distance_m in range(1, max_distance + 1):
            y = self._point_to_pixel(
                np.asarray([float(distance_m)]),
                np.asarray([0.0]),
            )[1][0]
            cv2.line(image, (24, y), (width - 24, y), grid_color, 1)
            cv2.putText(
                image,
                f"{distance_m}m",
                (30, max(18, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

        half_range = self._debug_lateral_range_m / 2.0
        for lateral_m in (-half_range, half_range):
            x = self._point_to_pixel(
                np.asarray([0.0]),
                np.asarray([lateral_m]),
            )[0][0]
            cv2.line(image, (x, 24), (x, origin_y), grid_color, 1)

        cv2.circle(image, (origin_x, origin_y), 6, (0, 255, 255), -1)
        cv2.putText(
            image,
            "LiDAR top view",
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _draw_debug_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
    ) -> None:
        x = points[:, 0]
        z = points[:, 2]
        pixel_x, pixel_y = self._point_to_pixel(x, z)
        inside = (
            (pixel_x >= 0)
            & (pixel_x < self._debug_image_width)
            & (pixel_y >= 0)
            & (pixel_y < self._debug_image_height)
        )
        pixel_x = pixel_x[inside]
        pixel_y = pixel_y[inside]
        distance_ratio = np.clip(x[inside] / self._debug_range_x_m, 0.0, 1.0)

        colors = np.column_stack(
            (
                220.0 * distance_ratio,
                80.0 + 120.0 * distance_ratio,
                255.0 * (1.0 - distance_ratio),
            )
        ).astype(np.uint8)
        point_layer = np.zeros_like(image)
        point_layer[pixel_y, pixel_x] = colors
        point_layer = cv2.dilate(
            point_layer,
            np.ones((2, 2), dtype=np.uint8),
        )
        mask = np.any(point_layer > 0, axis=2)
        image[mask] = point_layer[mask]

    def _draw_debug_status(
        self,
        image: np.ndarray,
        raw_points: np.ndarray,
        filtered_points: np.ndarray,
    ) -> None:
        if filtered_points.size:
            distances = np.linalg.norm(
                filtered_points[:, (0, 2)],
                axis=1,
            )
            nearest_text = f"{float(np.min(distances)):.2f}m"
        else:
            nearest_text = "none"

        if raw_points.size == 0:
            status = "no points"
        else:
            status = (
                f"points {len(filtered_points)}/{len(raw_points)}  "
                f"nearest {nearest_text}"
            )

        cv2.putText(
            image,
            status,
            (12, self._debug_image_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _on_imu(self, message: Imu) -> None:
        """Report LiDAR IMU values and receive rate."""
        receive_rate = self._update_rate(self._imu_arrivals)
        if not self._imu_first_received:
            self._imu_first_received = True
            self.get_logger().info("First LiDAR IMU sample received.")

        if not self._should_log(self._imu_last_log):
            return
        self._imu_last_log = self.get_clock().now()

        orientation = message.orientation
        angular = message.angular_velocity
        acceleration = message.linear_acceleration
        self.get_logger().info(
            "LiDAR IMU\n"
            f"  frame_id: {message.header.frame_id}\n"
            "  orientation: "
            f"[{orientation.x:.6f}, {orientation.y:.6f}, "
            f"{orientation.z:.6f}, {orientation.w:.6f}]\n"
            "  angular_velocity: "
            f"[{angular.x:.6f}, {angular.y:.6f}, {angular.z:.6f}]\n"
            "  linear_acceleration: "
            f"[{acceleration.x:.6f}, {acceleration.y:.6f}, "
            f"{acceleration.z:.6f}]\n"
            f"  receive_rate: {receive_rate:.1f} Hz"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LidarSubscriberNode()
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

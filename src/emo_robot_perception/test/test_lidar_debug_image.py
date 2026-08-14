"""Unit tests for LiDAR debug image rendering."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


_PARAM_OVERRIDES = {}


def _install_ros_stubs() -> None:
    rclpy = ModuleType("rclpy")
    rclpy.init = Mock()
    rclpy.spin = Mock()
    rclpy.ok = Mock(return_value=False)
    rclpy.shutdown = Mock()

    rclpy_node = ModuleType("rclpy.node")

    class _Duration:
        def __init__(self, nanoseconds=0):
            self.nanoseconds = nanoseconds

    class _Time:
        nanoseconds = 0

        def __sub__(self, _other):
            return _Duration(0)

        def to_msg(self):
            return "stamp"

    class _Node:
        def __init__(self, _name):
            self._parameters = {}
            self.publishers = []
            self.subscriptions = []
            self.logger = Mock()

        def declare_parameter(self, name, default):
            value = _PARAM_OVERRIDES.get(name, default)
            self._parameters[name] = SimpleNamespace(value=value)

        def get_parameter(self, name):
            return self._parameters[name]

        def create_publisher(self, *_args):
            publisher = SimpleNamespace(publish=Mock())
            self.publishers.append(publisher)
            return publisher

        def create_subscription(self, *args):
            self.subscriptions.append(args)
            return object()

        def get_clock(self):
            return SimpleNamespace(now=lambda: _Time())

        def get_logger(self):
            return self.logger

    rclpy_node.Node = _Node
    rclpy.node = rclpy_node

    rclpy_qos = ModuleType("rclpy.qos")

    class _QoSProfile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    rclpy_qos.QoSHistoryPolicy = SimpleNamespace(KEEP_LAST=1)
    rclpy_qos.QoSProfile = _QoSProfile
    rclpy_qos.QoSReliabilityPolicy = SimpleNamespace(BEST_EFFORT=1)
    rclpy_qos.qos_profile_sensor_data = object()

    cv_bridge = ModuleType("cv_bridge")

    class _CvBridge:
        def cv2_to_imgmsg(self, frame, encoding):
            return SimpleNamespace(
                header=SimpleNamespace(stamp=None, frame_id=""),
                encoding=encoding,
                height=frame.shape[0],
                width=frame.shape[1],
                data=frame.copy(),
            )

    cv_bridge.CvBridge = _CvBridge

    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")

    class _Image:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id="")

    sensor_msgs_msg.Image = _Image
    sensor_msgs_msg.Imu = type("Imu", (), {})
    sensor_msgs_msg.PointCloud2 = type("PointCloud2", (), {})
    sensor_msgs.msg = sensor_msgs_msg

    emo_robot_interfaces = ModuleType("emo_robot_interfaces")
    emo_robot_interfaces_msg = ModuleType("emo_robot_interfaces.msg")

    class _LidarTarget:
        def __init__(self):
            self.header = None
            self.valid = False
            self.forward_m = 0.0
            self.lateral_m = 0.0
            self.angle_rad = 0.0
            self.point_count = 0

    emo_robot_interfaces_msg.LidarTarget = _LidarTarget
    emo_robot_interfaces.msg = emo_robot_interfaces_msg

    ament_index_python = ModuleType("ament_index_python")
    ament_index_packages = ModuleType("ament_index_python.packages")
    ament_index_packages.get_package_share_directory = (
        lambda _package_name: ""
    )
    ament_index_packages.get_package_prefix = lambda _package_name: ""
    ament_index_python.packages = ament_index_packages

    sensor_msgs_py = ModuleType("sensor_msgs_py")
    point_cloud2 = ModuleType("sensor_msgs_py.point_cloud2")
    point_cloud2.read_points = (
        lambda message, field_names, skip_nans: message.points
    )
    sensor_msgs_py.point_cloud2 = point_cloud2

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["rclpy.qos"] = rclpy_qos
    sys.modules["cv_bridge"] = cv_bridge
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg
    sys.modules["sensor_msgs_py"] = sensor_msgs_py
    sys.modules["sensor_msgs_py.point_cloud2"] = point_cloud2
    sys.modules["emo_robot_interfaces"] = emo_robot_interfaces
    sys.modules["emo_robot_interfaces.msg"] = emo_robot_interfaces_msg
    sys.modules["ament_index_python"] = ament_index_python
    sys.modules["ament_index_python.packages"] = ament_index_packages


_install_ros_stubs()

from emo_robot_perception.lidar_subscriber_node import (  # noqa: E402
    LidarSubscriberNode,
)
from emo_robot_perception.lidar_target_detector_node import (  # noqa: E402
    LidarTargetDetectorNode,
)


def _field(name):
    return SimpleNamespace(name=name, datatype=7)


def _pointcloud(points, fields=("x", "y", "z")):
    return SimpleNamespace(
        header=SimpleNamespace(stamp="source_stamp", frame_id="lidar_frame"),
        fields=[_field(name) for name in fields],
        width=len(points),
        height=1,
        point_step=12,
        data=b"\x00" * (12 * len(points)),
        points=points,
    )


def _node(**params):
    _PARAM_OVERRIDES.clear()
    _PARAM_OVERRIDES.update(params)
    return LidarSubscriberNode()


def test_publish_debug_image_false_does_not_create_debug_publisher():
    node = _node(topic_type="pointcloud", publish_debug_image=False)

    assert node._debug_pub is None


def test_empty_pointcloud_publishes_blank_debug_image():
    node = _node(topic_type="pointcloud", publish_debug_image=True)

    node._on_pointcloud(_pointcloud([]))

    node._debug_pub.publish.assert_called_once()
    message = node._debug_pub.publish.call_args.args[0]
    assert message.encoding == "bgr8"
    assert message.width == 640
    assert message.height == 480
    assert message.header.stamp == "source_stamp"
    assert message.header.frame_id == "lidar_frame"


def test_valid_pointcloud_publishes_bgr8_debug_image():
    node = _node(topic_type="pointcloud", publish_debug_image=True)

    node._on_pointcloud(
        _pointcloud(
            [
                (0.5, 0.0, 0.1),
                (1.2, -0.4, 0.0),
                (2.5, 0.8, 0.2),
            ]
        )
    )

    message = node._debug_pub.publish.call_args.args[0]
    assert message.encoding == "bgr8"
    assert np.count_nonzero(message.data) > 0


def test_nan_inf_and_out_of_range_points_are_filtered():
    node = _node(topic_type="pointcloud", publish_debug_image=True)
    points = np.asarray(
        [
            [0.5, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [1.0, np.inf, 0.0],
            [5.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    filtered = node._filter_debug_points(points)

    assert filtered.shape == (1, 3)
    assert filtered[0, 0] == 0.5


def test_positive_z_is_drawn_on_robot_left():
    node = _node(topic_type="pointcloud", publish_debug_image=True)

    center_x = node._debug_image_width // 2
    positive_z_pixel = node._point_to_pixel(
        np.asarray([1.0]),
        np.asarray([0.5]),
    )[0][0]

    assert positive_z_pixel < center_x


def test_missing_xyz_fields_do_not_publish_debug_image():
    node = _node(topic_type="pointcloud", publish_debug_image=True)

    node._on_pointcloud(_pointcloud([(1.0, 0.0)], fields=("x", "y")))

    node._debug_pub.publish.assert_not_called()
    node.get_logger().warning.assert_called_once()


def _target_node(**params):
    _PARAM_OVERRIDES.clear()
    _PARAM_OVERRIDES.update(params)
    return LidarTargetDetectorNode()


def test_target_detector_converts_structured_xyz_array():
    points = np.asarray(
        [(1.0, 0.2, 0.0), (2.0, -0.3, 0.1)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )
    message = _pointcloud(points)

    converted = LidarTargetDetectorNode._read_xyz_points(message)

    assert converted.shape == (2, 3)
    assert converted.dtype == np.float32
    assert converted[1].tolist() == pytest.approx([2.0, -0.3, 0.1])


def test_target_detector_publishes_invalid_for_empty_cloud():
    node = _target_node(min_cluster_points=3)
    source = _pointcloud([])

    node._on_pointcloud(source)

    message = node._target_pub.publish.call_args.args[0]
    assert message.header is source.header
    assert message.valid is False
    assert message.forward_m == 0.0
    assert message.lateral_m == 0.0
    assert message.angle_rad == 0.0
    assert message.point_count == 0
    node._debug_pub.publish.assert_called_once()


def test_target_detector_publishes_valid_target_and_debug_cluster():
    node = _target_node(min_cluster_points=3)
    source = _pointcloud(
        [
            (2.0, 0.0, 0.03),
            (2.1, 0.0, 0.05),
            (2.2, 0.0, 0.07),
        ]
    )

    node._on_pointcloud(source)

    target = node._target_pub.publish.call_args.args[0]
    assert target.valid is True
    assert target.forward_m == pytest.approx(2.1)
    assert target.lateral_m == pytest.approx(0.05)
    assert target.point_count == 3
    debug = node._debug_pub.publish.call_args.args[0]
    assert debug.header.stamp == "source_stamp"
    assert debug.header.frame_id == "lidar_frame"
    assert np.any(debug.data[:, :, 1] == 255)


def test_target_detector_missing_field_still_publishes_invalid():
    node = _target_node(min_cluster_points=3)
    source = _pointcloud([(1.0, 0.0)], fields=("x", "y"))

    node._on_pointcloud(source)

    target = node._target_pub.publish.call_args.args[0]
    assert target.valid is False
    node.get_logger().error.assert_called_once()

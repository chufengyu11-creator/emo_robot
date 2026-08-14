"""Launch gesture, single-person LiDAR fusion, and safe approach nodes.

The approach controller loads in observation-only mode by default; it does not
register an AimDK input source unless explicitly armed in the YAML config.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _model_override_values(context):
    """Return only model-related launch arguments set by the caller."""
    overrides = {}
    for parameter_name in ("model_backend", "model_path", "yolo_device"):
        value = LaunchConfiguration(parameter_name).perform(context).strip()
        if value:
            overrides[parameter_name] = value
    return overrides


def _gesture_detector_node(context, config):
    """Load YAML defaults and apply only non-empty CLI overrides."""
    overrides = {
        name: ParameterValue(value, value_type=str)
        for name, value in _model_override_values(context).items()
    }

    parameters = [config]
    if overrides:
        parameters.append(overrides)
    return [
        Node(
            package="emo_robot_perception",
            executable="gesture_detector",
            name="gesture_detector",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(
        get_package_share_directory("emo_robot_bringup"),
        "config",
        "gesture_approach.yaml",
    )
    camera_device = LaunchConfiguration("camera_device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_device",
                default_value="/dev/video11",
                description="V4L2 device for the external USB camera",
            ),
            DeclareLaunchArgument(
                "model_backend",
                default_value="",
                description=(
                    "Packaged model backend override: pt or onnx; "
                    "empty uses gesture_approach.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description=(
                    "Custom YOLO pose model override; empty uses "
                    "model_backend from gesture_approach.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "yolo_device",
                default_value="",
                description=(
                    "Ultralytics device override: auto, cpu, or CUDA "
                    "index such as 0; empty uses gesture_approach.yaml"
                ),
            ),
            # ---------- perception ----------
            Node(
                package="emo_robot_perception",
                executable="usb_camera",
                name="usb_camera",
                output="screen",
                parameters=[config, {"device": camera_device}],
            ),
            # Raw point-cloud/IMU diagnostics remain available via:
            # ros2 run emo_robot_perception lidar_subscriber ...
            # Node(
            #     package="emo_robot_perception",
            #     executable="lidar_subscriber",
            #     name="lidar_subscriber",
            #     output="screen",
            #     parameters=[config],
            # ),
            Node(
                package="emo_robot_perception",
                executable="lidar_target_detector",
                name="lidar_target_detector",
                output="screen",
                parameters=[config],
            ),
            OpaqueFunction(
                function=_gesture_detector_node,
                args=[config],
            ),
            # ---------- response ----------
            Node(
                package="emo_robot_control",
                executable="interaction_responder",
                name="interaction_responder",
                output="screen",
                parameters=[config],
            ),
            # ---------- single-person approach (safe default) ----------
            Node(
                package="emo_robot_perception",
                executable="person_tracker",
                name="person_tracker",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="emo_robot_control",
                executable="approach_controller",
                name="approach_controller",
                output="screen",
                parameters=[config],
            ),
        ]
    )

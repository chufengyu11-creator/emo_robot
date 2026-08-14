#!/usr/bin/env python3

"""Write selected approach-pipeline ROS logs to a line-buffered text file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, Sequence

import rclpy
from rcl_interfaces.msg import Log
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


TARGET_LOGGERS = frozenset(
    {
        "gesture_detector",
        "lidar_target_detector",
        "person_tracker",
        "approach_controller",
    }
)

_LEVEL_NAMES = {
    10: "DEBUG",
    20: "INFO",
    30: "WARN",
    40: "ERROR",
    50: "FATAL",
}


def logger_is_selected(
    logger_name: str,
    targets: Iterable[str] = TARGET_LOGGERS,
) -> bool:
    """Return whether a logger belongs to one of the selected ROS nodes."""
    components = {
        component
        for component in logger_name.strip("/").replace("/", ".").split(".")
        if component
    }
    return bool(components.intersection(targets))


def severity_name(level: int) -> str:
    """Return a stable text label for a ROS logging level."""
    return _LEVEL_NAMES.get(int(level), f"LEVEL_{int(level)}")


def format_rosout_message(message: Log) -> str:
    """Format one ROS log message as a single, immediately writable line."""
    seconds = int(message.stamp.sec)
    nanoseconds = int(message.stamp.nanosec)
    try:
        stamp = datetime.fromtimestamp(
            seconds + nanoseconds / 1_000_000_000.0
        ).astimezone().isoformat(timespec="milliseconds")
    except (OSError, OverflowError, ValueError):
        stamp = f"{seconds}.{nanoseconds:09d}"
    text = str(message.msg).replace("\r", "\\r").replace("\n", "\\n")
    return (
        f"{stamp} [{severity_name(message.level)}] "
        f"[{message.name}] {text}"
    )


class ApproachRosoutLogger(Node):
    """Subscribe to /rosout and persist logs from the approach pipeline."""

    def __init__(self, output_file: Path) -> None:
        super().__init__("approach_rosout_logger")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self._stream = output_file.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        qos = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(
            Log,
            "/rosout",
            self._on_log,
            qos,
        )
        self.get_logger().info(
            f"Filtering approach-pipeline /rosout to {output_file}"
        )

    def _on_log(self, message: Log) -> None:
        if not logger_is_selected(message.name):
            return
        self._stream.write(format_rosout_message(message) + "\n")
        self._stream.flush()

    def destroy_node(self):
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        return super().destroy_node()


def _parse_args(args: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Save selected emo_robot /rosout messages as text."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Text file to append selected ROS log messages to.",
    )
    return parser.parse_known_args(args)


def main(args=None) -> None:
    cli_args = list(sys.argv[1:] if args is None else args)
    parsed, ros_args = _parse_args(cli_args)
    rclpy.init(args=ros_args)
    node = ApproachRosoutLogger(parsed.output_file)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Publish frames from a V4L2 USB camera as ROS 2 Image messages."""

import threading
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class UsbCameraNode(Node):
    """Capture a configurable USB camera and publish BGR images."""

    def __init__(self) -> None:
        super().__init__("usb_camera")
        self.declare_parameter("device", "/dev/video11")
        self.declare_parameter("image_topic", "/emo_robot/camera/image_raw")
        self.declare_parameter("frame_id", "usb_camera_optical_frame")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("pixel_format", "YUYV")
        self.declare_parameter("buffer_size", 2)
        self.declare_parameter("reconnect_interval_s", 2.0)

        self._device = str(self.get_parameter("device").value)
        self._image_topic = str(self.get_parameter("image_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._width = int(self.get_parameter("width").value)
        self._height = int(self.get_parameter("height").value)
        self._fps = float(self.get_parameter("fps").value)
        self._pixel_format = str(
            self.get_parameter("pixel_format").value
        ).upper()
        self._buffer_size = int(self.get_parameter("buffer_size").value)
        self._reconnect_interval_s = float(
            self.get_parameter("reconnect_interval_s").value
        )

        if self._fps <= 0.0:
            raise ValueError("fps must be greater than zero")
        if len(self._pixel_format) != 4:
            raise ValueError("pixel_format must be a four-character V4L2 code")
        if self._buffer_size <= 0:
            raise ValueError("buffer_size must be greater than zero")

        self._bridge = CvBridge()
        self._publisher = self.create_publisher(
            Image,
            self._image_topic,
            qos_profile_sensor_data,
        )
        self._capture: Optional[cv2.VideoCapture] = None
        self._next_open_time = 0.0
        self._first_frame_published = False
        self._last_read_warning = 0.0
        self._destroy_started = False
        self._stop_event = threading.Event()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="usb-camera-capture",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Read and publish at the camera's native delivery rate."""
        try:
            while not self._stop_event.is_set():
                if not rclpy.ok():
                    break
                if self._capture is None or not self._capture.isOpened():
                    reopen_delay = self._next_open_time - time.monotonic()
                    if reopen_delay > 0.0:
                        self._stop_event.wait(reopen_delay)
                        continue
                    if not self._open_camera():
                        continue

                self._capture_and_publish()
        finally:
            self._release_camera()

    def _open_camera(self) -> bool:
        """Open the configured V4L2 device and apply capture settings."""
        self._release_camera()
        now = time.monotonic()
        self._next_open_time = now + self._reconnect_interval_s

        capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            self.get_logger().error(
                f"Cannot open USB camera {self._device}; retrying every "
                f"{self._reconnect_interval_s:.1f}s. Check the device path "
                "and whether another process owns the camera."
            )
            return False

        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self._pixel_format),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
        self._capture = capture

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        actual_buffer_size = int(capture.get(cv2.CAP_PROP_BUFFERSIZE))
        self.get_logger().info(
            f"USB camera opened: device={self._device}, "
            f"requested={self._width}x{self._height}@{self._fps:.1f} "
            f"{self._pixel_format}, actual={actual_width}x{actual_height}"
            f"@{actual_fps:.1f}, buffers={actual_buffer_size}; "
            f"publishing {self._image_topic}"
        )
        return True

    def _capture_and_publish(self) -> bool:
        """Read one frame; retry camera opening after recoverable failures."""
        if self._capture is None or not self._capture.isOpened():
            return False

        success, frame = self._capture.read()
        if not success or frame is None:
            now = time.monotonic()
            if now - self._last_read_warning >= self._reconnect_interval_s:
                self.get_logger().warning(
                    f"Failed to read {self._device}; reopening the camera."
                )
                self._last_read_warning = now
            self._next_open_time = now + self._reconnect_interval_s
            self._release_camera()
            return False

        if self._stop_event.is_set():
            return False

        message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        try:
            self._publisher.publish(message)
        except Exception:
            if self._stop_event.is_set() or not rclpy.ok():
                return False
            raise

        if not self._first_frame_published:
            self._first_frame_published = True
            self.get_logger().info(
                f"First USB camera frame published: "
                f"{frame.shape[1]}x{frame.shape[0]} bgr8"
            )
        return True

    def _release_camera(self) -> None:
        """Release the V4L2 handle if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _stop_capture_thread(self) -> None:
        """Stop the capture worker and release its V4L2 handle."""
        self._stop_event.set()
        self._capture_thread.join(
            timeout=max(1.0, self._reconnect_interval_s + 0.5)
        )
        if self._capture_thread.is_alive():
            self.get_logger().warning(
                "USB camera capture thread did not stop in time; "
                "forcing camera release."
            )
            self._release_camera()
            self._capture_thread.join(timeout=1.0)
        else:
            self._release_camera()

    def destroy_node(self):
        """Release the camera before destroying ROS resources."""
        if self._destroy_started:
            return True

        self._destroy_started = True
        self._stop_capture_thread()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

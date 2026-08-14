"""Unit tests for the USB camera capture worker."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from emo_robot_perception import usb_camera_node as camera_module
from emo_robot_perception.usb_camera_node import UsbCameraNode


def _logger():
    return SimpleNamespace(
        debug=Mock(),
        info=Mock(),
        warning=Mock(),
        error=Mock(),
    )


def _capture_node(capture):
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=None, frame_id="")
    )
    bridge = SimpleNamespace(
        cv2_to_imgmsg=Mock(return_value=message)
    )
    publisher = SimpleNamespace(publish=Mock())
    logger = _logger()
    node = SimpleNamespace(
        _capture=capture,
        _stop_event=SimpleNamespace(is_set=lambda: False),
        _bridge=bridge,
        _publisher=publisher,
        _frame_id="usb_camera_optical_frame",
        _first_frame_published=False,
        _last_read_warning=0.0,
        _reconnect_interval_s=2.0,
        _next_open_time=0.0,
        _device="/dev/video11",
        _release_camera=Mock(),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        ),
        get_logger=lambda: logger,
    )
    return node, message, bridge, publisher, logger


def test_initialization_uses_capture_thread_without_ros_timer():
    source = inspect.getsource(UsbCameraNode.__init__)

    assert "threading.Thread" in source
    assert "create_timer" not in source
    assert 'declare_parameter("buffer_size", 2)' in source


def test_successful_read_publishes_one_bgr8_message():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (True, frame),
    )
    node, message, bridge, publisher, logger = _capture_node(capture)

    published = UsbCameraNode._capture_and_publish(node)

    assert published is True
    bridge.cv2_to_imgmsg.assert_called_once_with(frame, encoding="bgr8")
    publisher.publish.assert_called_once_with(message)
    assert message.header.stamp == "stamp"
    assert message.header.frame_id == "usb_camera_optical_frame"
    assert node._first_frame_published is True
    logger.info.assert_called_once()


def test_publish_error_is_ignored_after_ros_shutdown(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (True, frame),
    )
    node, _message, _bridge, publisher, _logger_instance = _capture_node(
        capture
    )
    publisher.publish.side_effect = RuntimeError("context is invalid")
    monkeypatch.setattr(camera_module.rclpy, "ok", lambda: False)

    published = UsbCameraNode._capture_and_publish(node)

    assert published is False


def test_failed_read_releases_camera_and_schedules_reconnect(monkeypatch):
    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (False, None),
    )
    node, _message, _bridge, publisher, logger = _capture_node(capture)
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: 10.0)

    published = UsbCameraNode._capture_and_publish(node)

    assert published is False
    assert node._next_open_time == 12.0
    node._release_camera.assert_called_once()
    publisher.publish.assert_not_called()
    logger.warning.assert_called_once()


class _StopAfterWait:
    def __init__(self):
        self.stopped = False
        self.wait_calls = []

    def is_set(self):
        return self.stopped

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        self.stopped = True
        return True


def test_reconnect_wait_is_interruptible_and_does_not_busy_loop(monkeypatch):
    stop_event = _StopAfterWait()
    monkeypatch.setattr(camera_module.rclpy, "ok", lambda: True)
    node = SimpleNamespace(
        _stop_event=stop_event,
        _capture=None,
        _next_open_time=5.0,
        _open_camera=Mock(),
        _capture_and_publish=Mock(),
        _release_camera=Mock(),
    )
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: 2.0)

    UsbCameraNode._capture_loop(node)

    assert stop_event.wait_calls == [3.0]
    node._open_camera.assert_not_called()
    node._capture_and_publish.assert_not_called()
    node._release_camera.assert_called_once()


def test_open_camera_applies_configured_buffer_size(monkeypatch):
    capture = SimpleNamespace(
        isOpened=lambda: True,
        release=Mock(),
        set=Mock(return_value=True),
    )
    values = {
        camera_module.cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        camera_module.cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        camera_module.cv2.CAP_PROP_FPS: 30.0,
        camera_module.cv2.CAP_PROP_BUFFERSIZE: 2.0,
    }
    capture.get = lambda prop: values.get(prop, 0.0)
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda _device, _backend: capture,
    )
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: 5.0)
    logger = _logger()
    node = SimpleNamespace(
        _capture=None,
        _device="/dev/video11",
        _pixel_format="YUYV",
        _width=640,
        _height=480,
        _fps=30.0,
        _buffer_size=2,
        _image_topic="/emo_robot/camera/image_raw",
        _reconnect_interval_s=2.0,
        _next_open_time=0.0,
        _release_camera=Mock(),
        get_logger=lambda: logger,
    )

    assert UsbCameraNode._open_camera(node) is True
    capture.set.assert_any_call(camera_module.cv2.CAP_PROP_BUFFERSIZE, 2)
    assert node._capture is capture
    logger.info.assert_called_once()


def test_release_camera_is_idempotent():
    capture = SimpleNamespace(release=Mock())
    node = SimpleNamespace(_capture=capture)

    UsbCameraNode._release_camera(node)
    UsbCameraNode._release_camera(node)

    capture.release.assert_called_once()
    assert node._capture is None


def test_stop_capture_thread_signals_joins_and_releases():
    stop_event = SimpleNamespace(set=Mock())
    capture_thread = SimpleNamespace(
        join=Mock(),
        is_alive=Mock(return_value=False),
    )
    node = SimpleNamespace(
        _stop_event=stop_event,
        _capture_thread=capture_thread,
        _reconnect_interval_s=2.0,
        _release_camera=Mock(),
        get_logger=lambda: _logger(),
    )

    UsbCameraNode._stop_capture_thread(node)

    stop_event.set.assert_called_once()
    capture_thread.join.assert_called_once_with(timeout=2.5)
    node._release_camera.assert_called_once()


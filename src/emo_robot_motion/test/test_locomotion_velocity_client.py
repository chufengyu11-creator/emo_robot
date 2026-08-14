"""Unit tests for the locomotion velocity client."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _install_ros_stubs() -> None:
    if "rclpy" in sys.modules:
        return

    rclpy = ModuleType("rclpy")
    rclpy_node = ModuleType("rclpy.node")
    rclpy_node.Node = object
    rclpy.node = rclpy_node

    aimdk_msgs = ModuleType("aimdk_msgs")
    aimdk_msgs_msg = ModuleType("aimdk_msgs.msg")
    aimdk_msgs_srv = ModuleType("aimdk_msgs.srv")

    class _CommonState:
        RUNNING = 1

    class _McInputAction:
        INPUTACTION_ADD = 1001
        INPUTACTION_DELETE = 1003

        def __init__(self):
            self.value = 0

    class _ValueMessage:
        def __init__(self, value=0):
            self.value = value

    class _MessageHeader:
        def __init__(self):
            self.stamp = None

    class _RequestHeader:
        def __init__(self):
            self.stamp = None

    class _McLocomotionVelocity:
        def __init__(self):
            self.header = None
            self.source = ""
            self.forward_velocity = 0.0
            self.lateral_velocity = 0.0
            self.angular_velocity = 0.0

    class _SetMcInputSourceRequest:
        def __init__(self):
            self.request = SimpleNamespace(
                header=SimpleNamespace(stamp=None),
            )
            self.action = _McInputAction()
            self.input_source = SimpleNamespace(
                name="",
                priority=0,
                timeout=0,
            )

    class _SetMcInputSource:
        Request = _SetMcInputSourceRequest

    class _SetMcPresetMotionRequest:
        def __init__(self):
            self.header = None
            self.motion = None
            self.area = None
            self.interrupt = False

    class _SetMcPresetMotion:
        Request = _SetMcPresetMotionRequest

    aimdk_msgs_msg.CommonState = _CommonState
    aimdk_msgs_msg.McControlArea = _ValueMessage
    aimdk_msgs_msg.McInputAction = _McInputAction
    aimdk_msgs_msg.McLocomotionVelocity = _McLocomotionVelocity
    aimdk_msgs_msg.McPresetMotion = _ValueMessage
    aimdk_msgs_msg.MessageHeader = _MessageHeader
    aimdk_msgs_msg.RequestHeader = _RequestHeader
    aimdk_msgs_srv.SetMcInputSource = _SetMcInputSource
    aimdk_msgs_srv.SetMcPresetMotion = _SetMcPresetMotion
    aimdk_msgs.msg = aimdk_msgs_msg
    aimdk_msgs.srv = aimdk_msgs_srv

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["aimdk_msgs"] = aimdk_msgs
    sys.modules["aimdk_msgs.msg"] = aimdk_msgs_msg
    sys.modules["aimdk_msgs.srv"] = aimdk_msgs_srv


_install_ros_stubs()

from emo_robot_motion.locomotion_velocity_client import (  # noqa: E402
    LocomotionVelocityClient,
)


class _Timer:
    def __init__(self, period, callback):
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Future:
    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception
        self.cancelled = False

    def done(self):
        return True

    def cancel(self):
        self.cancelled = True

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, callback):
        callback(self)


class _Publisher:
    def __init__(self):
        self.publish = Mock()


class _Client:
    def __init__(self, ready_sequence=None, future=None):
        self._ready_sequence = list(
            ready_sequence if ready_sequence is not None else [True]
        )
        self._future = future or _Future(_success_response())
        self.call_async = Mock(side_effect=self._call_async)
        self.last_request = None

    def service_is_ready(self):
        if len(self._ready_sequence) > 1:
            return self._ready_sequence.pop(0)
        return self._ready_sequence[0]

    def _call_async(self, request):
        self.last_request = request
        return self._future


class _Node:
    def __init__(self, client=None):
        self.publisher = _Publisher()
        self.client = client or _Client()
        self.timers = []
        self.destroyed_timers = []
        self.logger = Mock()

    def create_publisher(self, *_args):
        return self.publisher

    def create_client(self, *_args):
        return self.client

    def create_timer(self, period, callback):
        timer = _Timer(period, callback)
        self.timers.append(timer)
        return timer

    def destroy_timer(self, timer):
        self.destroyed_timers.append(timer)
        if timer in self.timers:
            self.timers.remove(timer)

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp"),
        )

    def get_logger(self):
        return self.logger


def _success_response(code=0, state=0):
    return SimpleNamespace(
        response=SimpleNamespace(
            header=SimpleNamespace(code=code),
            state=SimpleNamespace(value=state),
            task_id="task-1",
        ),
    )


def test_constructor_does_not_publish_or_create_timer():
    node = _Node()

    LocomotionVelocityClient(node)

    node.publisher.publish.assert_not_called()
    assert node.timers == []


def test_register_input_source_uses_async_request_and_timer_retry():
    client = _Client(ready_sequence=[False, True])
    node = _Node(client=client)
    done = Mock()
    locomotion = LocomotionVelocityClient(node)

    locomotion.register_input_source_async(done_callback=done)
    assert client.call_async.call_count == 0
    assert len(node.timers) == 1

    node.timers[0].callback()

    request = client.last_request
    assert request.action.value == 1001
    assert request.input_source.name == "node"
    assert request.input_source.priority == 40
    assert request.input_source.timeout == 1000
    assert request.request.header.stamp == "stamp"
    assert locomotion.registered is True
    done.assert_called_once_with(True)


def test_unregister_input_source_uses_delete_and_is_idempotent():
    node = _Node()
    done = Mock()
    locomotion = LocomotionVelocityClient(
        node,
        source_name="emo_robot_locomotion_test",
    )
    locomotion.register_input_source_async()
    node.client.call_async.reset_mock()

    locomotion.unregister_input_source_async(done_callback=done)

    request = node.client.last_request
    assert request.action.value == 1003
    assert request.input_source.name == "emo_robot_locomotion_test"
    assert request.input_source.priority == 40
    assert request.input_source.timeout == 1000
    assert locomotion.registered is False
    done.assert_called_once_with(True)

    locomotion.unregister_input_source_async(done_callback=done)
    assert node.client.call_async.call_count == 1
    assert done.call_count == 2


def test_start_publishing_creates_timer_after_registration():
    node = _Node()
    locomotion = LocomotionVelocityClient(node)
    locomotion.register_input_source_async()

    locomotion.start_publishing()

    assert len(node.timers) == 1
    assert node.timers[0].period == 0.02


def test_set_velocity_validates_and_publishes_registered_values():
    node = _Node()
    locomotion = LocomotionVelocityClient(node)
    locomotion.register_input_source_async()
    locomotion.set_velocity(0.2, -0.3, 0.1)

    locomotion._publish_velocity()

    message = node.publisher.publish.call_args.args[0]
    assert message.source == "node"
    assert message.forward_velocity == 0.2
    assert message.lateral_velocity == -0.3
    assert message.angular_velocity == 0.1


def test_unregistered_publish_forces_zero_velocity():
    node = _Node()
    locomotion = LocomotionVelocityClient(node)
    locomotion.set_velocity(0.2, -0.3, 0.1)

    locomotion._publish_velocity()

    message = node.publisher.publish.call_args.args[0]
    assert message.forward_velocity == 0.0
    assert message.lateral_velocity == 0.0
    assert message.angular_velocity == 0.0


def test_invalid_velocity_raises_value_error():
    locomotion = LocomotionVelocityClient(_Node())

    with pytest.raises(ValueError):
        locomotion.set_velocity(0.1, 0.0, 0.0)
    with pytest.raises(ValueError):
        locomotion.set_velocity(0.0, 1.1, 0.0)
    with pytest.raises(ValueError):
        locomotion.set_velocity(0.0, 0.0, 0.05)
    with pytest.raises(ValueError):
        locomotion.set_velocity(float("nan"), 0.0, 0.0)


def test_stop_publishes_one_zero_frame():
    node = _Node()
    locomotion = LocomotionVelocityClient(node)
    locomotion.register_input_source_async()
    locomotion.set_velocity(0.2, 0.0, 0.1)

    locomotion.stop()

    assert node.publisher.publish.call_count == 1
    message = node.publisher.publish.call_args.args[0]
    assert message.forward_velocity == 0.0
    assert message.lateral_velocity == 0.0
    assert message.angular_velocity == 0.0


def test_emergency_stop_publishes_zero_frames_and_cancels_timer():
    node = _Node()
    locomotion = LocomotionVelocityClient(node)
    locomotion.register_input_source_async()
    locomotion.start_publishing()

    locomotion.emergency_stop()

    assert node.publisher.publish.call_count == 3
    assert node.timers == []
    assert node.destroyed_timers
    for call in node.publisher.publish.call_args_list:
        message = call.args[0]
        assert message.forward_velocity == 0.0
        assert message.lateral_velocity == 0.0
        assert message.angular_velocity == 0.0

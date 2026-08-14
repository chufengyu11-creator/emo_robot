"""Unit tests for the reusable GetMcAction state client."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


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

    class _ValueMessage:
        def __init__(self, value=0):
            self.value = value

    class _CommonRequest:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    class _CommonState:
        RUNNING = 1

    class _McActionStatus:
        RUNNING = 100
        TRANSITION = 200

    class _McInputAction:
        INPUTACTION_ADD = 1001
        INPUTACTION_DELETE = 1003

        def __init__(self):
            self.value = 0

    class _MessageHeader:
        def __init__(self):
            self.stamp = None

    class _McLocomotionVelocity:
        def __init__(self):
            self.header = None
            self.source = ""
            self.forward_velocity = 0.0
            self.lateral_velocity = 0.0
            self.angular_velocity = 0.0

    class _GetMcAction:
        class Request:
            def __init__(self):
                self.request = _CommonRequest()

    class _SetMcInputSourceRequest:
        def __init__(self):
            self.request = _CommonRequest()
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

    aimdk_msgs_msg.CommonRequest = _CommonRequest
    aimdk_msgs_msg.CommonState = _CommonState
    aimdk_msgs_msg.McActionStatus = _McActionStatus
    aimdk_msgs_msg.McControlArea = _ValueMessage
    aimdk_msgs_msg.McInputAction = _McInputAction
    aimdk_msgs_msg.McLocomotionVelocity = _McLocomotionVelocity
    aimdk_msgs_msg.McPresetMotion = _ValueMessage
    aimdk_msgs_msg.MessageHeader = _MessageHeader
    aimdk_msgs_msg.RequestHeader = _MessageHeader
    aimdk_msgs_srv.GetMcAction = _GetMcAction
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

from emo_robot_motion.mc_action_state_client import (  # noqa: E402
    McActionState,
    McActionStateClient,
)


class _Timer:
    def __init__(self, period, callback):
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Future:
    def __init__(self, result=None, exception=None, complete=True):
        self._result = result
        self._exception = exception
        self._complete = complete
        self._callbacks = []
        self.cancelled = False

    def done(self):
        return self._complete

    def cancel(self):
        self.cancelled = True
        self._complete = True

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, callback):
        self._callbacks.append(callback)
        if self._complete:
            callback(self)

    def fire_late(self):
        self._complete = True
        for callback in list(self._callbacks):
            callback(self)


class _Client:
    def __init__(self, ready_sequence=None, futures=None):
        self._ready_sequence = list(
            ready_sequence if ready_sequence is not None else [True]
        )
        self._futures = list(
            futures if futures is not None else [_Future(_response())]
        )
        self.call_async = Mock(side_effect=self._call_async)
        self.last_request = None

    def service_is_ready(self):
        if len(self._ready_sequence) > 1:
            return self._ready_sequence.pop(0)
        return self._ready_sequence[0]

    def _call_async(self, request):
        self.last_request = request
        if len(self._futures) > 1:
            return self._futures.pop(0)
        return self._futures[0]


class _Node:
    def __init__(self, client=None):
        self.client = client or _Client()
        self.timers = []
        self.destroyed_timers = []
        self.logger = Mock()

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


def _response(
    action=300,
    description="LOCOMOTION_DEFAULT",
    status=100,
    code=0,
):
    return SimpleNamespace(
        header=SimpleNamespace(code=code),
        info=SimpleNamespace(
            current_action=SimpleNamespace(value=action),
            action_desc=description,
            status=SimpleNamespace(value=status),
        ),
    )


def test_success_parses_state_and_stamps_request():
    node = _Node()
    done = Mock()
    client = McActionStateClient(node)

    client.get_state_async(done)

    state = done.call_args.args[1]
    assert done.call_args.args == (True, state, "")
    assert state == McActionState(300, "LOCOMOTION_DEFAULT", 100, 0)
    assert state.is_running(300) is True
    assert state.is_running(200) is False
    assert node.client.last_request.request.header.stamp == "stamp"
    assert client.active is False


def test_service_not_ready_retries_then_succeeds():
    ros_client = _Client(ready_sequence=[False, True])
    node = _Node(ros_client)
    done = Mock()
    client = McActionStateClient(node)

    client.get_state_async(done)
    assert ros_client.call_async.call_count == 0
    assert len(node.timers) == 1

    node.timers[0].callback()

    assert ros_client.call_async.call_count == 1
    done.assert_called_once()
    assert done.call_args.args[0] is True


def test_retry_exhaustion_reports_failure_once():
    node = _Node(_Client(ready_sequence=[False]))
    done = Mock()
    client = McActionStateClient(node, max_attempts=2)

    client.get_state_async(done)
    node.timers[0].callback()
    node.timers[0].callback()

    done.assert_called_once()
    success, state, error = done.call_args.args
    assert success is False
    assert state is None
    assert "retries exhausted" in error


def test_timeout_cancels_future_and_reports_exhaustion():
    future = _Future(_response(), complete=False)
    node = _Node(_Client(futures=[future]))
    done = Mock()
    client = McActionStateClient(node, max_attempts=1)

    client.get_state_async(done)
    node.timers[0].callback()

    assert future.cancelled is True
    done.assert_called_once()
    assert done.call_args.args[0] is False
    assert "no response" in done.call_args.args[2]


def test_response_exception_is_retried():
    failed = _Future(exception=RuntimeError("transport error"))
    succeeded = _Future(_response())
    node = _Node(_Client(futures=[failed, succeeded]))
    done = Mock()
    client = McActionStateClient(node, max_attempts=2)

    client.get_state_async(done)

    done.assert_called_once()
    assert done.call_args.args[0] is True
    assert node.client.call_async.call_count == 2


def test_nonzero_response_code_returns_state_and_error():
    node = _Node(_Client(futures=[_Future(_response(code=7))]))
    done = Mock()
    client = McActionStateClient(node)

    client.get_state_async(done)

    success, state, error = done.call_args.args
    assert success is False
    assert state.response_code == 7
    assert state.is_running(300) is False
    assert "code=7" in error


def test_cancel_ignores_late_response_and_is_idempotent():
    future = _Future(_response(), complete=False)
    node = _Node(_Client(futures=[future]))
    done = Mock()
    client = McActionStateClient(node)

    client.get_state_async(done)
    client.cancel()
    client.cancel()
    future.fire_late()

    done.assert_not_called()
    assert future.cancelled is True
    assert client.active is False
    assert node.timers == []

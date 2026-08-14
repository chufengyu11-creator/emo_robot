"""Unit tests for the safety-gated locomotion test node."""

from __future__ import annotations

import math
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _install_ros_stubs() -> None:
    rclpy = ModuleType("rclpy")
    rclpy.init = Mock()
    rclpy.ok = Mock(return_value=True)
    rclpy.spin_once = Mock()
    rclpy.shutdown = Mock()
    rclpy_node = ModuleType("rclpy.node")

    class _NodeBase:
        def __init__(self, *_args, **_kwargs):
            pass

    rclpy_node.Node = _NodeBase
    rclpy_signals = ModuleType("rclpy.signals")
    rclpy_signals.SignalHandlerOptions = SimpleNamespace(NO=0)
    rclpy.node = rclpy_node
    rclpy.signals = rclpy_signals

    aimdk_msgs = ModuleType("aimdk_msgs")
    aimdk_msgs_msg = ModuleType("aimdk_msgs.msg")
    aimdk_msgs_srv = ModuleType("aimdk_msgs.srv")

    class _CommonState:
        RUNNING = 1

    class _McAction:
        STAND_DEFAULT = 200
        LOCOMOTION_DEFAULT = 300

    class _McActionStatus:
        RUNNING = 100
        TRANSITION = 200

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

    class _CommonRequest:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    class _McLocomotionVelocity:
        def __init__(self):
            self.header = None
            self.source = ""
            self.forward_velocity = 0.0
            self.lateral_velocity = 0.0
            self.angular_velocity = 0.0

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

    class _GetMcAction:
        class Request:
            def __init__(self):
                self.request = _CommonRequest()

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
    aimdk_msgs_msg.McAction = _McAction
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
    sys.modules["rclpy.signals"] = rclpy_signals
    sys.modules["aimdk_msgs"] = aimdk_msgs
    sys.modules["aimdk_msgs.msg"] = aimdk_msgs_msg
    sys.modules["aimdk_msgs.srv"] = aimdk_msgs_srv


_install_ros_stubs()

from emo_robot_motion.locomotion_test_node import (  # noqa: E402
    LocomotionTestCommand,
    LocomotionTestNode,
)
from emo_robot_motion.mc_action_state_client import (  # noqa: E402
    McActionState,
)


def _bare_node(motion=None):
    node = object.__new__(LocomotionTestNode)
    node._finished = False
    node._cleanup_started = False
    node._cleanup_success = True
    node._mode_state_client = Mock()
    node._motion_timer = None
    node._motion = motion
    node.get_logger = lambda: Mock()
    node.destroy_timer = Mock()
    return node


def _motion(registered=False, registering=False):
    motion = Mock()
    motion.registered = registered
    motion.registering = registering
    motion.unregister_input_source_async.side_effect = (
        lambda done_callback=None: done_callback(True)
        if done_callback
        else None
    )
    return motion


def test_safe_defaults_are_locked_and_valid():
    command = LocomotionTestCommand()

    assert command.validated_velocities() == (0.0, 0.0, 0.0)
    assert command.motion_enabled is False


def test_disabled_node_creates_no_clients_or_publishers(monkeypatch):
    values = {
        "motion_enabled": False,
        "confirmation": "",
        "forward_velocity": 0.0,
        "lateral_velocity": 0.0,
        "angular_velocity": 0.0,
        "duration_s": 2.0,
    }
    create_client = Mock()
    create_publisher = Mock()
    monkeypatch.setattr(
        LocomotionTestNode,
        "declare_parameter",
        lambda *_args: None,
        raising=False,
    )
    monkeypatch.setattr(
        LocomotionTestNode,
        "get_parameter",
        lambda _self, name: SimpleNamespace(value=values[name]),
        raising=False,
    )
    monkeypatch.setattr(
        LocomotionTestNode,
        "get_logger",
        lambda _self: Mock(),
        raising=False,
    )
    monkeypatch.setattr(
        LocomotionTestNode,
        "create_client",
        create_client,
        raising=False,
    )
    monkeypatch.setattr(
        LocomotionTestNode,
        "create_publisher",
        create_publisher,
        raising=False,
    )

    node = LocomotionTestNode()

    assert node.finished is True
    create_client.assert_not_called()
    create_publisher.assert_not_called()


def test_motion_requires_exact_confirmation():
    with pytest.raises(ValueError, match="I_UNDERSTAND"):
        LocomotionTestCommand(
            motion_enabled=True,
            forward_velocity=0.2,
        ).validated_velocities()


def test_only_one_nonzero_axis_is_allowed():
    with pytest.raises(ValueError, match="only one"):
        LocomotionTestCommand(
            motion_enabled=True,
            confirmation="I_UNDERSTAND",
            forward_velocity=0.2,
            angular_velocity=0.1,
        ).validated_velocities()


@pytest.mark.parametrize(
    "command",
    [
        LocomotionTestCommand(duration_s=0.0),
        LocomotionTestCommand(duration_s=5.1),
        LocomotionTestCommand(forward_velocity=0.1),
        LocomotionTestCommand(lateral_velocity=1.1),
        LocomotionTestCommand(angular_velocity=0.05),
        LocomotionTestCommand(forward_velocity=math.inf),
    ],
)
def test_duration_and_velocity_bounds_are_enforced(command):
    with pytest.raises(ValueError):
        command.validated_velocities()


@pytest.mark.parametrize(
    ("action", "description"),
    [
        (200, "STAND_DEFAULT"),
        (300, "LOCOMOTION_DEFAULT"),
    ],
)
def test_supported_running_mode_registers_input_source(
    action,
    description,
):
    motion = _motion()
    node = _bare_node(motion)
    state = McActionState(
        action=action,
        description=description,
        status=100,
        response_code=0,
    )

    node._on_mode_state(True, state, "")

    motion.register_input_source_async.assert_called_once_with(
        done_callback=node._on_input_source_registered,
    )


@pytest.mark.parametrize(
    ("action", "status"),
    [
        (100, 100),
        (200, 200),
        (300, 200),
    ],
)
def test_unsupported_or_transition_mode_refuses_motion(action, status):
    motion = _motion()
    node = _bare_node(motion)
    node._abort = Mock()
    state = McActionState(
        action=action,
        description="not ready",
        status=status,
        response_code=0,
    )

    node._on_mode_state(True, state, "")

    node._abort.assert_called_once()
    motion.register_input_source_async.assert_not_called()


def test_mode_query_failure_never_registers_input_source():
    motion = _motion()
    node = _bare_node(motion)
    node._abort = Mock()

    node._on_mode_state(False, None, "GetMcAction retries exhausted")

    node._abort.assert_called_once_with("GetMcAction retries exhausted")
    motion.register_input_source_async.assert_not_called()


def test_registration_failure_stops_without_starting_publisher():
    motion = _motion()
    node = _bare_node(motion)

    node._on_input_source_registered(False)

    motion.start_publishing.assert_not_called()
    motion.emergency_stop.assert_called_once()
    assert node.finished is True
    assert node._cleanup_success is False


def test_success_starts_50hz_client_and_duration_timer():
    motion = _motion(registered=True)
    node = _bare_node(motion)
    node._velocities = (0.2, 0.0, 0.0)
    node._command = LocomotionTestCommand(
        motion_enabled=True,
        confirmation="I_UNDERSTAND",
        forward_velocity=0.2,
        duration_s=2.0,
    )
    timer = Mock()
    node.create_timer = Mock(return_value=timer)

    node._on_input_source_registered(True)

    motion.set_velocity.assert_called_once_with(0.2, 0.0, 0.0)
    motion.start_publishing.assert_called_once()
    node.create_timer.assert_called_once_with(2.0, node._on_motion_timeout)
    assert node._motion_timer is timer


def test_duration_stop_zeros_unregisters_and_is_idempotent():
    motion = _motion(registered=True)
    node = _bare_node(motion)

    node.request_stop("duration")
    node.request_stop("duplicate")

    motion.emergency_stop.assert_called_once()
    motion.unregister_input_source_async.assert_called_once()
    assert node.finished is True
    assert node._cleanup_success is True


def test_stop_during_registration_waits_then_unregisters():
    motion = _motion(registering=True)
    node = _bare_node(motion)

    node.request_stop("signal 2")
    assert node.finished is False

    motion.registering = False
    motion.registered = True
    node._on_input_source_registered(True)

    assert motion.emergency_stop.call_count == 2
    motion.unregister_input_source_async.assert_called_once()
    assert node.finished is True

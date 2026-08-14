"""Unit tests for gesture-specific interaction responses."""

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock


def _install_ros_stubs() -> None:
    if "rclpy" in sys.modules:
        return

    rclpy = ModuleType("rclpy")
    rclpy.init = Mock()
    rclpy.spin = Mock()
    rclpy.ok = Mock(return_value=False)
    rclpy.shutdown = Mock()
    rclpy_node = ModuleType("rclpy.node")
    rclpy_node.Node = object
    rclpy.node = rclpy_node

    aimdk_msgs = ModuleType("aimdk_msgs")
    aimdk_msgs_msg = ModuleType("aimdk_msgs.msg")
    aimdk_msgs_srv = ModuleType("aimdk_msgs.srv")

    class _ValueMessage:
        def __init__(self, value=0):
            self.value = value

    class _Header:
        def __init__(self):
            self.stamp = None

    class _RequestHeader:
        def __init__(self):
            self.stamp = None

    class _CommonState:
        RUNNING = 1

    class _PresetMotionRequest:
        def __init__(self):
            self.header = None
            self.motion = None
            self.area = None
            self.interrupt = False

    class _SetMcPresetMotion:
        Request = _PresetMotionRequest

    class _PlayTtsRequest:
        def __init__(self):
            self.tts_req = SimpleNamespace(
                text="",
                domain="",
                trace_id="",
                is_interrupted=False,
                priority_weight=0,
                priority_level=SimpleNamespace(value=0),
            )
            self.header = SimpleNamespace(header=_Header())

    class _PlayTts:
        Request = _PlayTtsRequest

    aimdk_msgs_msg.CommonState = _CommonState
    aimdk_msgs_msg.McControlArea = _ValueMessage
    aimdk_msgs_msg.McPresetMotion = _ValueMessage
    aimdk_msgs_msg.RequestHeader = _RequestHeader
    aimdk_msgs_srv.SetMcPresetMotion = _SetMcPresetMotion
    aimdk_msgs_srv.PlayTts = _PlayTts
    aimdk_msgs.msg = aimdk_msgs_msg
    aimdk_msgs.srv = aimdk_msgs_srv

    interfaces = ModuleType("emo_robot_interfaces")
    interfaces_msg = ModuleType("emo_robot_interfaces.msg")

    class _GestureDetection:
        GESTURE_NONE = 0
        GESTURE_WAVE = 1
        GESTURE_COME = 2

        def __init__(self):
            self.gesture = self.GESTURE_NONE
            self.valid = False
            self.confidence = 0.0
            self.track_id = 0

    interfaces_msg.GestureDetection = _GestureDetection
    interfaces.msg = interfaces_msg

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["aimdk_msgs"] = aimdk_msgs
    sys.modules["aimdk_msgs.msg"] = aimdk_msgs_msg
    sys.modules["aimdk_msgs.srv"] = aimdk_msgs_srv
    sys.modules["emo_robot_interfaces"] = interfaces
    sys.modules["emo_robot_interfaces.msg"] = interfaces_msg


_install_ros_stubs()

from emo_robot_control.interaction_responder_node import (
    InteractionResponderNode,
)
from emo_robot_interfaces.msg import GestureDetection


class _Clock:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def now(self):
        return SimpleNamespace(
            nanoseconds=int(self.seconds * 1e9),
        )


def _node(seconds: float = 100.0):
    logger = Mock()
    return SimpleNamespace(
        _tts_text_by_gesture={
            GestureDetection.GESTURE_WAVE: "wave text",
            GestureDetection.GESTURE_COME: "come text",
        },
        _tts_domain="emo_robot",
        _response_cooldown_s=5.0,
        _response_in_flight=False,
        _active_gesture=None,
        _last_response_time={
            GestureDetection.GESTURE_WAVE: -1e9,
            GestureDetection.GESTURE_COME: -1e9,
        },
        _speech=SimpleNamespace(say_async=Mock()),
        _on_tts_done=Mock(),
        _motion_enabled=False,
        _motion_area=2,
        _motion_id=1002,
        _motion=None,
        _on_motion_done=Mock(),
        _gesture_name=InteractionResponderNode._gesture_name,
        get_clock=lambda: _Clock(seconds),
        get_logger=lambda: logger,
    )


def _message(gesture: int):
    message = GestureDetection()
    message.gesture = gesture
    message.valid = True
    message.confidence = 1.2
    message.track_id = 4
    return message


def test_wave_and_come_use_distinct_text_and_cooldowns():
    node = _node()

    InteractionResponderNode._on_gesture(
        node,
        _message(GestureDetection.GESTURE_WAVE),
    )
    assert node._speech.say_async.call_args.args[0] == "wave text"
    InteractionResponderNode._on_tts_done(node, True)

    InteractionResponderNode._on_gesture(
        node,
        _message(GestureDetection.GESTURE_COME),
    )
    assert node._speech.say_async.call_count == 2
    assert node._speech.say_async.call_args.args[0] == "come text"
    InteractionResponderNode._on_tts_done(node, True)

    InteractionResponderNode._on_gesture(
        node,
        _message(GestureDetection.GESTURE_WAVE),
    )
    assert node._speech.say_async.call_count == 2


def test_come_never_calls_preset_motion():
    node = _node()
    node._motion_enabled = True
    node._motion = SimpleNamespace(play_async=Mock())

    InteractionResponderNode._on_gesture(
        node,
        _message(GestureDetection.GESTURE_COME),
    )

    assert node._speech.say_async.call_args.args[0] == "come text"
    node._motion.play_async.assert_not_called()


def test_wave_starts_tts_and_preset_motion_in_parallel():
    node = _node()
    node._motion_enabled = True
    node._motion = SimpleNamespace(play_async=Mock())

    InteractionResponderNode._on_gesture(
        node,
        _message(GestureDetection.GESTURE_WAVE),
    )

    assert node._speech.say_async.call_args.args[0] == "wave text"
    node._motion.play_async.assert_called_once_with(
        area=node._motion_area,
        motion_id=node._motion_id,
        interrupt=False,
        done_callback=node._on_motion_done,
    )


def test_motion_disabled_does_not_require_motion_client():
    node = _node()
    node._motion_enabled = False
    node._motion = None

    InteractionResponderNode._on_gesture(
        node,
        _message(GestureDetection.GESTURE_WAVE),
    )

    assert node._motion is None
    assert node._speech.say_async.call_args.args[0] == "wave text"


def test_tts_done_no_longer_starts_preset_motion():
    node = _node()
    node._motion_enabled = True
    node._motion = SimpleNamespace(play_async=Mock())
    node._response_in_flight = True
    node._active_gesture = GestureDetection.GESTURE_WAVE

    InteractionResponderNode._on_tts_done(node, True)

    node._motion.play_async.assert_not_called()
    assert node._response_in_flight is False

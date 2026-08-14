#!/usr/bin/env python3

"""Interaction responder — wave → TTS + preset motion.

Subscribes to ``/emo_robot/perception/gesture_detection`` and, when a
valid WAVE gesture is detected, speaks a configurable phrase via the
robot's TTS engine and optionally performs a preset arm motion.

Safety:  motion is **disabled by default**.  Enable it only after
validating that TTS alone works correctly on the real robot.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from emo_robot_interfaces.msg import GestureDetection
from emo_robot_motion import PresetMotionClient
from emo_robot_speech.speech_client import SpeechClient


class InteractionResponderNode(Node):
    """WAVE/COME → gesture-specific TTS; optional WAVE motion only."""

    def __init__(self) -> None:
        super().__init__("interaction_responder")

        # -- parameters -------------------------------------------------
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter(
            "gesture_topic",
            "/emo_robot/perception/gesture_detection",
        )
        self.declare_parameter(
            "wave_tts_text",
            "你好，你在向我招手吗？",
        )
        self.declare_parameter(
            "come_tts_text",
            "你好，你是在叫我过去吗？",
        )
        self.declare_parameter("tts_domain", "emo_robot")
        self.declare_parameter("motion_area", 2)
        self.declare_parameter("motion_id", 1002)
        self.declare_parameter("response_cooldown_s", 5.0)

        self._motion_enabled = (
            self.get_parameter("motion_enabled")
            .get_parameter_value()
            .bool_value
        )
        gesture_topic = (
            self.get_parameter("gesture_topic")
            .get_parameter_value()
            .string_value
        )
        wave_tts_text = str(self.get_parameter("wave_tts_text").value)
        come_tts_text = str(self.get_parameter("come_tts_text").value)
        tts_domain = (
            self.get_parameter("tts_domain")
            .get_parameter_value()
            .string_value
        )
        self._motion_area = (
            self.get_parameter("motion_area")
            .get_parameter_value()
            .integer_value
        )
        self._motion_id = (
            self.get_parameter("motion_id")
            .get_parameter_value()
            .integer_value
        )
        response_cooldown_s = (
            self.get_parameter("response_cooldown_s")
            .get_parameter_value()
            .double_value
        )

        self._tts_text_by_gesture = {
            GestureDetection.GESTURE_WAVE: wave_tts_text,
            GestureDetection.GESTURE_COME: come_tts_text,
        }
        self._tts_domain = tts_domain
        self._response_cooldown_s = response_cooldown_s
        self._response_in_flight = False
        self._active_gesture = None

        # -- speech client ----------------------------------------------
        self._speech = SpeechClient(self)

        # -- preset-motion client ---------------------------------------
        self._motion = None
        if self._motion_enabled:
            self._motion = PresetMotionClient(self)

        # -- subscription -----------------------------------------------
        self.create_subscription(
            GestureDetection,
            gesture_topic,
            self._on_gesture,
            10,
        )

        self._last_response_time = {
            GestureDetection.GESTURE_WAVE: -1e9,
            GestureDetection.GESTURE_COME: -1e9,
        }

        mode = "TTS + motion" if self._motion_enabled else "TTS only"
        self.get_logger().info(
            f"Interaction responder ready ({mode}) "
            f"on {gesture_topic}"
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    @staticmethod
    def _gesture_name(gesture: int) -> str:
        if gesture == GestureDetection.GESTURE_WAVE:
            return "WAVE"
        if gesture == GestureDetection.GESTURE_COME:
            return "COME"
        return "UNKNOWN"

    def _on_gesture(self, msg: GestureDetection) -> None:
        if not msg.valid or msg.gesture not in self._tts_text_by_gesture:
            return

        gesture = int(msg.gesture)
        gesture_name = self._gesture_name(gesture)
        if self._response_in_flight:
            self.get_logger().debug(
                f"Ignoring {gesture_name} while a response is in flight."
            )
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if (
            now - self._last_response_time[gesture]
            < self._response_cooldown_s
        ):
            return

        self._last_response_time[gesture] = now
        self._response_in_flight = True
        self._active_gesture = gesture
        self.get_logger().info(
            f"Responding to {gesture_name}  "
            f"score={msg.confidence:.2f} track_id={msg.track_id}"
        )

        self._speech.say_async(
            self._tts_text_by_gesture[gesture],
            domain=self._tts_domain,
            done_callback=self._on_tts_done,
        )
        if (
            self._motion_enabled
            and gesture == GestureDetection.GESTURE_WAVE
            and self._motion is not None
        ):
            self.get_logger().info("TTS + preset motion started for WAVE.")
            self._motion.play_async(
                area=self._motion_area,
                motion_id=self._motion_id,
                interrupt=False,
                done_callback=self._on_motion_done,
            )

    def _on_tts_done(self, success: bool) -> None:
        """Finish TTS and release the response guard."""
        self._active_gesture = None
        self._response_in_flight = False
        if success:
            self.get_logger().info("TTS completed successfully.")
        else:
            self.get_logger().warning("TTS did not complete.")

    def _on_motion_done(self, success: bool) -> None:
        if success:
            self.get_logger().info("Preset motion completed successfully.")
        else:
            self.get_logger().warning("Preset motion did not complete.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InteractionResponderNode()
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

#!/usr/bin/env python3

"""Fuse one WAVE/COME event with LiDAR in a single-person scene."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import rclpy
from emo_robot_interfaces.msg import GestureDetection, LidarTarget, PersonTarget
from rclpy.node import Node


@dataclass(frozen=True)
class GestureTrigger:
    """Minimal gesture event retained for one approach session."""

    track_id: int
    confidence: float
    gesture: int
    received_s: float


@dataclass(frozen=True)
class LidarObservation:
    """ROS-independent LiDAR target observation."""

    valid: bool
    forward_m: float
    lateral_m: float
    angle_rad: float
    point_count: int
    received_s: float


@dataclass(frozen=True)
class FusedPersonTarget:
    """ROS-independent single-person fusion result."""

    valid: bool = False
    track_id: int = -1
    confidence: float = 0.0
    horizontal_error: float = 0.0
    lateral_m: float = 0.0
    distance_m: float = 0.0
    gesture_confirmed: bool = False


class SinglePersonTargetFusion:
    """Latch one approach gesture and treat the LiDAR cluster as that person."""

    APPROACH_GESTURES = frozenset(
        (
            GestureDetection.GESTURE_WAVE,
            GestureDetection.GESTURE_COME,
        )
    )

    def __init__(
        self,
        *,
        lidar_timeout_s: float = 0.4,
        trigger_match_window_s: float = 1.0,
        tracking_session_timeout_s: float = 8.5,
        reacquire_timeout_s: float = 3.0,
        min_target_points: int = 50,
        max_forward_jump_m: float = 0.75,
        max_lateral_jump_m: float = 0.5,
    ) -> None:
        if lidar_timeout_s <= 0.0:
            raise ValueError("lidar_timeout_s must be greater than zero")
        if trigger_match_window_s <= 0.0:
            raise ValueError(
                "trigger_match_window_s must be greater than zero"
            )
        if tracking_session_timeout_s <= trigger_match_window_s:
            raise ValueError(
                "tracking_session_timeout_s must exceed "
                "trigger_match_window_s"
            )
        if reacquire_timeout_s <= 0.0:
            raise ValueError("reacquire_timeout_s must be greater than zero")
        if min_target_points <= 0:
            raise ValueError("min_target_points must be greater than zero")
        if max_forward_jump_m <= 0.0 or max_lateral_jump_m <= 0.0:
            raise ValueError("target jump limits must be greater than zero")

        self.lidar_timeout_s = float(lidar_timeout_s)
        self.trigger_match_window_s = float(trigger_match_window_s)
        self.tracking_session_timeout_s = float(
            tracking_session_timeout_s
        )
        self.reacquire_timeout_s = float(reacquire_timeout_s)
        self.min_target_points = int(min_target_points)
        self.max_forward_jump_m = float(max_forward_jump_m)
        self.max_lateral_jump_m = float(max_lateral_jump_m)
        self._pending_trigger: Optional[GestureTrigger] = None
        self._active_trigger: Optional[GestureTrigger] = None
        self._last_lidar: Optional[LidarObservation] = None
        self._previous_target: Optional[LidarObservation] = None
        self._lost_since_s = 0.0
        self._session_deadline_s = 0.0
        self._last_error = ""

    @property
    def active(self) -> bool:
        """Return whether an approach gesture has an associated target."""
        return self._active_trigger is not None

    @property
    def last_error(self) -> str:
        """Return the reason the most recent session was released."""
        return self._last_error

    @classmethod
    def is_approach_gesture(cls, valid: bool, gesture: int) -> bool:
        """Return whether a gesture message may start an approach session."""
        return bool(valid and gesture in cls.APPROACH_GESTURES)

    def accept_trigger(self, trigger: GestureTrigger) -> FusedPersonTarget:
        """Start a pending session; repeated gesture events are ignored."""
        if self._active_trigger is not None or self._pending_trigger is not None:
            current = self.current(trigger.received_s)
            if (
                self._active_trigger is not None
                or self._pending_trigger is not None
            ):
                return current
        self._pending_trigger = trigger
        self._session_deadline_s = (
            trigger.received_s + self.tracking_session_timeout_s
        )
        self._last_error = ""
        if self._last_lidar is not None:
            return self._try_activate(self._last_lidar, trigger.received_s)
        return FusedPersonTarget()

    def accept_lidar(
        self,
        observation: LidarObservation,
    ) -> FusedPersonTarget:
        """Update the active target or match it to a pending gesture."""
        self._last_lidar = observation
        now_s = observation.received_s
        if self._session_expired(now_s):
            self.reset("tracking session expired")
            return FusedPersonTarget()
        if self._active_trigger is None:
            return self._try_activate(observation, now_s)
        if not self._observation_usable(observation, now_s):
            self._mark_lost("LiDAR target became invalid", now_s)
            return self._make_lost_target()
        if self._lost_since_s <= 0.0 and self._target_jumped(observation):
            self._mark_lost(
                "LiDAR target jumped beyond the configured limit",
                now_s,
            )
            return self._make_lost_target()
        self._lost_since_s = 0.0
        self._last_error = ""
        self._previous_target = observation
        return self._make_valid_target(observation)

    def current(self, now_s: float) -> FusedPersonTarget:
        """Return the current target, failing closed on every timeout."""
        if self._session_expired(now_s):
            self.reset("tracking session expired")
            return FusedPersonTarget()
        if self._active_trigger is None:
            if (
                self._pending_trigger is not None
                and now_s - self._pending_trigger.received_s
                > self.trigger_match_window_s
            ):
                self.reset("no valid LiDAR target near the gesture event")
            return FusedPersonTarget()
        observation = self._last_lidar
        if self._lost_since_s > 0.0:
            if self._lost_timeout_expired(now_s):
                self.reset("LiDAR target reacquire timed out")
                return FusedPersonTarget()
            return self._make_lost_target()
        if (
            observation is None
            or not self._observation_usable(observation, now_s)
        ):
            self._mark_lost("LiDAR target timed out", now_s)
            return self._make_lost_target()
        return self._make_valid_target(observation)

    def reset(self, reason: str = "") -> None:
        """Release the gesture latch so a new gesture is required."""
        self._pending_trigger = None
        self._active_trigger = None
        self._previous_target = None
        self._lost_since_s = 0.0
        self._session_deadline_s = 0.0
        self._last_error = reason

    def _try_activate(
        self,
        observation: LidarObservation,
        now_s: float,
    ) -> FusedPersonTarget:
        trigger = self._pending_trigger
        if trigger is None:
            return FusedPersonTarget()
        if abs(observation.received_s - trigger.received_s) > (
            self.trigger_match_window_s
        ):
            return FusedPersonTarget()
        if not self._observation_usable(observation, now_s):
            return FusedPersonTarget()
        self._pending_trigger = None
        self._active_trigger = trigger
        self._previous_target = observation
        self._lost_since_s = 0.0
        return self._make_valid_target(observation)

    def _mark_lost(self, reason: str, now_s: float) -> None:
        if self._lost_since_s <= 0.0:
            self._lost_since_s = float(now_s)
            self._last_error = reason
        if self._lost_timeout_expired(now_s):
            self.reset("LiDAR target reacquire timed out")

    def _lost_timeout_expired(self, now_s: float) -> bool:
        return bool(
            self._lost_since_s > 0.0
            and now_s - self._lost_since_s >= self.reacquire_timeout_s
        )

    def _observation_usable(
        self,
        observation: LidarObservation,
        now_s: float,
    ) -> bool:
        return bool(
            observation.valid
            and observation.point_count >= self.min_target_points
            and math.isfinite(observation.forward_m)
            and observation.forward_m > 0.0
            and math.isfinite(observation.lateral_m)
            and math.isfinite(observation.angle_rad)
            and 0.0 <= now_s - observation.received_s
            <= self.lidar_timeout_s
        )

    def _target_jumped(self, observation: LidarObservation) -> bool:
        previous = self._previous_target
        if previous is None:
            return False
        return bool(
            abs(observation.forward_m - previous.forward_m)
            > self.max_forward_jump_m
            or abs(observation.lateral_m - previous.lateral_m)
            > self.max_lateral_jump_m
        )

    def _session_expired(self, now_s: float) -> bool:
        return bool(
            self._session_deadline_s > 0.0
            and now_s >= self._session_deadline_s
        )

    def _make_valid_target(
        self,
        observation: LidarObservation,
    ) -> FusedPersonTarget:
        trigger = self._active_trigger
        if trigger is None:
            return FusedPersonTarget()
        return FusedPersonTarget(
            valid=True,
            track_id=trigger.track_id,
            confidence=trigger.confidence,
            horizontal_error=observation.angle_rad,
            lateral_m=observation.lateral_m,
            distance_m=observation.forward_m,
            gesture_confirmed=True,
        )

    def _make_lost_target(self) -> FusedPersonTarget:
        trigger = self._active_trigger
        if trigger is None:
            return FusedPersonTarget()
        observation = self._previous_target
        if observation is None:
            return FusedPersonTarget(
                valid=False,
                track_id=trigger.track_id,
                confidence=trigger.confidence,
                gesture_confirmed=True,
            )
        return FusedPersonTarget(
            valid=False,
            track_id=trigger.track_id,
            confidence=trigger.confidence,
            horizontal_error=observation.angle_rad,
            lateral_m=observation.lateral_m,
            distance_m=observation.forward_m,
            gesture_confirmed=True,
        )


class PersonTrackerNode(Node):
    """Publish one gesture-latched LiDAR target for a controlled test scene."""

    def __init__(self) -> None:
        super().__init__("person_tracker")
        self.declare_parameter(
            "gesture_topic",
            "/emo_robot/perception/gesture_detection",
        )
        self.declare_parameter(
            "lidar_target_topic",
            "/emo_robot/perception/lidar_target",
        )
        self.declare_parameter(
            "target_topic",
            "/emo_robot/perception/person_target",
        )
        self.declare_parameter("lidar_timeout_s", 0.4)
        self.declare_parameter("trigger_match_window_s", 1.0)
        self.declare_parameter("tracking_session_timeout_s", 8.5)
        self.declare_parameter("reacquire_timeout_s", 3.0)
        self.declare_parameter("min_target_points", 50)
        self.declare_parameter("max_forward_jump_m", 0.75)
        self.declare_parameter("max_lateral_jump_m", 0.5)

        self._fusion = SinglePersonTargetFusion(
            lidar_timeout_s=float(
                self.get_parameter("lidar_timeout_s").value
            ),
            trigger_match_window_s=float(
                self.get_parameter("trigger_match_window_s").value
            ),
            tracking_session_timeout_s=float(
                self.get_parameter("tracking_session_timeout_s").value
            ),
            reacquire_timeout_s=float(
                self.get_parameter("reacquire_timeout_s").value
            ),
            min_target_points=int(
                self.get_parameter("min_target_points").value
            ),
            max_forward_jump_m=float(
                self.get_parameter("max_forward_jump_m").value
            ),
            max_lateral_jump_m=float(
                self.get_parameter("max_lateral_jump_m").value
            ),
        )
        self._publisher = self.create_publisher(
            PersonTarget,
            str(self.get_parameter("target_topic").value),
            10,
        )
        self.create_subscription(
            GestureDetection,
            str(self.get_parameter("gesture_topic").value),
            self._on_gesture,
            10,
        )
        self.create_subscription(
            LidarTarget,
            str(self.get_parameter("lidar_target_topic").value),
            self._on_lidar,
            10,
        )
        self._watchdog = self.create_timer(0.05, self._on_watchdog)
        self._last_published_valid = False
        self._last_source_header = None
        self.get_logger().warning(
            "Single-person fusion ready: after WAVE/COME, the current LiDAR "
            "cluster is assumed to be that person. Use only in an empty, "
            "single-person test area."
        )

    def _on_gesture(self, message: GestureDetection) -> None:
        gesture = int(message.gesture)
        if not self._fusion.is_approach_gesture(message.valid, gesture):
            return
        result = self._fusion.accept_trigger(
            GestureTrigger(
                track_id=int(message.track_id),
                confidence=float(message.confidence),
                gesture=gesture,
                received_s=time.monotonic(),
            )
        )
        gesture_name = (
            "WAVE"
            if gesture == GestureDetection.GESTURE_WAVE
            else "COME"
        )
        self.get_logger().info(
            f"{gesture_name} received for single-person approach: "
            f"track_id={message.track_id}"
        )
        if result.valid:
            self._publish_result(result, self._last_source_header)

    def _on_lidar(self, message: LidarTarget) -> None:
        now_s = time.monotonic()
        self._last_source_header = message.header
        was_active = self._fusion.active
        was_valid = self._last_published_valid
        result = self._fusion.accept_lidar(
            LidarObservation(
                valid=bool(message.valid),
                forward_m=float(message.forward_m),
                lateral_m=float(message.lateral_m),
                angle_rad=float(message.angle_rad),
                point_count=int(message.point_count),
                received_s=now_s,
            )
        )
        session_released = bool(was_active and not self._fusion.active)
        if was_valid and not result.valid:
            state = "temporarily lost" if self._fusion.active else "released"
            self.get_logger().warning(
                f"Person target {state}: "
                f"{self._fusion.last_error or 'target invalid'}"
            )
        elif session_released:
            self.get_logger().warning(
                "Person target released: "
                f"{self._fusion.last_error or 'target invalid'}"
            )
        self._publish_result(result, message.header)

    def _on_watchdog(self) -> None:
        was_active = self._fusion.active
        result = self._fusion.current(time.monotonic())
        if result.valid:
            return
        if self._last_published_valid:
            state = "temporarily lost" if self._fusion.active else "released"
            self.get_logger().warning(
                f"Person target {state}: "
                f"{self._fusion.last_error or 'target invalid'}"
            )
            self._publish_result(result, None)
        elif was_active and not self._fusion.active:
            self.get_logger().warning(
                "Person target released: "
                f"{self._fusion.last_error or 'target invalid'}"
            )
            self._publish_result(result, None)

    def _publish_result(self, result, source_header) -> None:
        message = PersonTarget()
        if source_header is not None:
            message.header = source_header
        else:
            message.header.stamp = self.get_clock().now().to_msg()
        message.track_id = int(result.track_id)
        message.confidence = float(result.confidence)
        message.horizontal_error = float(result.horizontal_error)
        message.lateral_m = float(result.lateral_m)
        message.distance_m = float(result.distance_m)
        message.gesture_confirmed = bool(result.gesture_confirmed)
        message.valid = bool(result.valid)
        self._publisher.publish(message)
        self._last_published_valid = bool(result.valid)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonTrackerNode()
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

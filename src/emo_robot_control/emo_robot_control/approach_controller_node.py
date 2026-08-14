#!/usr/bin/env python3

"""Safety-gated single-person LiDAR approach controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
import signal
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from aimdk_msgs.msg import McAction, McActionStatus
from emo_robot_interfaces.msg import PersonTarget
from emo_robot_motion.locomotion_velocity_client import (
    LocomotionVelocityClient,
)
from emo_robot_motion.mc_action_state_client import (
    McActionState,
    McActionStateClient,
)


_ALLOWED_APPROACH_ACTIONS = (
    McAction.STAND_DEFAULT,
    McAction.LOCOMOTION_DEFAULT,
)
_ALLOWED_APPROACH_DESCRIPTIONS = {
    "STAND_DEFAULT",
    "LOCOMOTION_DEFAULT",
}


def is_approach_mode_running(state: Optional[McActionState]) -> bool:
    """Accept stable stand or locomotion mode, including AimDK value=0."""
    if (
        state is None
        or state.response_code != 0
        or state.status != McActionStatus.RUNNING
    ):
        return False

    description = state.description.strip().upper()
    if description in _ALLOWED_APPROACH_DESCRIPTIONS:
        return True

    return any(
        state.is_running(action)
        for action in _ALLOWED_APPROACH_ACTIONS
    )


@dataclass(frozen=True)
class ApproachDecision:
    """One fail-closed velocity decision."""

    forward: float = 0.0
    lateral: float = 0.0
    angular: float = 0.0
    arrived: bool = False
    transition: str = ""
    fault: str = ""


class ApproachPolicy:
    """Align laterally and angularly, then approach in a straight line."""

    ALIGN = "ALIGN"
    APPROACH = "APPROACH"

    def __init__(
        self,
        *,
        stop_distance_m: float = 1.3,
        forward_velocity: float = 0.2,
        align_lateral_velocity: float = 0.2,
        align_angular_velocity: float = 0.1,
        lateral_direction_sign: float = 1.0,
        align_angle_deg: float = 6.0,
        align_lateral_m: float = 0.20,
        align_stable_frames: int = 4,
        align_timeout_s: float = 5.0,
        realign_angle_deg: float = 14.0,
        realign_lateral_m: float = 0.45,
        max_target_angle_deg: float = 35.0,
        max_approach_duration_s: float = 5.0,
        max_total_duration_s: float = 8.0,
        arrival_frames: int = 3,
    ) -> None:
        if stop_distance_m <= 0.0:
            raise ValueError("stop_distance_m must be greater than zero")
        LocomotionVelocityClient._validate_velocity(
            "forward", forward_velocity, min_abs=0.2, max_abs=1.0
        )
        LocomotionVelocityClient._validate_velocity(
            "lateral", align_lateral_velocity, min_abs=0.2, max_abs=1.0
        )
        LocomotionVelocityClient._validate_velocity(
            "angular", align_angular_velocity, min_abs=0.1, max_abs=1.0
        )
        if forward_velocity <= 0.0:
            raise ValueError("forward_velocity must be positive")
        if align_lateral_velocity <= 0.0:
            raise ValueError("align_lateral_velocity must be positive")
        if align_angular_velocity <= 0.0:
            raise ValueError("align_angular_velocity must be positive")
        if lateral_direction_sign not in (-1.0, 1.0):
            raise ValueError("lateral_direction_sign must be 1.0 or -1.0")
        if not 0.0 < align_angle_deg < realign_angle_deg < (
            max_target_angle_deg
        ) < 90.0:
            raise ValueError(
                "angle limits must satisfy 0 < align < realign < max < 90"
            )
        if not 0.0 < align_lateral_m < realign_lateral_m:
            raise ValueError(
                "lateral limits must satisfy 0 < align < realign"
            )
        if align_stable_frames <= 0 or arrival_frames <= 0:
            raise ValueError("frame counters must be greater than zero")
        if align_timeout_s <= 0.0:
            raise ValueError("align_timeout_s must be greater than zero")
        if max_approach_duration_s <= 0.0:
            raise ValueError(
                "max_approach_duration_s must be greater than zero"
            )
        if max_total_duration_s <= 0.0:
            raise ValueError("max_total_duration_s must be greater than zero")

        self.stop_distance_m = float(stop_distance_m)
        self.forward_velocity = float(forward_velocity)
        self.align_lateral_velocity = float(align_lateral_velocity)
        self.align_angular_velocity = float(align_angular_velocity)
        self.lateral_direction_sign = float(lateral_direction_sign)
        self.align_angle_rad = math.radians(align_angle_deg)
        self.align_lateral_m = float(align_lateral_m)
        self.align_stable_frames = int(align_stable_frames)
        self.align_timeout_s = float(align_timeout_s)
        self.realign_angle_rad = math.radians(realign_angle_deg)
        self.realign_lateral_m = float(realign_lateral_m)
        self.max_target_angle_rad = math.radians(max_target_angle_deg)
        self.max_approach_duration_s = float(max_approach_duration_s)
        self.max_total_duration_s = float(max_total_duration_s)
        self.arrival_frames = int(arrival_frames)
        self.reset(0.0)

    def reset(self, now_s: float = 0.0) -> None:
        """Start a new ALIGN phase and clear all counters and timers."""
        self.phase = self.ALIGN
        self._task_started_s = float(now_s)
        self._align_started_s = float(now_s)
        self._last_decision_s = float(now_s)
        self._approach_elapsed_s = 0.0
        self._align_stable_count = 0
        self._arrival_count = 0

    def restart_align(self, now_s: float) -> None:
        """Start a fresh ALIGN phase while preserving task-level timers."""
        self.phase = self.ALIGN
        self._align_started_s = float(now_s)
        self._last_decision_s = float(now_s)
        self._align_stable_count = 0
        self._arrival_count = 0

    def decide(
        self,
        distance_m: float,
        lateral_m: float,
        angle_rad: float,
        now_s: float,
    ) -> ApproachDecision:
        """Return the next safe two-phase command."""
        values = (distance_m, lateral_m, angle_rad, now_s)
        if (
            not all(math.isfinite(value) for value in values)
            or distance_m <= 0
        ):
            return ApproachDecision(
                fault="target values are not finite/positive"
            )
        if now_s < self._last_decision_s:
            return ApproachDecision(fault="control time moved backwards")

        elapsed_s = now_s - self._last_decision_s
        if self.phase == self.APPROACH:
            self._approach_elapsed_s += elapsed_s
        self._last_decision_s = now_s

        if now_s - self._task_started_s >= self.max_total_duration_s:
            return ApproachDecision(fault="maximum total duration elapsed")
        if abs(angle_rad) > self.max_target_angle_rad:
            return ApproachDecision(fault="target angle exceeds safe limit")

        if distance_m <= self.stop_distance_m:
            self._arrival_count += 1
            return ApproachDecision(
                arrived=self._arrival_count >= self.arrival_frames
            )
        self._arrival_count = 0

        if self.phase == self.ALIGN:
            if now_s - self._align_started_s >= self.align_timeout_s:
                return ApproachDecision(fault="ALIGN timeout elapsed")
            aligned = bool(
                abs(angle_rad) <= self.align_angle_rad
                and abs(lateral_m) <= self.align_lateral_m
            )
            if aligned:
                self._align_stable_count += 1
                if self._align_stable_count >= self.align_stable_frames:
                    self.phase = self.APPROACH
                    return ApproachDecision(transition=self.APPROACH)
                return ApproachDecision()

            self._align_stable_count = 0
            lateral = 0.0
            angular = 0.0
            if abs(lateral_m) > self.align_lateral_m:
                lateral = math.copysign(
                    self.align_lateral_velocity,
                    lateral_m * self.lateral_direction_sign,
                )
            if abs(angle_rad) > self.align_angle_rad:
                angular = math.copysign(
                    self.align_angular_velocity,
                    angle_rad,
                )
            return ApproachDecision(lateral=lateral, angular=angular)

        if self._approach_elapsed_s >= self.max_approach_duration_s:
            return ApproachDecision(
                fault="maximum cumulative APPROACH duration elapsed"
            )
        if (
            abs(angle_rad) > self.realign_angle_rad
            or abs(lateral_m) > self.realign_lateral_m
        ):
            self.phase = self.ALIGN
            self._align_started_s = now_s
            self._align_stable_count = 0
            return ApproachDecision(transition=self.ALIGN)
        return ApproachDecision(forward=self.forward_velocity)


class ApproachControllerNode(Node):
    """Observe or execute one gesture-triggered, time-limited approach."""

    _CONFIRMATION = "I_UNDERSTAND"
    _SOURCE_NAME = "emo_robot_gesture_approach"
    _SNAPSHOT_TURN = "SNAPSHOT_TURN"
    _SETTLE = "SETTLE"
    _REACQUIRE = "REACQUIRE"
    _BLIND_APPROACH = "BLIND_APPROACH"
    _POLICY_STATES = {ApproachPolicy.ALIGN, ApproachPolicy.APPROACH}
    _PRE_APPROACH_STATES = {_SNAPSHOT_TURN, _SETTLE, _REACQUIRE}
    _CONTROL_STATES = {
        *_PRE_APPROACH_STATES,
        *_POLICY_STATES,
        _BLIND_APPROACH,
    }

    def __init__(self) -> None:
        super().__init__("approach_controller")
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter("confirmation", "")
        self.declare_parameter(
            "target_topic", "/emo_robot/perception/person_target"
        )
        self.declare_parameter(
            "debug_cmd_topic", "/emo_robot/control/approach_cmd_debug"
        )
        self.declare_parameter("stop_distance_m", 1.3)
        self.declare_parameter("forward_velocity", 0.2)
        self.declare_parameter("align_lateral_velocity", 0.2)
        self.declare_parameter("align_angular_velocity", 0.1)
        self.declare_parameter("lateral_direction_sign", 1.0)
        self.declare_parameter("align_angle_deg", 6.0)
        self.declare_parameter("align_lateral_m", 0.20)
        self.declare_parameter("align_stable_frames", 4)
        self.declare_parameter("align_timeout_s", 5.0)
        self.declare_parameter("snapshot_turn_enabled", True)
        self.declare_parameter("snapshot_turn_min_s", 0.2)
        self.declare_parameter("snapshot_turn_max_s", 2.5)
        self.declare_parameter("snapshot_turn_direction_sign", 1.0)
        self.declare_parameter("settle_duration_s", 0.6)
        self.declare_parameter("reacquire_timeout_s", 3.0)
        self.declare_parameter("blind_approach_enabled", True)
        self.declare_parameter("realign_angle_deg", 14.0)
        self.declare_parameter("realign_lateral_m", 0.45)
        self.declare_parameter("max_target_angle_deg", 35.0)
        self.declare_parameter("target_timeout_s", 0.4)
        self.declare_parameter("max_approach_duration_s", 5.0)
        self.declare_parameter("max_total_duration_s", 8.0)
        self.declare_parameter("arrival_frames", 3)

        self._motion_enabled = bool(
            self.get_parameter("motion_enabled").value
        )
        confirmation = str(self.get_parameter("confirmation").value)
        if self._motion_enabled and confirmation != self._CONFIRMATION:
            raise ValueError(
                "confirmation must be exactly I_UNDERSTAND when "
                "motion_enabled is true"
            )

        self._target_timeout_s = float(
            self.get_parameter("target_timeout_s").value
        )
        if self._target_timeout_s <= 0.0:
            raise ValueError("target_timeout_s must be greater than zero")
        self._snapshot_turn_enabled = bool(
            self.get_parameter("snapshot_turn_enabled").value
        )
        self._snapshot_turn_min_s = float(
            self.get_parameter("snapshot_turn_min_s").value
        )
        self._snapshot_turn_max_s = float(
            self.get_parameter("snapshot_turn_max_s").value
        )
        self._snapshot_turn_direction_sign = float(
            self.get_parameter("snapshot_turn_direction_sign").value
        )
        self._settle_duration_s = float(
            self.get_parameter("settle_duration_s").value
        )
        self._reacquire_timeout_s = float(
            self.get_parameter("reacquire_timeout_s").value
        )
        self._blind_approach_enabled = bool(
            self.get_parameter("blind_approach_enabled").value
        )
        if (
            self._snapshot_turn_min_s < 0.0
            or self._snapshot_turn_max_s <= 0.0
            or self._snapshot_turn_min_s > self._snapshot_turn_max_s
        ):
            raise ValueError(
                "snapshot_turn_min_s/max_s must satisfy "
                "0 <= min <= max"
            )
        if self._snapshot_turn_direction_sign not in (-1.0, 1.0):
            raise ValueError(
                "snapshot_turn_direction_sign must be 1.0 or -1.0"
            )
        if self._settle_duration_s < 0.0:
            raise ValueError("settle_duration_s must be non-negative")
        if self._reacquire_timeout_s <= 0.0:
            raise ValueError("reacquire_timeout_s must be greater than zero")
        self._policy = ApproachPolicy(
            stop_distance_m=float(
                self.get_parameter("stop_distance_m").value
            ),
            forward_velocity=float(
                self.get_parameter("forward_velocity").value
            ),
            align_lateral_velocity=float(
                self.get_parameter("align_lateral_velocity").value
            ),
            align_angular_velocity=float(
                self.get_parameter("align_angular_velocity").value
            ),
            lateral_direction_sign=float(
                self.get_parameter("lateral_direction_sign").value
            ),
            align_angle_deg=float(
                self.get_parameter("align_angle_deg").value
            ),
            align_lateral_m=float(
                self.get_parameter("align_lateral_m").value
            ),
            align_stable_frames=int(
                self.get_parameter("align_stable_frames").value
            ),
            align_timeout_s=float(
                self.get_parameter("align_timeout_s").value
            ),
            realign_angle_deg=float(
                self.get_parameter("realign_angle_deg").value
            ),
            realign_lateral_m=float(
                self.get_parameter("realign_lateral_m").value
            ),
            max_target_angle_deg=float(
                self.get_parameter("max_target_angle_deg").value
            ),
            max_approach_duration_s=float(
                self.get_parameter("max_approach_duration_s").value
            ),
            max_total_duration_s=float(
                self.get_parameter("max_total_duration_s").value
            ),
            arrival_frames=int(self.get_parameter("arrival_frames").value),
        )

        self._debug_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("debug_cmd_topic").value),
            10,
        )
        self.create_subscription(
            PersonTarget,
            str(self.get_parameter("target_topic").value),
            self._on_target,
            10,
        )
        self._control_timer = self.create_timer(0.05, self._on_control_tick)

        self._state = "IDLE"
        self._target: Optional[PersonTarget] = None
        self._target_received_s = 0.0
        self._snapshot_distance_m = 0.0
        self._snapshot_lateral_m = 0.0
        self._snapshot_angle_rad = 0.0
        self._approach_snapshot_distance_m = 0.0
        self._approach_snapshot_lateral_m = 0.0
        self._approach_snapshot_angle_rad = 0.0
        self._blind_approach_started_s = 0.0
        self._blind_approach_deadline_s = 0.0
        self._blind_arrival_count = 0
        self._state_deadline_s = 0.0
        self._awaiting_release = False
        self._last_mode_check_s = 0.0
        self._mode_query_kind = ""
        self._shutting_down = False
        self._shutdown_complete = False

        self._state_client: Optional[McActionStateClient] = None
        self._motion: Optional[LocomotionVelocityClient] = None
        if self._motion_enabled:
            self._state_client = McActionStateClient(
                self,
                request_timeout_s=0.25,
                max_attempts=2,
            )
            self._motion = LocomotionVelocityClient(
                self,
                source_name=self._SOURCE_NAME,
                publish_period_s=0.02,
            )
            self.get_logger().warning(
                "REAL ROBOT APPROACH IS ARMED. Motion still requires a "
                "fresh WAVE/COME target and STAND_DEFAULT/RUNNING or "
                "LOCOMOTION_DEFAULT/RUNNING."
            )
        else:
            self.get_logger().warning(
                "Approach controller is observation-only. Debug velocity "
                "will be published, but no AimDK input source is created."
            )

    @property
    def shutdown_complete(self) -> bool:
        """Return whether shutdown cleanup has completed."""
        return self._shutdown_complete

    @staticmethod
    def _message_session_active(message: PersonTarget) -> bool:
        return bool(message.gesture_confirmed and int(message.track_id) >= 0)

    def _on_target(self, message: PersonTarget) -> None:
        session_active = self._message_session_active(message)
        if not message.valid or not message.gesture_confirmed:
            self._target = None
            if self._state in {
                *self._POLICY_STATES,
            }:
                self._stop_task("person target became invalid", fault=True)
            elif self._state == "WAIT_RELEASE" and not session_active:
                self._release_for_next_wave()
            return

        self._target = message
        self._target_received_s = time.monotonic()
        if self._state == "IDLE" and not self._awaiting_release:
            self._begin_task()

    def _begin_task(self) -> None:
        self._awaiting_release = True
        now_s = time.monotonic()
        self._policy.reset(now_s)
        self._snapshot_distance_m = float(self._target.distance_m)
        self._snapshot_lateral_m = float(self._target.lateral_m)
        self._snapshot_angle_rad = float(self._target.horizontal_error)
        self._state_deadline_s = 0.0
        if not self._motion_enabled:
            self._enter_first_control_state(now_s)
            self.get_logger().info(
                f"Observation-only {self._state} started; inspect "
                "/emo_robot/control/approach_cmd_debug."
            )
            return
        self._state = "VERIFY_MODE"
        self._query_mode("start")

    def _enter_first_control_state(self, now_s: float) -> None:
        if (
            self._snapshot_turn_enabled
            and abs(self._snapshot_angle_rad) > self._policy.align_angle_rad
        ):
            duration_s = self._snapshot_turn_duration_s(
                self._snapshot_angle_rad
            )
            self._state = self._SNAPSHOT_TURN
            self._state_deadline_s = now_s + duration_s
            self.get_logger().info(
                "Snapshot turn started: "
                f"angle={math.degrees(self._snapshot_angle_rad):.1f} deg, "
                f"duration={duration_s:.2f}s, then settle/reacquire."
            )
            return
        self._state = self._policy.phase

    def _snapshot_turn_duration_s(self, angle_rad: float) -> float:
        duration_s = abs(float(angle_rad)) / self._policy.align_angular_velocity
        return min(
            max(duration_s, self._snapshot_turn_min_s),
            self._snapshot_turn_max_s,
        )

    def _query_mode(self, kind: str) -> None:
        if self._state_client is None or self._state_client.active:
            return
        self._mode_query_kind = kind
        self._state_client.get_state_async(self._on_mode_state)

    def _on_mode_state(
        self,
        success: bool,
        state: Optional[McActionState],
        error: str,
    ) -> None:
        kind = self._mode_query_kind
        self._mode_query_kind = ""
        if self._shutting_down:
            return
        valid_mode = bool(
            success and is_approach_mode_running(state)
        )
        if not valid_mode:
            detail = error or (
                f"action={getattr(state, 'action', None)} "
                f"status={getattr(state, 'status', None)}"
            )
            self._stop_task(
                "robot is not STAND_DEFAULT/RUNNING or "
                "LOCOMOTION_DEFAULT/RUNNING: "
                f"{detail}",
                fault=True,
            )
            return
        self._last_mode_check_s = time.monotonic()
        if kind == "start" and self._state == "VERIFY_MODE":
            self._state = "REGISTERING"
            self._motion.register_input_source_async(
                done_callback=self._on_input_registered
            )

    def _on_input_registered(self, success: bool) -> None:
        if self._state == "STOPPING" or self._shutting_down:
            if success:
                self._motion.emergency_stop()
                self._motion.unregister_input_source_async(
                    done_callback=self._finish_cleanup
                )
            else:
                self._finish_cleanup(False)
            return
        if self._state != "REGISTERING":
            return
        if not success:
            self._stop_task("input-source registration failed", fault=True)
            return
        self._motion.set_velocity(0.0, 0.0, 0.0)
        self._motion.start_publishing()
        self._enter_first_control_state(time.monotonic())
        self.get_logger().warning(
            f"Approach motion started in {self._state}: "
            f"snapshot={self._target_summary()}, "
            f"forward=0 until approach snapshot, "
            f"angular<={self._policy.align_angular_velocity:.2f} rad/s."
        )

    def _on_control_tick(self) -> None:
        if self._state not in self._CONTROL_STATES:
            self._publish_debug(0.0, 0.0, 0.0)
            return
        now_s = time.monotonic()
        if self._state in self._PRE_APPROACH_STATES:
            self._run_pre_approach_tick(now_s)
            return
        if self._state == self._BLIND_APPROACH:
            self._run_blind_approach_tick(now_s)
            return
        if self._target is None or (
            now_s - self._target_received_s > self._target_timeout_s
        ):
            self._stop_task("person target timed out", fault=True)
            return

        decision = self._policy.decide(
            float(self._target.distance_m),
            float(self._target.lateral_m),
            float(self._target.horizontal_error),
            now_s,
        )
        self._state = self._policy.phase
        self._publish_debug(
            decision.forward,
            decision.lateral,
            decision.angular,
        )
        if decision.fault:
            self._stop_task(decision.fault, fault=True)
            return
        if decision.arrived:
            self._stop_task("stop distance confirmed")
            return
        if decision.transition:
            if (
                decision.transition == ApproachPolicy.APPROACH
                and self._blind_approach_enabled
            ):
                self._enter_blind_approach(now_s)
                return
            self.get_logger().info(
                f"Approach phase changed to {decision.transition}; "
                "zero speed applied for this control frame."
            )
        if self._motion_enabled:
            try:
                self._motion.set_velocity(
                    decision.forward,
                    decision.lateral,
                    decision.angular,
                )
            except ValueError as exc:
                self._stop_task(f"invalid velocity command: {exc}", fault=True)
                return
            if now_s - self._last_mode_check_s >= 1.0:
                self._query_mode("periodic")

    def _enter_blind_approach(self, now_s: float) -> None:
        if self._target is None:
            self._stop_task("cannot snapshot approach without target", fault=True)
            return
        distance_m = float(self._target.distance_m)
        lateral_m = float(self._target.lateral_m)
        angle_rad = float(self._target.horizontal_error)
        if (
            not all(
                math.isfinite(value)
                for value in (distance_m, lateral_m, angle_rad)
            )
            or distance_m <= 0.0
        ):
            self._stop_task("approach snapshot target is invalid", fault=True)
            return
        self._approach_snapshot_distance_m = distance_m
        self._approach_snapshot_lateral_m = lateral_m
        self._approach_snapshot_angle_rad = angle_rad
        self._blind_approach_started_s = now_s
        self._blind_approach_deadline_s = (
            now_s + self._blind_approach_duration_s(distance_m)
        )
        self._blind_arrival_count = 0
        self._state = self._BLIND_APPROACH
        self.get_logger().info(
            "Approach snapshot captured: "
            f"{self._approach_snapshot_summary()}; entering BLIND_APPROACH."
        )

    def _blind_approach_duration_s(self, distance_m: float) -> float:
        travel_m = max(0.0, float(distance_m) - self._policy.stop_distance_m)
        planned_s = travel_m / self._policy.forward_velocity
        return min(planned_s, self._policy.max_approach_duration_s)

    def _run_blind_approach_tick(self, now_s: float) -> None:
        if now_s - self._policy._task_started_s >= (
            self._policy.max_total_duration_s
        ):
            self._stop_task("maximum total duration elapsed", fault=True)
            return
        if now_s - self._blind_approach_started_s >= (
            self._policy.max_approach_duration_s
        ):
            self._stop_task(
                "maximum cumulative BLIND_APPROACH duration elapsed",
                fault=True,
            )
            return

        fresh_target = bool(
            self._target is not None
            and now_s - self._target_received_s <= self._target_timeout_s
        )
        if fresh_target:
            distance_m = float(self._target.distance_m)
            if math.isfinite(distance_m) and distance_m <= (
                self._policy.stop_distance_m
            ):
                self._blind_arrival_count += 1
                if self._blind_arrival_count >= self._policy.arrival_frames:
                    self._stop_task("stop distance confirmed")
                    return
            else:
                self._blind_arrival_count = 0

        if now_s >= self._blind_approach_deadline_s:
            self._stop_task("approach snapshot travel time elapsed")
            return

        self._send_velocity(self._policy.forward_velocity, 0.0, 0.0)
        if (
            self._motion_enabled
            and now_s - self._last_mode_check_s >= 1.0
        ):
            self._query_mode("periodic")

    def _run_pre_approach_tick(self, now_s: float) -> None:
        if now_s - self._policy._task_started_s >= (
            self._policy.max_total_duration_s
        ):
            self._stop_task("maximum total duration elapsed", fault=True)
            return

        if self._state == self._SNAPSHOT_TURN:
            if now_s >= self._state_deadline_s:
                self._state = self._SETTLE
                self._state_deadline_s = now_s + self._settle_duration_s
                self._send_velocity(0.0, 0.0, 0.0)
                self.get_logger().info(
                    "Snapshot turn complete; settling before LiDAR "
                    "reacquire."
                )
                return
            angular = math.copysign(
                self._policy.align_angular_velocity,
                self._snapshot_angle_rad * self._snapshot_turn_direction_sign,
            )
            self._send_velocity(0.0, 0.0, angular)
            return

        if self._state == self._SETTLE:
            self._send_velocity(0.0, 0.0, 0.0)
            if now_s >= self._state_deadline_s:
                self._state = self._REACQUIRE
                self._state_deadline_s = now_s + self._reacquire_timeout_s
                self.get_logger().info(
                    "Settle complete; waiting for fresh LiDAR "
                    "PersonTarget before approach."
                )
            return

        if self._state == self._REACQUIRE:
            self._send_velocity(0.0, 0.0, 0.0)
            fresh_target = bool(
                self._target is not None
                and now_s - self._target_received_s <= self._target_timeout_s
            )
            if fresh_target:
                self._policy.restart_align(now_s)
                self._state = self._policy.phase
                self.get_logger().info(
                    f"Person target reacquired: {self._target_summary()}; "
                    "entering ALIGN."
                )
                return
            if now_s >= self._state_deadline_s:
                self._stop_task("person target reacquire timed out", fault=True)
            return

    def _send_velocity(
        self,
        forward: float,
        lateral: float,
        angular: float,
    ) -> None:
        self._publish_debug(forward, lateral, angular)
        if not self._motion_enabled:
            return
        try:
            self._motion.set_velocity(forward, lateral, angular)
        except ValueError as exc:
            self._stop_task(f"invalid velocity command: {exc}", fault=True)

    def _target_summary(self) -> str:
        if self._target is None:
            distance_m = self._snapshot_distance_m
            lateral_m = self._snapshot_lateral_m
            angle_rad = self._snapshot_angle_rad
        else:
            distance_m = float(self._target.distance_m)
            lateral_m = float(self._target.lateral_m)
            angle_rad = float(self._target.horizontal_error)
        return (
            f"distance={distance_m:.2f}m, lateral={lateral_m:.2f}m, "
            f"angle={math.degrees(angle_rad):.1f}deg"
        )

    def _approach_snapshot_summary(self) -> str:
        return (
            f"distance={self._approach_snapshot_distance_m:.2f}m, "
            f"lateral={self._approach_snapshot_lateral_m:.2f}m, "
            f"angle={math.degrees(self._approach_snapshot_angle_rad):.1f}deg, "
            f"duration="
            f"{self._blind_approach_deadline_s - self._blind_approach_started_s:.2f}s"
        )

    def _publish_debug(
        self,
        forward: float,
        lateral: float,
        angular: float,
    ) -> None:
        message = Twist()
        message.linear.x = float(forward)
        message.linear.y = float(lateral)
        message.angular.z = float(angular)
        self._debug_pub.publish(message)

    def _stop_task(self, reason: str, fault: bool = False) -> None:
        if self._state in {"STOPPING", "WAIT_RELEASE"}:
            return
        self.get_logger().warning(f"Stopping approach: {reason}")
        self._publish_debug(0.0, 0.0, 0.0)
        self._policy.reset(time.monotonic())
        if self._state_client is not None:
            self._state_client.cancel()
        if not self._motion_enabled or self._motion is None:
            self._state = "WAIT_RELEASE"
            if self._shutting_down:
                self._release_for_next_wave()
            return

        self._state = "STOPPING"
        self._motion.emergency_stop()
        if self._motion.registered:
            self._motion.unregister_input_source_async(
                done_callback=self._finish_cleanup
            )
        elif not self._motion.registering:
            self._finish_cleanup(True)

    def _finish_cleanup(self, success: bool) -> None:
        if not success:
            self.get_logger().error(
                "Approach input-source cleanup failed; the AimDK 1000 ms "
                "source timeout remains the fallback."
            )
        self._state = "WAIT_RELEASE"
        if self._shutting_down:
            self._release_for_next_wave()

    def _release_for_next_wave(self) -> None:
        self._awaiting_release = False
        self._target = None
        self._state = "IDLE"
        if self._shutting_down:
            self._shutdown_complete = True

    def request_shutdown(self, reason: str = "node shutdown") -> None:
        """Fail closed and begin idempotent asynchronous cleanup."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._target = None
        if self._state == "STOPPING":
            return
        if self._state in {
            "VERIFY_MODE",
            "REGISTERING",
            *self._CONTROL_STATES,
        }:
            self._stop_task(reason)
        elif self._motion is not None and self._motion.registered:
            self._state = "STOPPING"
            self._motion.emergency_stop()
            self._motion.unregister_input_source_async(
                done_callback=self._finish_cleanup
            )
        else:
            if self._motion is not None:
                self._motion.emergency_stop()
            self._shutdown_complete = True

    def destroy_node(self):
        if self._state_client is not None:
            self._state_client.cancel()
        if self._motion is not None:
            self._motion.emergency_stop()
            if self._motion.registered:
                self._motion.unregister_input_source_async()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    received_signal = {"number": None}
    previous_handlers = {}

    def _signal_handler(signum, _frame) -> None:
        received_signal["number"] = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _signal_handler)
        node = ApproachControllerNode()
        while rclpy.ok() and not node.shutdown_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
            if received_signal["number"] is not None:
                signum = received_signal["number"]
                received_signal["number"] = None
                node.request_shutdown(f"received signal {signum}")
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Approach controller exception: {exc}")
            node.request_shutdown("unhandled exception")
        else:
            get_logger("approach_controller").error(
                f"Approach controller failed during startup: {exc}"
            )
    finally:
        if node is not None and not node.shutdown_complete:
            node.request_shutdown()
            deadline = time.monotonic() + 3.0
            while (
                rclpy.ok()
                and not node.shutdown_complete
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(node, timeout_sec=0.05)
        if node is not None:
            node.destroy_node()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

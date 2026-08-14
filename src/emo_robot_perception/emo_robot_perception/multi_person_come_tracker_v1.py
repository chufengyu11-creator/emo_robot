from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from emo_robot_perception.single_person_come_recognizer_v1 import (
    ArmFeatures,
    ContinuousOscillationRecognizer,
    HandFeatures,
    HybridActiveArmSelector,
    MediaPipeHandDetector,
    distance,
    draw_hand,
    extract_arm_features,
    open_camera,
    signed_angle_degrees,
)


# COCO Pose keypoint indices used by the explicit greeting-wave detector.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


@dataclass
class PersonObservation:
    track_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    bbox_height_ratio: float
    bbox_area_ratio: float
    keypoints: np.ndarray


@dataclass
class GestureEvent:
    track_id: int
    action: str  # COME / WAVE
    source: str
    arm: str
    score: float
    timestamp: float
    palm_front_score: Optional[float] = None
    radial_fraction: Optional[float] = None
    ownership_score: float = 0.0
    wrist_travel_px: float = 0.0
    wrist_travel_ratio: float = 0.0
    bbox_height_ratio: float = 0.0
    hand_surface: str = "UNKNOWN"
    hand_surface_score: Optional[float] = None
    motion_source: str = ""
    pose_fallback: bool = False
    quality_band: str = "NORMAL"


@dataclass
class ExtendedHandFeatures(HandFeatures):
    # Absolute front-facing confidence:
    #   1.0 -> hand plane faces the camera
    #   0.0 -> hand is edge-on / orientation is unreliable
    palm_front_score: float = 0.0

    # Signed hand-surface score after anatomical LEFT/RIGHT correction:
    #   positive -> PALM faces camera
    #   negative -> BACK faces camera
    hand_surface_score: float = 0.0


@dataclass
class ArmMotionSample:
    timestamp: float
    reach_ratio: float
    wrist_xy: np.ndarray
    body_scale: float


@dataclass
class WaveArmMetrics:
    raised: bool = False
    hold_duration: float = 0.0
    lateral_range: float = 0.0
    reach_range: float = 0.0
    direction_changes: int = 0
    candidate: bool = False


class PoseWaveRecognizer:
    """
    Multi-person adaptation of the colleague's greeting-wave logic.

    For each anatomical arm, store:
        timestamp,
        wrist horizontal position relative to elbow,
        shoulder-to-wrist reach ratio,
        whether the hand is raised.

    A greeting WAVE requires a raised hand and genuine left-right motion.
    Static raised-hand triggering is optional and disabled by default,
    because it conflicts with COME preparation.
    """

    def __init__(
        self,
        window_seconds: float,
        motion_hold_seconds: float,
        static_hold_seconds: float,
        cooldown_seconds: float,
        min_lateral_range: float,
        min_direction_changes: int,
        direction_delta: float,
        lateral_dominance: float,
        down_reset_seconds: float,
        history_gap_reset_seconds: float,
        min_kp_conf: float,
    ) -> None:
        self.window_seconds = window_seconds
        self.motion_hold_seconds = motion_hold_seconds
        self.static_hold_seconds = static_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_lateral_range = min_lateral_range
        self.min_direction_changes = min_direction_changes
        self.direction_delta = direction_delta
        self.lateral_dominance = lateral_dominance
        self.down_reset_seconds = down_reset_seconds
        self.history_gap_reset_seconds = history_gap_reset_seconds
        self.min_kp_conf = min_kp_conf

        self.history: Dict[
            str,
            Deque[Tuple[float, float, float, bool]],
        ] = {
            "LEFT": deque(),
            "RIGHT": deque(),
        }
        self.latched = {
            "LEFT": False,
            "RIGHT": False,
        }
        self.hand_down_since: Dict[str, Optional[float]] = {
            "LEFT": None,
            "RIGHT": None,
        }
        self.last_wave_time = {
            "LEFT": -1e9,
            "RIGHT": -1e9,
        }
        self.metrics = {
            "LEFT": WaveArmMetrics(),
            "RIGHT": WaveArmMetrics(),
        }

    def reset(self) -> None:
        for history in self.history.values():
            history.clear()

        self.latched = {
            "LEFT": False,
            "RIGHT": False,
        }
        self.hand_down_since = {
            "LEFT": None,
            "RIGHT": None,
        }
        self.last_wave_time = {
            "LEFT": -1e9,
            "RIGHT": -1e9,
        }
        self.metrics = {
            "LEFT": WaveArmMetrics(),
            "RIGHT": WaveArmMetrics(),
        }

    def reset_arm(self, arm_name: str) -> None:
        self.history[arm_name].clear()
        self.latched[arm_name] = False
        self.hand_down_since[arm_name] = None
        self.metrics[arm_name] = WaveArmMetrics()

    @staticmethod
    def _keypoint_valid(
        keypoints: np.ndarray,
        index: int,
        min_conf: float,
    ) -> bool:
        return (
            0 <= index < len(keypoints)
            and keypoints[index].shape[0] >= 3
            and float(keypoints[index][2]) >= min_conf
            and float(keypoints[index][0]) > 0
            and float(keypoints[index][1]) > 0
        )

    def _count_direction_changes(
        self,
        values: list[float],
    ) -> int:
        if len(values) < 3:
            return 0

        signs: list[int] = []

        for previous, current in zip(values[:-1], values[1:]):
            delta = current - previous

            if abs(delta) < self.direction_delta:
                continue

            sign = 1 if delta > 0 else -1

            if not signs or signs[-1] != sign:
                signs.append(sign)

        return max(0, len(signs) - 1)

    @staticmethod
    def _contiguous_raised_duration(
        history: Deque[Tuple[float, float, float, bool]],
        now: float,
    ) -> float:
        start_time: Optional[float] = None

        for timestamp, _, _, raised in reversed(history):
            if not raised:
                break
            start_time = timestamp

        if start_time is None:
            return 0.0

        return max(0.0, now - start_time)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds

        for history in self.history.values():
            while history and history[0][0] < cutoff:
                history.popleft()

    def _append_arm_sample(
        self,
        arm_name: str,
        keypoints: np.ndarray,
        arm: ArmFeatures,
        now: float,
    ) -> None:
        if arm_name == "LEFT":
            shoulder_index = LEFT_SHOULDER
            elbow_index = LEFT_ELBOW
            wrist_index = LEFT_WRIST
        else:
            shoulder_index = RIGHT_SHOULDER
            elbow_index = RIGHT_ELBOW
            wrist_index = RIGHT_WRIST

        required = (
            shoulder_index,
            elbow_index,
            wrist_index,
        )

        if (
            not arm.valid
            or not arm.side_ok
            or not all(
                self._keypoint_valid(
                    keypoints,
                    index,
                    self.min_kp_conf,
                )
                for index in required
            )
        ):
            self.history[arm_name].append(
                (now, 0.0, 0.0, False)
            )
            return

        shoulder = keypoints[shoulder_index]
        elbow = keypoints[elbow_index]
        wrist = keypoints[wrist_index]
        scale = max(arm.body_scale, 1.0)

        raised = (
            float(wrist[1])
            <= float(shoulder[1]) + 0.90 * scale
            and float(wrist[1])
            <= float(elbow[1]) + 0.55 * scale
            and float(
                np.linalg.norm(wrist[:2] - shoulder[:2])
            )
            >= 0.35 * scale
        )

        wrist_relative_x = (
            float(wrist[0]) - float(elbow[0])
        ) / scale

        self.history[arm_name].append(
            (
                now,
                wrist_relative_x,
                arm.reach_ratio,
                raised,
            )
        )

    def update(
        self,
        keypoints: np.ndarray,
        arms: Dict[str, ArmFeatures],
        now: float,
    ) -> Optional[str]:
        for arm_name in ("LEFT", "RIGHT"):
            self._append_arm_sample(
                arm_name,
                keypoints,
                arms[arm_name],
                now,
            )

        self._trim(now)

        triggered_arms: list[str] = []

        for arm_name in ("LEFT", "RIGHT"):
            history = self.history[arm_name]
            raised_now = bool(history and history[-1][3])

            if not raised_now:
                if self.hand_down_since[arm_name] is None:
                    self.hand_down_since[arm_name] = now

                if (
                    now - self.hand_down_since[arm_name]
                    >= self.down_reset_seconds
                ):
                    self.latched[arm_name] = False

                if (
                    now - self.hand_down_since[arm_name]
                    >= self.history_gap_reset_seconds
                ):
                    self.history[arm_name].clear()

                self.metrics[arm_name] = WaveArmMetrics()
                continue

            self.hand_down_since[arm_name] = None

            raised_lateral = [
                lateral
                for _, lateral, _, raised in history
                if raised
            ]
            raised_reach = [
                reach
                for _, _, reach, raised in history
                if raised
            ]

            hold_duration = self._contiguous_raised_duration(
                history,
                now,
            )

            lateral_range = (
                max(raised_lateral) - min(raised_lateral)
                if len(raised_lateral) >= 2
                else 0.0
            )
            reach_range = (
                max(raised_reach) - min(raised_reach)
                if len(raised_reach) >= 2
                else 0.0
            )
            direction_changes = self._count_direction_changes(
                raised_lateral
            )

            lateral_dominates = (
                lateral_range
                >= self.lateral_dominance
                * max(reach_range, 0.015)
            )

            motion_trigger = (
                hold_duration >= self.motion_hold_seconds
                and lateral_range >= self.min_lateral_range
                and direction_changes
                >= self.min_direction_changes
                and lateral_dominates
            )

            # Disabled by default. This exists only for compatibility
            # experiments with the colleague's very-easy raised-hand mode.
            static_trigger = (
                self.static_hold_seconds > 0
                and hold_duration >= self.static_hold_seconds
                and lateral_range
                >= 0.35 * self.min_lateral_range
                and lateral_dominates
            )

            candidate = (
                hold_duration
                >= 0.65 * self.motion_hold_seconds
                and lateral_range
                >= 0.55 * self.min_lateral_range
                and direction_changes >= 1
                and lateral_dominates
            )

            self.metrics[arm_name] = WaveArmMetrics(
                raised=True,
                hold_duration=hold_duration,
                lateral_range=lateral_range,
                reach_range=reach_range,
                direction_changes=direction_changes,
                candidate=candidate,
            )

            if (
                not self.latched[arm_name]
                and (
                    motion_trigger
                    or static_trigger
                )
                and (
                    now - self.last_wave_time[arm_name]
                    >= self.cooldown_seconds
                )
            ):
                self.latched[arm_name] = True
                self.last_wave_time[arm_name] = now
                triggered_arms.append(arm_name)

        if not triggered_arms:
            return None

        # Choose the arm with stronger lateral evidence.
        return max(
            triggered_arms,
            key=lambda arm_name: (
                self.metrics[arm_name].lateral_range
                / max(
                    self.metrics[arm_name].reach_range,
                    0.015,
                )
            ),
        )

    def is_candidate(self, arm_name: str) -> bool:
        return self.metrics[arm_name].candidate

    def event_score(self, arm_name: str) -> float:
        metrics = self.metrics[arm_name]
        return (
            metrics.lateral_range
            / max(self.min_lateral_range, 1e-6)
        )


@dataclass
class PersonGestureState:
    track_id: int
    arm_recognizers: Dict[str, ContinuousOscillationRecognizer]
    wrist_recognizers: Dict[str, ContinuousOscillationRecognizer]
    selector: HybridActiveArmSelector
    wave_recognizer: PoseWaveRecognizer
    created_at: float
    last_seen: float
    seen_frames: int = 0

    action_cooldown_until: float = 0.0
    last_action_time: float = -1e9
    last_action_name: str = "NONE"

    last_event_until: float = 0.0
    last_event_action: str = "NONE"
    last_event_source: str = ""
    last_event_arm: str = ""
    active_arm: Optional[str] = None

    palm_front_ema: Dict[str, Optional[float]] = field(
        default_factory=lambda: {
            "LEFT": None,
            "RIGHT": None,
        }
    )
    hand_surface_history: Dict[
        str,
        Deque[Tuple[float, float]],
    ] = field(
        default_factory=lambda: {
            "LEFT": deque(),
            "RIGHT": deque(),
        }
    )
    arm_motion_history: Dict[str, Deque[ArmMotionSample]] = field(
        default_factory=lambda: {
            "LEFT": deque(),
            "RIGHT": deque(),
        }
    )

    def reset_action_detectors(self) -> None:
        """
        Reset all mutually competing action detectors while preserving:
          - recent physical motion used for ownership attribution,
          - the current UI event message,
          - track age and identity.
        """
        for recognizer in self.arm_recognizers.values():
            recognizer.reset()
        for recognizer in self.wrist_recognizers.values():
            recognizer.reset()

        self.wave_recognizer.reset()
        self.selector.reset()
        self.active_arm = None

    def begin_action_cooldown(
        self,
        action: str,
        now: float,
        cooldown_seconds: float,
    ) -> None:
        self.last_action_time = now
        self.last_action_name = action
        self.action_cooldown_until = (
            now + max(0.0, cooldown_seconds)
        )

        # This is essential: without clearing all three paths, the residual
        # trajectory of a WAVE can immediately complete a COME trajectory.
        self.reset_action_detectors()

    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self.action_cooldown_until - now)

    def clear_gesture_histories(self) -> None:
        self.reset_action_detectors()

        self.action_cooldown_until = 0.0
        self.last_action_time = -1e9
        self.last_action_name = "NONE"

        self.palm_front_ema = {
            "LEFT": None,
            "RIGHT": None,
        }
        self.hand_surface_history = {
            "LEFT": deque(),
            "RIGHT": deque(),
        }
        self.arm_motion_history = {
            "LEFT": deque(),
            "RIGHT": deque(),
        }

        self.last_event_until = 0.0
        self.last_event_action = "NONE"
        self.last_event_source = ""
        self.last_event_arm = ""


class TargetManager:
    def __init__(self, release_timeout: float) -> None:
        self.release_timeout = release_timeout
        self.target_id: Optional[int] = None
        self.task_id: Optional[str] = None
        self.locked_at: Optional[float] = None
        self.last_seen_at: Optional[float] = None

    def lock(self, track_id: int, now: float) -> bool:
        if self.target_id is None:
            self.target_id = track_id
            self.task_id = f"approach_{int(time.time() * 1000)}"
            self.locked_at = now
            self.last_seen_at = now
            return True

        return self.target_id == track_id

    def release(self) -> Optional[int]:
        old_target = self.target_id
        self.target_id = None
        self.task_id = None
        self.locked_at = None
        self.last_seen_at = None
        return old_target

    def update_visibility(
        self,
        seen_ids: set[int],
        now: float,
    ) -> Optional[int]:
        if self.target_id is None:
            return None

        if self.target_id in seen_ids:
            self.last_seen_at = now
            return None

        if (
            self.last_seen_at is not None
            and now - self.last_seen_at >= self.release_timeout
        ):
            return self.release()

        return None

    def lost_duration(self, now: float) -> float:
        if self.target_id is None or self.last_seen_at is None:
            return 0.0
        return max(0.0, now - self.last_seen_at)


def make_person_state(
    track_id: int,
    args: argparse.Namespace,
    now: float,
) -> PersonGestureState:
    arm_recognizers = {
        side: ContinuousOscillationRecognizer(
            gesture_window=args.arm_gesture_window,
            cooldown=args.arm_cooldown,
            min_stroke=args.arm_min_stroke_ratio,
            min_strokes=args.arm_min_strokes,
            smoothing_alpha=args.arm_smoothing_alpha,
            min_trigger_interval=args.gesture_min_interval,
            max_signal_gap=args.signal_max_gap,
            circular=False,
        )
        for side in ("LEFT", "RIGHT")
    }

    wrist_recognizers = {
        side: ContinuousOscillationRecognizer(
            gesture_window=args.wrist_gesture_window,
            cooldown=args.wrist_cooldown,
            min_stroke=args.wrist_min_stroke_angle,
            min_strokes=args.wrist_min_strokes,
            smoothing_alpha=args.wrist_smoothing_alpha,
            min_trigger_interval=args.gesture_min_interval,
            max_signal_gap=args.signal_max_gap,
            circular=True,
        )
        for side in ("LEFT", "RIGHT")
    }

    selector = HybridActiveArmSelector(
        mode=args.arm,
        arm_motion_threshold=args.arm_motion_threshold,
        hand_motion_threshold=args.hand_motion_threshold,
        dominance_ratio=args.motion_dominance,
        select_frames=args.arm_select_frames,
        lock_timeout=args.arm_lock_timeout,
        invalid_timeout=args.arm_invalid_timeout,
        motion_alpha=args.motion_alpha,
    )

    wave_recognizer = PoseWaveRecognizer(
        window_seconds=args.wave_window,
        motion_hold_seconds=args.wave_motion_hold,
        static_hold_seconds=args.wave_static_hold,
        cooldown_seconds=args.wave_cooldown,
        min_lateral_range=args.wave_min_lateral_range,
        min_direction_changes=args.wave_min_direction_changes,
        direction_delta=args.wave_direction_delta,
        lateral_dominance=args.wave_lateral_dominance,
        down_reset_seconds=args.wave_down_reset,
        history_gap_reset_seconds=args.wave_history_gap_reset,
        min_kp_conf=args.wave_min_kp_conf,
    )

    return PersonGestureState(
        track_id=track_id,
        arm_recognizers=arm_recognizers,
        wrist_recognizers=wrist_recognizers,
        selector=selector,
        wave_recognizer=wave_recognizer,
        created_at=now,
        last_seen=now,
        seen_frames=0,
    )


def compute_palm_front_score(
    hand_landmarks: np.ndarray,
) -> float:
    """
    Return how directly the hand plane faces the camera, independent of
    whether the visible surface is the palm or the back of the hand.
    """
    wrist = hand_landmarks[0].astype(np.float64)
    index_mcp = hand_landmarks[5].astype(np.float64)
    pinky_mcp = hand_landmarks[17].astype(np.float64)

    vector_index = index_mcp - wrist
    vector_pinky = pinky_mcp - wrist
    normal = np.cross(vector_index, vector_pinky)

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-6:
        return 0.0

    return float(
        np.clip(abs(float(normal[2])) / normal_norm, 0.0, 1.0)
    )


def compute_hand_surface_score(
    hand_landmarks: np.ndarray,
    arm_name: str,
    invert_sign: bool,
) -> float:
    """
    Signed palm/back score from the orientation of the wrist-index-pinky
    triangle.

    The landmark ordering has opposite image orientation for anatomical
    LEFT and RIGHT hands, so the score is corrected using the YOLO arm side.

    After correction:
        positive -> PALM faces the camera
        negative -> BACK faces the camera

    Some camera pipelines mirror the frame. In that case use
    --invert-hand-surface.
    """
    wrist = hand_landmarks[0].astype(np.float64)
    index_mcp = hand_landmarks[5].astype(np.float64)
    pinky_mcp = hand_landmarks[17].astype(np.float64)

    vector_index = index_mcp - wrist
    vector_pinky = pinky_mcp - wrist
    normal = np.cross(vector_index, vector_pinky)

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-6:
        return 0.0

    signed_front = float(normal[2]) / normal_norm

    # Anatomical hands have opposite landmark winding.
    # The current camera/MediaPipe coordinate convention has the opposite
    # winding from the initial assumption. Flip the anatomical correction so
    # the semantic labels are:
    #   PALM -> WAVE
    #   BACK -> COME
    side_correction = -1.0 if arm_name == "RIGHT" else 1.0
    score = side_correction * signed_front

    if invert_sign:
        score = -score

    return float(np.clip(score, -1.0, 1.0))


def hand_features_from_landmarks(
    hand_landmarks: np.ndarray,
    arm: ArmFeatures,
    arm_name: str,
    match_distance: float,
    wrist_deadband_angle: float,
    invert_hand_surface: bool,
) -> ExtendedHandFeatures:
    if (
        not arm.valid
        or not arm.side_ok
        or arm.elbow_xy is None
        or arm.wrist_xy is None
    ):
        return ExtendedHandFeatures(valid=False)

    hand_wrist = hand_landmarks[0, :2]
    middle_mcp = hand_landmarks[9, :2]
    index_mcp = hand_landmarks[5, :2]
    pinky_mcp = hand_landmarks[17, :2]

    forearm_vector = arm.wrist_xy - arm.elbow_xy
    hand_vector = middle_mcp - hand_wrist
    angle = signed_angle_degrees(forearm_vector, hand_vector)

    if angle <= -wrist_deadband_angle:
        hand_state = "NEG"
    elif angle >= wrist_deadband_angle:
        hand_state = "POS"
    else:
        hand_state = "CENTER"

    palm_size = distance(index_mcp, pinky_mcp) / max(
        arm.body_scale,
        1.0,
    )

    return ExtendedHandFeatures(
        valid=True,
        angle=angle,
        state=hand_state,
        match_distance=match_distance,
        palm_size=palm_size,
        wrist_xy=hand_wrist.copy(),
        middle_mcp_xy=middle_mcp.copy(),
        landmarks_px=hand_landmarks,
        palm_front_score=compute_palm_front_score(hand_landmarks),
        hand_surface_score=compute_hand_surface_score(
            hand_landmarks=hand_landmarks,
            arm_name=arm_name,
            invert_sign=invert_hand_surface,
        ),
    )


def assign_hands_uniquely(
    hands_px: list[np.ndarray],
    person_arms: Dict[int, Dict[str, ArmFeatures]],
    max_match_ratio: float,
    wrist_deadband_angle: float,
    invert_hand_surface: bool,
) -> Dict[int, Dict[str, ExtendedHandFeatures]]:
    """
    One MediaPipe hand can belong to only one person-arm.
    """
    assignments: Dict[int, Dict[str, ExtendedHandFeatures]] = {
        track_id: {
            "LEFT": ExtendedHandFeatures(valid=False),
            "RIGHT": ExtendedHandFeatures(valid=False),
        }
        for track_id in person_arms
    }

    candidates: list[Tuple[float, int, int, str]] = []

    for hand_index, hand in enumerate(hands_px):
        hand_wrist = hand[0, :2]

        for track_id, arms in person_arms.items():
            for arm_name, arm in arms.items():
                if (
                    not arm.valid
                    or not arm.side_ok
                    or arm.wrist_xy is None
                ):
                    continue

                normalized_distance = distance(
                    hand_wrist,
                    arm.wrist_xy,
                ) / max(arm.body_scale, 1.0)

                if normalized_distance <= max_match_ratio:
                    candidates.append(
                        (
                            normalized_distance,
                            hand_index,
                            track_id,
                            arm_name,
                        )
                    )

    candidates.sort(key=lambda item: item[0])

    used_hands: set[int] = set()
    used_person_arms: set[Tuple[int, str]] = set()

    for match_distance, hand_index, track_id, arm_name in candidates:
        slot = (track_id, arm_name)

        if hand_index in used_hands or slot in used_person_arms:
            continue

        assignments[track_id][arm_name] = (
            hand_features_from_landmarks(
                hand_landmarks=hands_px[hand_index],
                arm=person_arms[track_id][arm_name],
                arm_name=arm_name,
                match_distance=match_distance,
                wrist_deadband_angle=wrist_deadband_angle,
                invert_hand_surface=invert_hand_surface,
            )
        )

        used_hands.add(hand_index)
        used_person_arms.add(slot)

    return assignments


def update_motion_context(
    state: PersonGestureState,
    arms: Dict[str, ArmFeatures],
    hands: Dict[str, ExtendedHandFeatures],
    args: argparse.Namespace,
    now: float,
) -> None:
    for arm_name in ("LEFT", "RIGHT"):
        arm = arms[arm_name]
        hand = hands[arm_name]

        if (
            arm.valid
            and arm.side_ok
            and arm.wrist_xy is not None
        ):
            history = state.arm_motion_history[arm_name]
            history.append(
                ArmMotionSample(
                    timestamp=now,
                    reach_ratio=arm.reach_ratio,
                    wrist_xy=arm.wrist_xy.copy(),
                    body_scale=max(arm.body_scale, 1.0),
                )
            )

            while (
                history
                and now - history[0].timestamp
                > args.arm_gesture_window
            ):
                history.popleft()

        surface_history = state.hand_surface_history[arm_name]

        if hand.valid:
            previous = state.palm_front_ema[arm_name]
            current = hand.palm_front_score

            if previous is None:
                state.palm_front_ema[arm_name] = current
            else:
                alpha = args.palm_ema_alpha
                state.palm_front_ema[arm_name] = (
                    alpha * current
                    + (1.0 - alpha) * previous
                )

            surface_history.append(
                (now, hand.hand_surface_score)
            )

        while (
            surface_history
            and now - surface_history[0][0]
            > args.hand_surface_window
        ):
            surface_history.popleft()


def arm_radial_fraction(
    state: PersonGestureState,
    arm_name: str,
) -> float:
    """
    Compare radial reach motion with total wrist image-plane travel.

    Beckoning is mainly radial (toward/away from the body).
    Greeting waves are mainly tangential/side-to-side.
    """
    history = state.arm_motion_history[arm_name]
    if len(history) < 3:
        return 0.0

    radial_travel = 0.0
    total_xy_travel = 0.0

    for previous, current in zip(history, list(history)[1:]):
        radial_travel += abs(
            current.reach_ratio - previous.reach_ratio
        )

        average_scale = max(
            (previous.body_scale + current.body_scale) / 2.0,
            1.0,
        )
        total_xy_travel += float(
            np.linalg.norm(current.wrist_xy - previous.wrist_xy)
            / average_scale
        )

    if total_xy_travel < 1e-6:
        return 1.0 if radial_travel > 0 else 0.0

    return float(
        np.clip(radial_travel / total_xy_travel, 0.0, 2.0)
    )


def classify_hand_surface(
    state: PersonGestureState,
    arm_name: str,
    args: argparse.Namespace,
    now: float,
) -> Tuple[str, float, int, float]:
    """
    Classify the visible hand surface using recent frames.

    Returns:
        label: PALM / BACK / UNKNOWN
        median_score
        valid sample count
        winning-vote fraction
    """
    samples = [
        score
        for timestamp, score
        in state.hand_surface_history[arm_name]
        if (
            now - timestamp <= args.hand_surface_window
            and abs(score) >= args.hand_surface_sample_threshold
        )
    ]

    if not samples:
        return "UNKNOWN", 0.0, 0, 0.0

    median_score = float(np.median(samples))
    palm_votes = sum(
        score >= args.hand_surface_threshold
        for score in samples
    )
    back_votes = sum(
        score <= -args.hand_surface_threshold
        for score in samples
    )
    winning_votes = max(palm_votes, back_votes)
    consensus = winning_votes / max(len(samples), 1)

    if (
        len(samples) < args.hand_surface_min_samples
        or consensus < args.hand_surface_consensus
    ):
        return (
            "UNKNOWN",
            median_score,
            len(samples),
            consensus,
        )

    if palm_votes > back_votes:
        return "PALM", median_score, len(samples), consensus

    if back_votes > palm_votes:
        return "BACK", median_score, len(samples), consensus

    return "UNKNOWN", median_score, len(samples), consensus


def fuse_motion_and_hand_surface(
    base_action: str,
    state: PersonGestureState,
    arm_name: str,
    args: argparse.Namespace,
    now: float,
    allow_pose_fallback: bool = False,
    ignore_hand_surface: bool = False,
) -> Tuple[
    Optional[str],
    str,
    float,
    int,
    float,
]:
    """
    Surface semantics:
        PALM -> WAVE
        BACK -> COME

    Modes:
        strict: UNKNOWN suppresses the event
        assist: UNKNOWN falls back to motion classification
        off:    ignore hand surface entirely
    """
    if args.hand_surface_mode == "off":
        return base_action, "UNKNOWN", 0.0, 0, 0.0

    if ignore_hand_surface:
        if allow_pose_fallback:
            return base_action, "UNKNOWN", 0.0, 0, 0.0

        return None, "UNKNOWN", 0.0, 0, 0.0

    label, score, sample_count, consensus = (
        classify_hand_surface(
            state=state,
            arm_name=arm_name,
            args=args,
            now=now,
        )
    )

    if label == "PALM":
        return "WAVE", label, score, sample_count, consensus

    if label == "BACK":
        return "COME", label, score, sample_count, consensus

    if args.hand_surface_mode == "assist":
        return (
            base_action,
            label,
            score,
            sample_count,
            consensus,
        )

    if allow_pose_fallback:
        return base_action, label, score, sample_count, consensus

    return None, label, score, sample_count, consensus


def pose_surface_fallback_allowed(
    bbox_height_ratio: float,
    motion_source: str,
    motion_score: float,
    radial_fraction: Optional[float],
    args: argparse.Namespace,
) -> bool:
    """
    Conservative distance fallback for when MediaPipe hand evidence disappears.

    Near-range behavior remains strict. Pose-only fallback is only used below
    a configurable person-box height, and only for motion sources that do not
    depend on MediaPipe hand angles.
    """
    if args.disable_pose_fallback:
        return False

    if bbox_height_ratio > args.pose_fallback_max_box_height:
        return False

    degraded_distance = bbox_height_ratio < args.min_gesture_box_height

    if motion_source == "POSE_LATERAL_OSCILLATION":
        if not degraded_distance:
            return motion_score >= args.pose_fallback_min_wave_score

        min_score = args.degraded_pose_fallback_min_wave_score
        return (
            motion_score >= min_score
            and radial_fraction is not None
            and radial_fraction
            <= args.degraded_pose_fallback_wave_max_radial_fraction
        )

    if motion_source == "POSE_LATERAL_CANDIDATE":
        if degraded_distance:
            return False

        return (
            motion_score
            >= args.pose_fallback_min_wave_candidate_score
        )

    if motion_source == "ARM_REACH_OSCILLATION":
        if not degraded_distance:
            return True

        min_score = args.degraded_pose_fallback_min_come_score
        return (
            motion_score >= min_score
            and radial_fraction is not None
            and radial_fraction
            >= args.degraded_pose_fallback_come_min_radial_fraction
        )

    return False


def mid_distance_come_update_overrides(
    person: PersonObservation,
    args: argparse.Namespace,
) -> Dict[str, float | int]:
    if args.disable_mid_come_adaptive:
        return {}

    if person.bbox_height_ratio < args.min_gesture_box_height:
        return {}

    if person.bbox_height_ratio > args.mid_come_max_box_height:
        return {}

    return {
        "min_stroke": args.mid_come_min_stroke_ratio,
        "min_trigger_interval": args.mid_come_min_trigger_interval,
        "min_strokes": args.mid_come_min_strokes,
    }


def arm_usable_for_mid_come(
    arm: ArmFeatures,
    use_mid_come_adaptive: bool,
    args: argparse.Namespace,
) -> bool:
    if not arm.valid:
        return False

    if arm.side_ok:
        return True

    return (
        use_mid_come_adaptive
        and args.mid_come_allow_cross_body
    )


def update_person_gesture(
    state: PersonGestureState,
    person: PersonObservation,
    keypoints: np.ndarray,
    arms: Dict[str, ArmFeatures],
    hands: Dict[str, ExtendedHandFeatures],
    args: argparse.Namespace,
    now: float,
) -> Optional[GestureEvent]:
    update_motion_context(
        state=state,
        arms=arms,
        hands=hands,
        args=args,
        now=now,
    )

    # Per-person cross-action cooldown. During this period none of the
    # competing action paths may accumulate a new gesture trajectory.
    if now < state.action_cooldown_until:
        return None

    # -----------------------------------------------------------------
    # Explicit greeting-wave path, adapted from the colleague's code.
    # It is evaluated first and is authoritative for WAVE.
    # -----------------------------------------------------------------
    wave_arm = state.wave_recognizer.update(
        keypoints=keypoints,
        arms=arms,
        now=now,
    )

    if wave_arm is not None:
        state.arm_recognizers[wave_arm].reset()
        state.wrist_recognizers[wave_arm].reset()
        state.selector.release()
        state.active_arm = wave_arm

        motion_source = "POSE_LATERAL_OSCILLATION"
        score = state.wave_recognizer.event_score(wave_arm)
        degraded_distance = (
            person.bbox_height_ratio < args.min_gesture_box_height
        )
        pose_only_distance = (
            person.bbox_height_ratio
            <= args.pose_only_hand_surface_max_box_height
        )
        radial_fraction = (
            arm_radial_fraction(state, wave_arm)
            if degraded_distance
            else None
        )
        allow_pose_fallback = pose_surface_fallback_allowed(
            bbox_height_ratio=person.bbox_height_ratio,
            motion_source=motion_source,
            motion_score=score,
            radial_fraction=radial_fraction,
            args=args,
        )

        action, surface, surface_score, sample_count, consensus = (
            fuse_motion_and_hand_surface(
                base_action="WAVE",
                state=state,
                arm_name=wave_arm,
                args=args,
                now=now,
                allow_pose_fallback=allow_pose_fallback,
                ignore_hand_surface=(
                    pose_only_distance and allow_pose_fallback
                ),
            )
        )

        if action is None:
            state.wave_recognizer.reset_arm(wave_arm)
            if args.debug_overlay:
                print(
                    f"[SURFACE UNCERTAIN] "
                    f"ID={state.track_id} arm={wave_arm} "
                    f"motion={motion_source} "
                    f"score={surface_score:+.3f} "
                    f"samples={sample_count} "
                    f"consensus={consensus:.2f}",
                    flush=True,
                )
            return None

        source = (
            "PALM_SURFACE_WAVE"
            if action == "WAVE" and surface == "PALM"
            else (
                "BACK_SURFACE_COME"
                if action == "COME" and surface == "BACK"
                else (
                    "POSE_ONLY_FALLBACK_WAVE"
                    if allow_pose_fallback and surface == "UNKNOWN"
                    else "POSE_LATERAL_FALLBACK"
                )
            )
        )
        state.last_event_until = now + args.status_hold
        state.last_event_action = action
        state.last_event_source = source
        state.last_event_arm = wave_arm

        return GestureEvent(
            track_id=state.track_id,
            action=action,
            source=source,
            arm=wave_arm,
            score=score,
            timestamp=time.time(),
            bbox_height_ratio=person.bbox_height_ratio,
            hand_surface=surface,
            hand_surface_score=surface_score,
            radial_fraction=radial_fraction,
            motion_source=motion_source,
            pose_fallback=allow_pose_fallback and surface == "UNKNOWN",
        )

    # -----------------------------------------------------------------
    # COME path: reuse the proven V4.1 oscillation recognizers.
    # Arm reach oscillation is directly COME; it is no longer reclassified
    # through the unreliable radial-fraction heuristic.
    # -----------------------------------------------------------------
    active_arm = state.selector.update(
        arms["LEFT"],
        arms["RIGHT"],
        hands["LEFT"],
        hands["RIGHT"],
        now,
    )
    state.active_arm = active_arm

    arm_triggered_by = {
        "LEFT": False,
        "RIGHT": False,
    }
    wrist_triggered_by = {
        "LEFT": False,
        "RIGHT": False,
    }
    mid_come_update_kwargs = mid_distance_come_update_overrides(
        person,
        args,
    )
    use_mid_come_adaptive = bool(mid_come_update_kwargs)

    if active_arm is None:
        # Preserve the beginning of both motions while choosing the arm.
        for arm_name in ("LEFT", "RIGHT"):
            arm = arms[arm_name]
            hand = hands[arm_name]

            if (
                use_mid_come_adaptive
                and arm.valid
                and not arm.arm_height_ok
            ):
                state.arm_recognizers[arm_name].reset()
            elif arm_usable_for_mid_come(
                arm,
                use_mid_come_adaptive,
                args,
            ):
                arm_triggered_by[arm_name] = (
                    state.arm_recognizers[arm_name].update(
                        arm.reach_ratio,
                        now,
                        **mid_come_update_kwargs,
                    )
                )

            if hand.valid:
                wrist_triggered_by[arm_name] = (
                    state.wrist_recognizers[arm_name].update(
                        hand.angle,
                        now,
                    )
                )

        completed_sides = [
            arm_name
            for arm_name in ("LEFT", "RIGHT")
            if (
                arm_triggered_by[arm_name]
                or wrist_triggered_by[arm_name]
            )
        ]

        if completed_sides:
            active_arm = max(
                completed_sides,
                key=state.selector.combined_score,
            )
            state.active_arm = active_arm

            inactive = (
                "RIGHT"
                if active_arm == "LEFT"
                else "LEFT"
            )
            state.arm_recognizers[inactive].reset()
            state.wrist_recognizers[inactive].reset()
    else:
        inactive = (
            "RIGHT"
            if active_arm == "LEFT"
            else "LEFT"
        )
        state.arm_recognizers[inactive].reset()
        state.wrist_recognizers[inactive].reset()

        selected_arm = arms[active_arm]
        selected_hand = hands[active_arm]

        if (
            use_mid_come_adaptive
            and selected_arm.valid
            and not selected_arm.arm_height_ok
        ):
            state.arm_recognizers[active_arm].reset()
        elif arm_usable_for_mid_come(
            selected_arm,
            use_mid_come_adaptive,
            args,
        ):
            arm_triggered_by[active_arm] = (
                state.arm_recognizers[active_arm].update(
                    selected_arm.reach_ratio,
                    now,
                    **mid_come_update_kwargs,
                )
            )

        if selected_hand.valid:
            wrist_triggered_by[active_arm] = (
                state.wrist_recognizers[active_arm].update(
                    selected_hand.angle,
                    now,
                )
            )

    if active_arm is None:
        return None

    arm_triggered = arm_triggered_by[active_arm]
    wrist_triggered = wrist_triggered_by[active_arm]

    if not arm_triggered and not wrist_triggered:
        return None

    if state.wave_recognizer.is_candidate(active_arm):
        base_action = "WAVE"
        motion_source = "POSE_LATERAL_CANDIDATE"
    elif arm_triggered:
        base_action = "COME"
        motion_source = "ARM_REACH_OSCILLATION"
    else:
        base_action = "COME"
        motion_source = "WRIST_ANGLE_OSCILLATION"

    score = state.selector.combined_score(active_arm)
    degraded_distance = (
        person.bbox_height_ratio < args.min_gesture_box_height
    )
    pose_only_distance = (
        person.bbox_height_ratio
        <= args.pose_only_hand_surface_max_box_height
    )
    radial_fraction = (
        arm_radial_fraction(state, active_arm)
        if degraded_distance
        else None
    )
    allow_pose_fallback = pose_surface_fallback_allowed(
        bbox_height_ratio=person.bbox_height_ratio,
        motion_source=motion_source,
        motion_score=score,
        radial_fraction=radial_fraction,
        args=args,
    )

    action, surface, surface_score, sample_count, consensus = (
        fuse_motion_and_hand_surface(
            base_action=base_action,
            state=state,
            arm_name=active_arm,
            args=args,
            now=now,
            allow_pose_fallback=allow_pose_fallback,
            ignore_hand_surface=(
                pose_only_distance and allow_pose_fallback
            ),
        )
    )
    state.selector.release()

    if action is None:
        state.arm_recognizers[active_arm].reset()
        state.wrist_recognizers[active_arm].reset()
        state.wave_recognizer.reset_arm(active_arm)
        if args.debug_overlay:
            print(
                f"[SURFACE UNCERTAIN] "
                f"ID={state.track_id} arm={active_arm} "
                f"motion={motion_source} "
                f"score={surface_score:+.3f} "
                f"samples={sample_count} "
                f"consensus={consensus:.2f}",
                flush=True,
            )
        return None

    source = (
        "PALM_SURFACE_WAVE"
        if action == "WAVE" and surface == "PALM"
        else (
            "BACK_SURFACE_COME"
            if action == "COME" and surface == "BACK"
            else (
                "POSE_ONLY_FALLBACK_COME"
                if (
                    action == "COME"
                    and allow_pose_fallback
                    and surface == "UNKNOWN"
                )
                else (
                    "POSE_ONLY_FALLBACK_WAVE"
                    if (
                        action == "WAVE"
                        and allow_pose_fallback
                        and surface == "UNKNOWN"
                    )
                    else (
                        "MOTION_FALLBACK_WAVE"
                        if action == "WAVE"
                        else "MOTION_FALLBACK_COME"
                    )
                )
            )
        )
    )
    state.last_event_until = now + args.status_hold
    state.last_event_action = action
    state.last_event_source = source
    state.last_event_arm = active_arm

    return GestureEvent(
        track_id=state.track_id,
        action=action,
        source=source,
        arm=active_arm,
        score=score,
        timestamp=time.time(),
        bbox_height_ratio=person.bbox_height_ratio,
        hand_surface=surface,
        hand_surface_score=surface_score,
        radial_fraction=radial_fraction,
        motion_source=motion_source,
        pose_fallback=allow_pose_fallback and surface == "UNKNOWN",
    )


def wrist_motion_metrics(
    state: PersonGestureState,
    arm_name: str,
    now: float,
    window_seconds: float,
) -> Tuple[float, float]:
    """
    Return recent total wrist path in:
        1. raw image pixels,
        2. body-scale-normalized units.

    The raw-pixel requirement prevents tiny distant-pose jitter from being
    amplified into a gesture merely because the body scale is small.
    """
    history = [
        sample
        for sample in state.arm_motion_history[arm_name]
        if now - sample.timestamp <= window_seconds
    ]

    if len(history) < 2:
        return 0.0, 0.0

    travel_px = 0.0
    travel_ratio = 0.0

    for previous, current in zip(history, history[1:]):
        step_px = float(
            np.linalg.norm(current.wrist_xy - previous.wrist_xy)
        )
        average_scale = max(
            (previous.body_scale + current.body_scale) / 2.0,
            1.0,
        )

        travel_px += step_px
        travel_ratio += step_px / average_scale

    return travel_px, travel_ratio


def person_motion_ownership_score(
    state: PersonGestureState,
    now: float,
    args: argparse.Namespace,
) -> Tuple[float, str, float, float]:
    """
    Find the arm with strongest recent physical motion.

    Combining normalized motion with an absolute-pixel factor makes foreground
    deliberate motion dominate small background keypoint jitter.
    """
    best_arm = "LEFT"
    best_score = 0.0
    best_px = 0.0
    best_ratio = 0.0

    for arm_name in ("LEFT", "RIGHT"):
        travel_px, travel_ratio = wrist_motion_metrics(
            state=state,
            arm_name=arm_name,
            now=now,
            window_seconds=args.ownership_window,
        )

        pixel_factor = min(
            travel_px / max(args.min_gesture_wrist_travel_px, 1e-6),
            2.5,
        )
        score = travel_ratio * pixel_factor

        if score > best_score:
            best_arm = arm_name
            best_score = score
            best_px = travel_px
            best_ratio = travel_ratio

    return best_score, best_arm, best_px, best_ratio


def event_attribution_metrics(
    event: GestureEvent,
    state: PersonGestureState,
    now: float,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    travel_px, travel_ratio = wrist_motion_metrics(
        state=state,
        arm_name=event.arm,
        now=now,
        window_seconds=args.ownership_window,
    )

    pixel_factor = min(
        travel_px / max(args.min_gesture_wrist_travel_px, 1e-6),
        2.5,
    )
    ownership_score = travel_ratio * pixel_factor

    return ownership_score, travel_px, travel_ratio


def event_source_is_eligible(
    event: GestureEvent,
    person: PersonObservation,
    state: PersonGestureState,
    global_max_ownership: float,
    now: float,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    ownership_score, travel_px, travel_ratio = (
        event_attribution_metrics(
            event=event,
            state=state,
            now=now,
            args=args,
        )
    )

    event.ownership_score = ownership_score
    event.wrist_travel_px = travel_px
    event.wrist_travel_ratio = travel_ratio

    track_age = now - state.created_at

    if person.confidence < args.min_gesture_hard_person_conf:
        return (
            False,
            f"person confidence {person.confidence:.2f} "
            f"< hard floor {args.min_gesture_hard_person_conf:.2f}",
        )

    if person.bbox_height_ratio < args.min_gesture_hard_box_height:
        return (
            False,
            f"box height {person.bbox_height_ratio:.3f} "
            f"< hard floor {args.min_gesture_hard_box_height:.3f}",
        )

    degraded_quality = (
        person.confidence < args.min_gesture_person_conf
        or person.bbox_height_ratio < args.min_gesture_box_height
    )
    event.quality_band = "DEGRADED" if degraded_quality else "NORMAL"

    min_person_conf = args.min_gesture_person_conf
    min_track_age = args.min_gesture_track_age
    min_seen_frames = args.min_gesture_seen_frames
    min_travel_px = args.min_gesture_wrist_travel_px
    min_travel_ratio = args.min_gesture_wrist_travel_ratio
    ownership_dominance_ratio = args.ownership_dominance_ratio

    if degraded_quality:
        min_person_conf = args.min_gesture_hard_person_conf
        min_track_age = max(
            min_track_age,
            args.degraded_gesture_min_track_age,
        )
        min_seen_frames = max(
            min_seen_frames,
            args.degraded_gesture_seen_frames,
        )
        min_travel_px = args.degraded_min_gesture_wrist_travel_px
        min_travel_ratio = max(
            min_travel_ratio,
            args.degraded_min_gesture_wrist_travel_ratio,
        )
        ownership_dominance_ratio = max(
            ownership_dominance_ratio,
            args.degraded_ownership_dominance_ratio,
        )

    if person.confidence < min_person_conf:
        return (
            False,
            f"person confidence {person.confidence:.2f} "
            f"< {min_person_conf:.2f}",
        )

    if track_age < min_track_age:
        return (
            False,
            f"track age {track_age:.2f}s "
            f"< {min_track_age:.2f}s",
        )

    if state.seen_frames < min_seen_frames:
        return (
            False,
            f"seen frames {state.seen_frames} "
            f"< {min_seen_frames}",
        )

    if travel_px < min_travel_px:
        return (
            False,
            f"wrist travel {travel_px:.1f}px "
            f"< {min_travel_px:.1f}px",
        )

    if travel_ratio < min_travel_ratio:
        return (
            False,
            f"normalized travel {travel_ratio:.2f} "
            f"< {min_travel_ratio:.2f}",
        )

    if (
        global_max_ownership > 0
        and ownership_score
        < ownership_dominance_ratio * global_max_ownership
    ):
        return (
            False,
            f"ownership {ownership_score:.2f} is weaker than "
            f"global max {global_max_ownership:.2f}",
        )

    return True, "accepted"


def direction_from_center(
    center_x: float,
    deadband: float,
) -> str:
    error = center_x - 0.5

    if error < -deadband:
        return "LEFT"
    if error > deadband:
        return "RIGHT"
    return "CENTER"


def draw_label_box(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    top = max(0, y - text_height - baseline - 8)
    right = min(image.shape[1] - 1, x + text_width + 12)

    cv2.rectangle(
        image,
        (x, top),
        (right, y + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 6, y - 5),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_person_overlay(
    output: np.ndarray,
    person: PersonObservation,
    state: PersonGestureState,
    arms: Dict[str, ArmFeatures],
    hands: Dict[str, ExtendedHandFeatures],
    target_id: Optional[int],
    now: float,
    deadband: float,
    debug_overlay: bool,
    args: argparse.Namespace,
) -> None:
    x1, y1, x2, y2 = person.bbox
    is_target = person.track_id == target_id

    event_text: Optional[str] = None
    if now < state.last_event_until:
        event_text = state.last_event_action

    if is_target:
        color = (0, 255, 0)
        thickness = 4
    elif event_text == "COME":
        color = (0, 220, 255)
        thickness = 3
    elif event_text == "WAVE":
        color = (255, 180, 0)
        thickness = 2
    else:
        color = (0, 220, 255)
        thickness = 2

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    parts = [f"ID {person.track_id}"]

    if is_target:
        parts.append("TARGET")
    if event_text is not None:
        parts.append(event_text)

    draw_label_box(
        output,
        " | ".join(parts),
        x1,
        max(24, y1),
        color,
    )

    if is_target:
        direction = direction_from_center(
            person.center_x,
            deadband,
        )
        direction_text = (
            f"{direction}  error={person.center_x - 0.5:+.2f}"
        )
        cv2.putText(
            output,
            direction_text,
            (x1, min(output.shape[0] - 115, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    if debug_overlay:
        active = state.active_arm or "NONE"
        wave_left = state.wave_recognizer.metrics["LEFT"]
        wave_right = state.wave_recognizer.metrics["RIGHT"]

        ownership_score, owner_arm, owner_px, owner_ratio = (
            person_motion_ownership_score(
                state=state,
                now=now,
                args=type(
                    "_DebugArgs",
                    (),
                    {
                        "ownership_window": 1.20,
                        "min_gesture_wrist_travel_px": 18.0,
                    },
                )(),
            )
        )

        cooldown_left = state.cooldown_remaining(now)

        debug_text_1 = (
            f"active={active} "
            f"motion-owner={owner_arm} "
            f"own={ownership_score:.2f} "
            f"px={owner_px:.0f} "
            f"| cooldown={cooldown_left:.1f}s"
        )
        left_surface, left_score, _, left_consensus = (
            classify_hand_surface(
                state=state,
                arm_name="LEFT",
                args=args,
                now=now,
            )
        )
        right_surface, right_score, _, right_consensus = (
            classify_hand_surface(
                state=state,
                arm_name="RIGHT",
                args=args,
                now=now,
            )
        )

        debug_text_2 = (
            f"surface L={left_surface}({left_score:+.2f},"
            f"{left_consensus:.0%}) "
            f"R={right_surface}({right_score:+.2f},"
            f"{right_consensus:.0%}) "
            f"| lateral L={wave_left.lateral_range:.2f} "
            f"R={wave_right.lateral_range:.2f}"
        )

        cv2.putText(
            output,
            debug_text_1,
            (x1, min(output.shape[0] - 88, y2 + 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            debug_text_2,
            (x1, min(output.shape[0] - 70, y2 + 58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

        draw_hand(output, hands["LEFT"], (255, 0, 255))
        draw_hand(output, hands["RIGHT"], (0, 255, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-person tracking with separate WAVE and COME "
            "classification and target locking."
        )
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.60)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)

    parser.add_argument("--min-kp-conf", type=float, default=0.40)
    parser.add_argument("--far-ratio", type=float, default=1.30)
    parser.add_argument("--near-ratio", type=float, default=0.72)
    parser.add_argument("--far-angle", type=float, default=135.0)
    parser.add_argument("--near-angle", type=float, default=90.0)
    parser.add_argument("--cross-body-margin", type=float, default=0.22)

    parser.add_argument("--arm-gesture-window", type=float, default=3.5)
    parser.add_argument("--arm-cooldown", type=float, default=2.5)
    parser.add_argument("--arm-min-stroke-ratio", type=float, default=0.16)
    parser.add_argument("--arm-min-strokes", type=int, default=2)
    parser.add_argument("--arm-smoothing-alpha", type=float, default=0.50)
    parser.add_argument(
        "--disable-mid-come-adaptive",
        action="store_true",
        help="Disable mid-distance adaptive COME stroke thresholds.",
    )
    parser.add_argument(
        "--mid-come-max-box-height",
        type=float,
        default=0.50,
        help=(
            "Adaptive COME thresholds apply only up to this person-box "
            "height ratio and above the normal degraded-distance threshold."
        ),
    )
    parser.add_argument(
        "--mid-come-min-stroke-ratio",
        type=float,
        default=0.085,
        help=(
            "Arm reach stroke ratio used for mid-distance COME only."
        ),
    )
    parser.add_argument(
        "--mid-come-allow-cross-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow mid-distance COME arm-reach updates even when the wrist "
            "slightly crosses the body centerline."
        ),
    )
    parser.add_argument(
        "--mid-come-min-strokes",
        type=int,
        default=2,
        help=(
            "Alternating reach strokes required for mid-distance COME only."
        ),
    )
    parser.add_argument(
        "--mid-come-min-trigger-interval",
        type=float,
        default=0.10,
        help=(
            "Minimum alternating-stroke interval used for mid-distance COME "
            "only."
        ),
    )
    # Explicit greeting-wave detector adapted from the colleague's code.
    parser.add_argument("--wave-window", type=float, default=0.90)
    parser.add_argument("--wave-motion-hold", type=float, default=0.18)
    parser.add_argument(
        "--wave-static-hold",
        type=float,
        default=0.0,
        help=(
            "0 disables raised-hand-only WAVE triggering. Keep disabled "
            "when separating WAVE from COME."
        ),
    )
    parser.add_argument("--wave-cooldown", type=float, default=0.80)
    parser.add_argument(
        "--wave-min-lateral-range",
        type=float,
        default=0.075,
    )
    parser.add_argument(
        "--wave-min-direction-changes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--wave-direction-delta",
        type=float,
        default=0.008,
    )
    parser.add_argument(
        "--wave-lateral-dominance",
        type=float,
        default=1.25,
        help=(
            "Lateral range must exceed this multiple of reach-range "
            "to count as greeting WAVE."
        ),
    )
    parser.add_argument("--wave-down-reset", type=float, default=0.20)
    parser.add_argument(
        "--wave-history-gap-reset",
        type=float,
        default=0.18,
        help=(
            "Clear accumulated WAVE motion after the hand is not raised for "
            "this long, preventing stale lateral motion from triggering on "
            "the next small movement."
        ),
    )
    parser.add_argument("--wave-min-kp-conf", type=float, default=0.25)

    parser.add_argument(
        "--arm",
        choices=("auto", "left", "right"),
        default="auto",
    )
    parser.add_argument("--arm-motion-threshold", type=float, default=0.018)
    parser.add_argument("--hand-motion-threshold", type=float, default=0.050)
    parser.add_argument("--motion-dominance", type=float, default=1.10)
    parser.add_argument("--arm-select-frames", type=int, default=1)
    parser.add_argument("--arm-lock-timeout", type=float, default=5.0)
    parser.add_argument("--arm-invalid-timeout", type=float, default=0.8)
    parser.add_argument("--motion-alpha", type=float, default=0.45)

    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--hand-detection-conf", type=float, default=0.50)
    parser.add_argument("--hand-tracking-conf", type=float, default=0.50)
    parser.add_argument("--hand-match-ratio", type=float, default=0.65)

    parser.add_argument("--wrist-deadband-angle", type=float, default=10.0)
    parser.add_argument("--wrist-gesture-window", type=float, default=3.0)
    parser.add_argument("--wrist-cooldown", type=float, default=2.0)
    parser.add_argument("--wrist-min-stroke-angle", type=float, default=7.0)
    parser.add_argument("--wrist-min-strokes", type=int, default=2)
    parser.add_argument("--wrist-smoothing-alpha", type=float, default=0.60)
    parser.add_argument("--palm-ema-alpha", type=float, default=0.30)
    parser.add_argument(
        "--hand-surface-mode",
        choices=("strict", "assist", "off"),
        default="strict",
        help=(
            "strict: PALM->WAVE, BACK->COME, UNKNOWN suppressed; "
            "assist: UNKNOWN falls back to motion; off: ignore surface."
        ),
    )
    parser.add_argument(
        "--hand-surface-window",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--hand-surface-min-samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--hand-surface-sample-threshold",
        type=float,
        default=0.06,
        help="Discard nearly edge-on hand frames below this magnitude.",
    )
    parser.add_argument(
        "--hand-surface-threshold",
        type=float,
        default=0.12,
        help="Signed score required for a PALM or BACK vote.",
    )
    parser.add_argument(
        "--hand-surface-consensus",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--invert-hand-surface",
        action="store_true",
        help=(
            "Swap PALM/BACK labels when the camera pipeline is mirrored "
            "or the displayed debug labels are reversed."
        ),
    )
    parser.add_argument(
        "--disable-pose-fallback",
        action="store_true",
        help=(
            "Disable conservative pose-only fallback when hand surface is "
            "unknown at mid/far distance."
        ),
    )
    parser.add_argument(
        "--pose-fallback-max-box-height",
        type=float,
        default=0.50,
        help=(
            "Pose-only surface fallback is allowed only at or below this "
            "person-box height ratio."
        ),
    )
    parser.add_argument(
        "--pose-only-hand-surface-max-box-height",
        type=float,
        default=0.42,
        help=(
            "At or below this person-box height, accepted pose fallback "
            "ignores hand-surface labels because MediaPipe hand orientation "
            "is often unstable."
        ),
    )
    parser.add_argument(
        "--pose-fallback-min-wave-score",
        type=float,
        default=1.25,
        help=(
            "Minimum explicit pose-wave score required when falling back "
            "without hand-surface evidence."
        ),
    )
    parser.add_argument(
        "--pose-fallback-min-come-score",
        type=float,
        default=1.20,
        help=(
            "Minimum active-arm motion score required for pose-only COME "
            "fallback."
        ),
    )
    parser.add_argument(
        "--pose-fallback-min-wave-candidate-score",
        type=float,
        default=1.35,
        help=(
            "Minimum active-arm score required for non-degraded pose-wave "
            "candidate fallback when hand surface is unknown."
        ),
    )
    parser.add_argument(
        "--degraded-pose-fallback-min-wave-score",
        type=float,
        default=1.75,
        help=(
            "Minimum explicit pose-wave score required for pose-only "
            "fallback below the normal-quality box-height threshold."
        ),
    )
    parser.add_argument(
        "--degraded-pose-fallback-min-come-score",
        type=float,
        default=1.35,
        help=(
            "Minimum active-arm motion score required for pose-only COME "
            "fallback below the normal-quality box-height threshold."
        ),
    )
    parser.add_argument(
        "--degraded-pose-fallback-wave-max-radial-fraction",
        type=float,
        default=0.40,
        help=(
            "Far/degraded pose-only WAVE fallback is rejected above this "
            "reach/xy motion fraction."
        ),
    )
    parser.add_argument(
        "--degraded-pose-fallback-come-min-radial-fraction",
        type=float,
        default=0.35,
        help=(
            "Far/degraded pose-only COME fallback is rejected below this "
            "reach/xy motion fraction."
        ),
    )

    parser.add_argument("--gesture-min-interval", type=float, default=0.18)
    parser.add_argument("--signal-max-gap", type=float, default=0.70)
    parser.add_argument("--status-hold", type=float, default=2.2)
    parser.add_argument(
        "--action-cooldown",
        type=float,
        default=1.8,
        help=(
            "Per-person cooldown shared by WAVE and COME. After either "
            "action is accepted, all action detectors for that person are "
            "cleared and blocked for this many seconds."
        ),
    )

    # Gesture-source ownership and safety gates.
    parser.add_argument("--ownership-window", type=float, default=1.20)
    parser.add_argument(
        "--min-gesture-person-conf",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--min-gesture-hard-person-conf",
        type=float,
        default=0.35,
        help=(
            "Absolute person-confidence floor. Below this, gesture events "
            "are always rejected."
        ),
    )
    parser.add_argument(
        "--min-gesture-box-height",
        type=float,
        default=0.14,
        help=(
            "Normal-quality person-box height as a fraction of frame "
            "height. Below this, degraded/far-distance gates are used."
        ),
    )
    parser.add_argument(
        "--min-gesture-hard-box-height",
        type=float,
        default=0.06,
        help=(
            "Absolute person-box height floor. Below this, gesture events "
            "are always rejected."
        ),
    )
    parser.add_argument(
        "--min-gesture-track-age",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--min-gesture-seen-frames",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-gesture-wrist-travel-px",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--min-gesture-wrist-travel-ratio",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--ownership-dominance-ratio",
        type=float,
        default=0.60,
        help=(
            "An event source must own at least this fraction of the "
            "strongest current person-motion score."
        ),
    )
    parser.add_argument(
        "--degraded-gesture-min-track-age",
        type=float,
        default=0.90,
        help=(
            "Minimum track age used when confidence or box height is below "
            "the normal-quality threshold."
        ),
    )
    parser.add_argument(
        "--degraded-gesture-seen-frames",
        type=int,
        default=10,
        help=(
            "Minimum seen frames used when confidence or box height is "
            "below the normal-quality threshold."
        ),
    )
    parser.add_argument(
        "--degraded-min-gesture-wrist-travel-px",
        type=float,
        default=10.0,
        help=(
            "Absolute wrist-travel requirement in degraded/far-distance "
            "mode. Lower than the normal threshold because far motion has "
            "fewer pixels."
        ),
    )
    parser.add_argument(
        "--degraded-min-gesture-wrist-travel-ratio",
        type=float,
        default=0.30,
        help=(
            "Normalized wrist-travel requirement in degraded/far-distance "
            "mode. Higher than normal to resist small pose jitter."
        ),
    )
    parser.add_argument(
        "--degraded-ownership-dominance-ratio",
        type=float,
        default=0.85,
        help=(
            "Ownership dominance required in degraded/far-distance mode."
        ),
    )

    parser.add_argument("--align-deadband", type=float, default=0.08)
    parser.add_argument("--track-state-timeout", type=float, default=2.0)
    parser.add_argument("--target-release-timeout", type=float, default=3.0)
    parser.add_argument("--print-target-json", action="store_true")
    parser.add_argument("--debug-overlay", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device is None:
        device: int | str = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Model: {args.model}")
    print(f"Tracker: {args.tracker}")
    print(f"Device: {device}")
    print(
        "Hand surface semantics: PALM->WAVE, BACK->COME "
        f"(mode={args.hand_surface_mode})."
    )
    print(
        "Pose fallback: "
        f"{'off' if args.disable_pose_fallback else 'on'} "
        f"(max_box={args.pose_fallback_max_box_height:.2f})."
    )
    print(
        "Gesture quality gates: "
        f"normal_box={args.min_gesture_box_height:.2f}, "
        f"hard_box={args.min_gesture_hard_box_height:.2f}, "
        f"normal_conf={args.min_gesture_person_conf:.2f}, "
        f"hard_conf={args.min_gesture_hard_person_conf:.2f}."
    )
    print(
        "Mid COME adaptive: "
        f"{'off' if args.disable_mid_come_adaptive else 'on'} "
        f"(max_box={args.mid_come_max_box_height:.2f}, "
        f"stroke={args.mid_come_min_stroke_ratio:.2f}, "
        f"strokes={args.mid_come_min_strokes}, "
        f"interval={args.mid_come_min_trigger_interval:.2f}s, "
        f"cross_body="
        f"{'on' if args.mid_come_allow_cross_body else 'off'})."
    )
    print("Only COME locks its source person as the target.")
    print("Keys: Q/Esc quit | R release target | C clear all states.")

    model = YOLO(args.model)
    hand_detector = MediaPipeHandDetector(
        max_hands=args.max_hands,
        detection_confidence=args.hand_detection_conf,
        tracking_confidence=args.hand_tracking_conf,
    )
    cap = open_camera(
        args.camera,
        args.width,
        args.height,
        args.fps,
    )

    person_states: Dict[int, PersonGestureState] = {}
    target_manager = TargetManager(args.target_release_timeout)

    global_status_until = 0.0
    global_action = "NONE"
    global_source_person_id: Optional[int] = None
    global_detail = "No new gesture"
    last_target_json = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.", file=sys.stderr)
                break

            now = time.monotonic()
            frame_height, frame_width = frame.shape[:2]

            result = model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            if args.debug_overlay:
                try:
                    output = result.plot(
                        boxes=False,
                        labels=False,
                    )
                except TypeError:
                    output = result.plot()
            else:
                output = frame.copy()

            hands_px = hand_detector.process(frame)

            people: Dict[int, PersonObservation] = {}
            person_arms: Dict[int, Dict[str, ArmFeatures]] = {}

            if (
                result.boxes is not None
                and result.boxes.id is not None
                and result.keypoints is not None
                and result.keypoints.data is not None
            ):
                track_ids = result.boxes.id.int().cpu().tolist()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                all_keypoints = result.keypoints.data.cpu().numpy()

                for track_id, box, confidence, keypoints in zip(
                    track_ids,
                    boxes_xyxy,
                    confidences,
                    all_keypoints,
                ):
                    x1, y1, x2, y2 = map(int, box)

                    bbox_width = max(0, x2 - x1)
                    bbox_height = max(0, y2 - y1)

                    people[track_id] = PersonObservation(
                        track_id=track_id,
                        confidence=float(confidence),
                        bbox=(x1, y1, x2, y2),
                        center_x=float(
                            ((x1 + x2) / 2.0) / frame_width
                        ),
                        center_y=float(
                            ((y1 + y2) / 2.0) / frame_height
                        ),
                        bbox_height_ratio=float(
                            bbox_height / max(frame_height, 1)
                        ),
                        bbox_area_ratio=float(
                            (bbox_width * bbox_height)
                            / max(frame_width * frame_height, 1)
                        ),
                        keypoints=keypoints,
                    )

                    person_arms[track_id] = {
                        "LEFT": extract_arm_features(
                            keypoints,
                            "left",
                            args.min_kp_conf,
                            args.far_ratio,
                            args.near_ratio,
                            args.far_angle,
                            args.near_angle,
                            args.cross_body_margin,
                        ),
                        "RIGHT": extract_arm_features(
                            keypoints,
                            "right",
                            args.min_kp_conf,
                            args.far_ratio,
                            args.near_ratio,
                            args.far_angle,
                            args.near_angle,
                            args.cross_body_margin,
                        ),
                    }

            hand_assignments = assign_hands_uniquely(
                hands_px=hands_px,
                person_arms=person_arms,
                max_match_ratio=args.hand_match_ratio,
                wrist_deadband_angle=args.wrist_deadband_angle,
                invert_hand_surface=args.invert_hand_surface,
            )

            events: list[GestureEvent] = []

            for track_id in people:
                state = person_states.get(track_id)

                if state is None:
                    state = make_person_state(
                        track_id,
                        args,
                        now,
                    )
                    person_states[track_id] = state

                state.last_seen = now
                state.seen_frames += 1

                event = update_person_gesture(
                    state=state,
                    person=people[track_id],
                    keypoints=people[track_id].keypoints,
                    arms=person_arms[track_id],
                    hands=hand_assignments[track_id],
                    args=args,
                    now=now,
                )
                if event is not None:
                    events.append(event)

            # -------------------------------------------------------------
            # Attribute gestures to the person who actually owns the motion.
            # -------------------------------------------------------------
            ownership_by_person: Dict[int, float] = {}

            for track_id, state in person_states.items():
                if track_id not in people:
                    continue

                ownership_score, _, _, _ = (
                    person_motion_ownership_score(
                        state=state,
                        now=now,
                        args=args,
                    )
                )
                ownership_by_person[track_id] = ownership_score

            global_max_ownership = max(
                ownership_by_person.values(),
                default=0.0,
            )

            accepted_events: list[GestureEvent] = []

            for event in events:
                person = people.get(event.track_id)
                state = person_states.get(event.track_id)

                if person is None or state is None:
                    continue

                eligible, reason = event_source_is_eligible(
                    event=event,
                    person=person,
                    state=state,
                    global_max_ownership=global_max_ownership,
                    now=now,
                    args=args,
                )

                if eligible:
                    accepted_events.append(event)

                    # Cross-action cooldown is per person, not global:
                    # one person's WAVE does not block another person's COME.
                    state.begin_action_cooldown(
                        action=event.action,
                        now=now,
                        cooldown_seconds=args.action_cooldown,
                    )

                    print(
                        f"[ACTION COOLDOWN] ID={event.track_id} "
                        f"after {event.action}: "
                        f"{args.action_cooldown:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[SUPPRESSED FALSE ATTRIBUTION] "
                        f"action={event.action} "
                        f"ID={event.track_id}: {reason}",
                        flush=True,
                    )

                    # Do not allow a rejected person's stale motion trajectory
                    # to trigger again on the next frame.
                    state.arm_recognizers[event.arm].reset()
                    state.wrist_recognizers[event.arm].reset()
                    state.wave_recognizer.reset_arm(event.arm)
                    state.last_event_until = 0.0
                    state.last_event_action = "NONE"
                    state.last_event_source = ""
                    state.last_event_arm = ""

            events = accepted_events

            seen_ids = set(people)
            released_target = target_manager.update_visibility(
                seen_ids,
                now,
            )
            if released_target is not None:
                print(
                    f"[TARGET RELEASED] ID={released_target} "
                    f"after {args.target_release_timeout:.1f}s unseen"
                )

            if events:
                for event in events:
                    print(
                        json.dumps(
                            {
                                "type": "gesture_event",
                                "action": event.action,
                                "source_person_id": event.track_id,
                                "track_id": event.track_id,
                                "source": event.source,
                                "arm": event.arm,
                                "score": round(event.score, 3),
                                "ownership_score": round(
                                    event.ownership_score,
                                    3,
                                ),
                                "wrist_travel_px": round(
                                    event.wrist_travel_px,
                                    1,
                                ),
                                "wrist_travel_ratio": round(
                                    event.wrist_travel_ratio,
                                    3,
                                ),
                                "bbox_height_ratio": round(
                                    event.bbox_height_ratio,
                                    3,
                                ),
                                "hand_surface": event.hand_surface,
                                "hand_surface_score": (
                                    round(
                                        event.hand_surface_score,
                                        3,
                                    )
                                    if event.hand_surface_score
                                    is not None
                                    else None
                                ),
                                "motion_source": event.motion_source,
                                "pose_fallback": event.pose_fallback,
                                "quality_band": event.quality_band,
                                "palm_front_score": (
                                    round(event.palm_front_score, 3)
                                    if event.palm_front_score is not None
                                    else None
                                ),
                                "radial_fraction": (
                                    round(event.radial_fraction, 3)
                                    if event.radial_fraction is not None
                                    else None
                                ),
                                "timestamp": event.timestamp,
                            },
                            ensure_ascii=False,
                        )
                    )

                come_events = [
                    event
                    for event in events
                    if event.action == "COME"
                ]

                accepted_event: Optional[GestureEvent] = None

                if come_events:
                    if target_manager.target_id is None:
                        accepted_event = max(
                            come_events,
                            key=lambda event: event.score,
                        )
                        target_manager.lock(
                            accepted_event.track_id,
                            now,
                        )

                        command = {
                            "type": "robot_command",
                            "action": "COME",
                            "source_person_id": accepted_event.track_id,
                            "target_id": accepted_event.track_id,
                            "task_id": target_manager.task_id,
                            "source": accepted_event.source,
                            "motion_source": accepted_event.motion_source,
                            "pose_fallback": accepted_event.pose_fallback,
                            "quality_band": accepted_event.quality_band,
                            "bbox_height_ratio": round(
                                accepted_event.bbox_height_ratio,
                                3,
                            ),
                            "hand_surface": accepted_event.hand_surface,
                            "hand_surface_score": (
                                round(
                                    accepted_event.hand_surface_score,
                                    3,
                                )
                                if accepted_event.hand_surface_score
                                is not None
                                else None
                            ),
                            "arm": accepted_event.arm,
                            "timestamp": time.time(),
                        }
                        print(
                            json.dumps(
                                command,
                                ensure_ascii=False,
                            )
                        )
                        print(
                            f"[TARGET LOCKED] person "
                            f"ID={accepted_event.track_id}"
                        )
                    else:
                        same_target_events = [
                            event
                            for event in come_events
                            if event.track_id
                            == target_manager.target_id
                        ]
                        if same_target_events:
                            accepted_event = max(
                                same_target_events,
                                key=lambda event: event.score,
                            )

                display_event = (
                    accepted_event
                    if accepted_event is not None
                    else max(events, key=lambda event: event.score)
                )

                global_status_until = now + args.status_hold
                global_action = display_event.action
                global_source_person_id = display_event.track_id

                if accepted_event is not None:
                    global_detail = (
                        f"PERSON ID {accepted_event.track_id} "
                        f"IS THE LOCKED TARGET"
                    )
                elif (
                    display_event.action == "COME"
                    and target_manager.target_id is not None
                    and display_event.track_id
                    != target_manager.target_id
                ):
                    global_detail = (
                        f"COME FROM ID {display_event.track_id} IGNORED; "
                        f"CURRENT TARGET IS ID "
                        f"{target_manager.target_id}"
                    )
                elif display_event.action == "WAVE":
                    global_detail = (
                        f"GREETING WAVE FROM PERSON ID "
                        f"{display_event.track_id}; NO TARGET LOCK"
                    )
                else:
                    global_detail = (
                        f"PERSON ID {display_event.track_id} "
                        f"| {display_event.source}"
                    )

            stale_ids = [
                track_id
                for track_id, state in person_states.items()
                if (
                    now - state.last_seen
                    >= args.track_state_timeout
                    and track_id != target_manager.target_id
                )
            ]
            for track_id in stale_ids:
                del person_states[track_id]

            for track_id, person in people.items():
                draw_person_overlay(
                    output=output,
                    person=person,
                    state=person_states[track_id],
                    arms=person_arms[track_id],
                    hands=hand_assignments[track_id],
                    target_id=target_manager.target_id,
                    now=now,
                    deadband=args.align_deadband,
                    debug_overlay=args.debug_overlay,
                    args=args,
                )

            # Compact top target bar.
            cv2.rectangle(
                output,
                (0, 0),
                (frame_width, 66),
                (0, 0, 0),
                -1,
            )

            if target_manager.target_id is None:
                target_text = (
                    f"TARGET: NONE    PEOPLE: {len(people)}    "
                    "WAITING FOR COME"
                )
                target_color = (0, 255, 255)
            elif target_manager.target_id in people:
                target = people[target_manager.target_id]
                direction = direction_from_center(
                    target.center_x,
                    args.align_deadband,
                )
                target_text = (
                    f"TARGET: PERSON ID {target.track_id}    "
                    f"{direction}    "
                    f"ERROR {target.center_x - 0.5:+.2f}"
                )
                target_color = (0, 255, 0)

                if (
                    args.print_target_json
                    and now - last_target_json >= 0.2
                ):
                    print(
                        json.dumps(
                            {
                                "type": "target_state",
                                "task_id": target_manager.task_id,
                                "target_id": target.track_id,
                                "source_person_id": target.track_id,
                                "visible": True,
                                "center_x": round(
                                    target.center_x,
                                    4,
                                ),
                                "direction_error": round(
                                    target.center_x - 0.5,
                                    4,
                                ),
                                "confidence": round(
                                    target.confidence,
                                    4,
                                ),
                                "timestamp": time.time(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    last_target_json = now
            else:
                lost_duration = target_manager.lost_duration(now)
                target_text = (
                    f"TARGET: PERSON ID "
                    f"{target_manager.target_id} LOST "
                    f"{lost_duration:.1f}s"
                )
                target_color = (0, 165, 255)

                if (
                    args.print_target_json
                    and now - last_target_json >= 0.2
                ):
                    print(
                        json.dumps(
                            {
                                "type": "target_state",
                                "task_id": target_manager.task_id,
                                "target_id": target_manager.target_id,
                                "source_person_id": (
                                    target_manager.target_id
                                ),
                                "visible": False,
                                "lost_duration": round(
                                    lost_duration,
                                    3,
                                ),
                                "timestamp": time.time(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    last_target_json = now

            cv2.putText(
                output,
                target_text,
                (18, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                target_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "R release target    C clear states    Q quit",
                (18, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Large event panel with explicit command source ID.
            if now >= global_status_until:
                global_action = "NONE"
                global_source_person_id = None
                global_detail = "No new gesture"

            if global_action == "COME":
                panel_color = (0, 145, 0)
            elif global_action == "WAVE":
                panel_color = (130, 75, 0)
            else:
                panel_color = (35, 35, 35)

            panel_height = 112
            panel_top = frame_height - panel_height

            cv2.rectangle(
                output,
                (0, panel_top),
                (frame_width, frame_height),
                panel_color,
                -1,
            )

            if global_source_person_id is None:
                main_status = "DETECTED: NONE"
            else:
                main_status = (
                    f"DETECTED: {global_action} "
                    f"FROM PERSON ID {global_source_person_id}"
                )

            cv2.putText(
                output,
                main_status,
                (22, panel_top + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.03,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                global_detail,
                (24, panel_top + 83),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Multi-person COME Tracker V3.4",
                output,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                released = target_manager.release()
                print(f"[MANUAL RELEASE] target ID={released}")

            if key == ord("c"):
                person_states.clear()
                released = target_manager.release()

                global_status_until = 0.0
                global_action = "NONE"
                global_source_person_id = None
                global_detail = "All states cleared"

                print(
                    f"[CLEAR] all states cleared; "
                    f"released target ID={released}"
                )

    finally:
        cap.release()
        hand_detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

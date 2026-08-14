from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import mediapipe as mp
except ImportError:
    mp = None


# COCO Pose 17 keypoints
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class ArmFeatures:
    valid: bool
    state: str = "UNKNOWN"  # FAR / NEAR / MID / UNKNOWN
    reach_ratio: float = 0.0
    elbow_angle: float = 0.0
    wrist_conf: float = 0.0
    body_scale: float = 1.0
    elbow_xy: Optional[np.ndarray] = None
    wrist_xy: Optional[np.ndarray] = None
    side_ok: bool = False
    arm_height_ok: bool = True
    wrist_shoulder_y_ratio: float = 0.0


@dataclass
class HandFeatures:
    valid: bool
    angle: float = 0.0
    state: str = "CENTER"  # NEG / CENTER / POS
    match_distance: float = 999.0
    palm_size: float = 0.0
    wrist_xy: Optional[np.ndarray] = None
    middle_mcp_xy: Optional[np.ndarray] = None
    landmarks_px: Optional[np.ndarray] = None


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:2] + b[:2]) / 2.0


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def signed_angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Signed 2D angle from vector_a to vector_b in [-180, 180]."""
    a = vector_a.astype(np.float64)
    b = vector_b.astype(np.float64)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0

    a /= norm_a
    b /= norm_b
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.dot(a, b))
    return math.degrees(math.atan2(cross, dot))


def wrapped_angle_difference(current: float, previous: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    delta = (current - previous + 180.0) % 360.0 - 180.0
    return abs(delta)


def keypoint_valid(
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


def compute_body_geometry(
    keypoints: np.ndarray,
    min_conf: float,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    if not (
        keypoint_valid(keypoints, LEFT_SHOULDER, min_conf)
        and keypoint_valid(keypoints, RIGHT_SHOULDER, min_conf)
    ):
        return None, None, None, None

    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    shoulder_width = distance(left_shoulder, right_shoulder)
    shoulder_mid = midpoint(left_shoulder, right_shoulder)

    torso_height = 0.0
    hip_y: Optional[float] = None
    torso_center_x = float(shoulder_mid[0])

    if (
        keypoint_valid(keypoints, LEFT_HIP, min_conf)
        and keypoint_valid(keypoints, RIGHT_HIP, min_conf)
    ):
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        hip_mid = midpoint(left_hip, right_hip)
        torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))
        hip_y = float(hip_mid[1])
        torso_center_x = float((shoulder_mid[0] + hip_mid[0]) / 2.0)

    body_scale = max(
        shoulder_width,
        0.50 * torso_height,
        25.0,
    )
    return body_scale, torso_height, hip_y, torso_center_x


def extract_arm_features(
    keypoints: np.ndarray,
    side: str,
    min_kp_conf: float,
    far_ratio: float,
    near_ratio: float,
    far_angle: float,
    near_angle: float,
    cross_body_margin: float,
) -> ArmFeatures:
    if side == "left":
        shoulder_idx = LEFT_SHOULDER
        elbow_idx = LEFT_ELBOW
        wrist_idx = LEFT_WRIST
    elif side == "right":
        shoulder_idx = RIGHT_SHOULDER
        elbow_idx = RIGHT_ELBOW
        wrist_idx = RIGHT_WRIST
    else:
        raise ValueError(f"Unsupported side: {side}")

    required = (shoulder_idx, elbow_idx, wrist_idx)
    if not all(
        keypoint_valid(keypoints, idx, min_kp_conf)
        for idx in required
    ):
        return ArmFeatures(valid=False)

    body_scale, torso_height, hip_y, torso_center_x = compute_body_geometry(
        keypoints,
        min_kp_conf,
    )
    if body_scale is None or torso_center_x is None:
        return ArmFeatures(valid=False)

    shoulder = keypoints[shoulder_idx]
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]

    reach_ratio = distance(wrist, shoulder) / body_scale
    elbow_angle = angle_degrees(shoulder, elbow, wrist)
    wrist_conf = float(wrist[2])
    wrist_shoulder_y_ratio = (
        float(wrist[1]) - float(shoulder[1])
    ) / max(body_scale, 1.0)

    margin_px = cross_body_margin * body_scale
    if side == "left":
        side_ok = (
            float(wrist[0]) >= torso_center_x - margin_px
            and float(elbow[0]) >= torso_center_x - 1.5 * margin_px
        )
    else:
        side_ok = (
            float(wrist[0]) <= torso_center_x + margin_px
            and float(elbow[0]) <= torso_center_x + 1.5 * margin_px
        )

    arm_height_ok = True
    if hip_y is not None:
        margin = max(20.0, 0.20 * float(torso_height or 0.0))
        arm_height_ok = float(wrist[1]) < hip_y + margin

    if not side_ok:
        state = "UNKNOWN"
    elif not arm_height_ok:
        state = "MID"
    elif reach_ratio >= far_ratio and elbow_angle >= far_angle:
        state = "FAR"
    elif reach_ratio <= near_ratio and elbow_angle <= near_angle:
        state = "NEAR"
    else:
        state = "MID"

    return ArmFeatures(
        valid=True,
        state=state,
        reach_ratio=reach_ratio,
        elbow_angle=elbow_angle,
        wrist_conf=wrist_conf,
        body_scale=body_scale,
        elbow_xy=elbow[:2].astype(np.float32),
        wrist_xy=wrist[:2].astype(np.float32),
        side_ok=side_ok,
        arm_height_ok=arm_height_ok,
        wrist_shoulder_y_ratio=wrist_shoulder_y_ratio,
    )


class ContinuousOscillationRecognizer:
    """
    Detect repeated back-and-forth motion in a continuous scalar signal.

    A single one-way movement never triggers. A valid gesture requires two
    completed alternating strokes, for example:

        outward -> inward -> outward
        left -> right -> left
    """

    def __init__(
        self,
        gesture_window: float,
        cooldown: float,
        min_stroke: float,
        min_strokes: int,
        smoothing_alpha: float,
        min_trigger_interval: float,
        max_signal_gap: float,
        circular: bool = False,
    ) -> None:
        self.gesture_window = gesture_window
        self.cooldown = cooldown
        self.min_stroke = min_stroke
        self.min_strokes = min_strokes
        self.smoothing_alpha = smoothing_alpha
        self.min_trigger_interval = min_trigger_interval
        self.max_signal_gap = max_signal_gap
        self.circular = circular

        self.smoothed_value: Optional[float] = None
        self.last_raw_value: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.anchor_value: Optional[float] = None
        self.extreme_value: Optional[float] = None
        self.direction = 0

        self.strokes: Deque[Tuple[int, float]] = deque()
        self.last_trigger_time = -1e9
        self.current_swing = 0.0

    def _reset_motion(self) -> None:
        self.smoothed_value = None
        self.last_raw_value = None
        self.last_update_time = None
        self.anchor_value = None
        self.extreme_value = None
        self.direction = 0
        self.strokes.clear()
        self.current_swing = 0.0

    def reset(self) -> None:
        self._reset_motion()

    @staticmethod
    def _wrapped_delta(current: float, previous: float) -> float:
        return (current - previous + 180.0) % 360.0 - 180.0

    def _smooth(self, raw_value: float) -> float:
        if self.smoothed_value is None:
            self.smoothed_value = raw_value
            self.last_raw_value = raw_value
            return raw_value

        if self.circular:
            assert self.last_raw_value is not None
            delta = self._wrapped_delta(raw_value, self.last_raw_value)
            unwrapped_value = self.smoothed_value + delta
        else:
            unwrapped_value = raw_value

        self.smoothed_value = (
            self.smoothing_alpha * unwrapped_value
            + (1.0 - self.smoothing_alpha) * self.smoothed_value
        )
        self.last_raw_value = raw_value
        return self.smoothed_value

    def _trim_old(self, now: float) -> None:
        while (
            self.strokes
            and now - self.strokes[0][1] > self.gesture_window
        ):
            self.strokes.popleft()

    def _record_stroke(self, direction: int, now: float) -> None:
        if self.strokes and self.strokes[-1][0] == direction:
            return
        self.strokes.append((direction, now))
        self._trim_old(now)

    def update(
        self,
        raw_value: float,
        now: float,
        min_stroke: Optional[float] = None,
        min_trigger_interval: Optional[float] = None,
        min_strokes: Optional[int] = None,
    ) -> bool:
        effective_min_stroke = (
            self.min_stroke if min_stroke is None else min_stroke
        )
        effective_min_trigger_interval = (
            self.min_trigger_interval
            if min_trigger_interval is None
            else min_trigger_interval
        )
        effective_min_strokes = (
            self.min_strokes if min_strokes is None else min_strokes
        )

        if (
            self.last_update_time is not None
            and now - self.last_update_time > self.max_signal_gap
        ):
            self._reset_motion()

        self.last_update_time = now
        value = self._smooth(raw_value)
        self._trim_old(now)

        if self.anchor_value is None:
            self.anchor_value = value
            self.extreme_value = value
            return False

        assert self.extreme_value is not None

        if self.direction == 0:
            displacement = value - self.anchor_value
            self.current_swing = abs(displacement)

            if abs(displacement) >= effective_min_stroke:
                self.direction = 1 if displacement > 0 else -1
                self.extreme_value = value
            return False

        if self.direction > 0:
            if value > self.extreme_value:
                self.extreme_value = value

            reverse_distance = self.extreme_value - value
            self.current_swing = reverse_distance

            if reverse_distance >= effective_min_stroke:
                self._record_stroke(+1, now)
                self.direction = -1
                self.anchor_value = self.extreme_value
                self.extreme_value = value

        else:
            if value < self.extreme_value:
                self.extreme_value = value

            reverse_distance = value - self.extreme_value
            self.current_swing = reverse_distance

            if reverse_distance >= effective_min_stroke:
                self._record_stroke(-1, now)
                self.direction = +1
                self.anchor_value = self.extreme_value
                self.extreme_value = value

        if now - self.last_trigger_time < self.cooldown:
            return False

        if len(self.strokes) >= effective_min_strokes:
            recent = list(self.strokes)[-effective_min_strokes:]
            directions = [direction for direction, _ in recent]
            timestamps = [timestamp for _, timestamp in recent]

            alternating = all(
                previous != current
                for previous, current in zip(directions, directions[1:])
            )
            duration_ok = (
                effective_min_strokes <= 1
                or timestamps[-1] - timestamps[0]
                >= effective_min_trigger_interval
            )

            if alternating and duration_ok:
                self.last_trigger_time = now
                current_value = value

                self._reset_motion()
                self.last_update_time = now
                self.smoothed_value = current_value
                self.last_raw_value = raw_value
                self.anchor_value = current_value
                self.extreme_value = current_value
                return True

        return False

    def history_text(self) -> str:
        sequence = "".join(
            "+" if direction > 0 else "-"
            for direction, _ in self.strokes
        )
        if not sequence:
            sequence = "-"

        return (
            f"seq={sequence} "
            f"n={len(self.strokes)} "
            f"swing={self.current_swing:.2f}"
        )


class MediaPipeHandDetector:
    def __init__(
        self,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> None:
        if mp is None:
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install mediapipe==0.10.14"
            )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self.hands.process(frame_rgb)

        hands_px: list[np.ndarray] = []
        if not result.multi_hand_landmarks:
            return hands_px

        height, width = frame_bgr.shape[:2]
        for hand_landmarks in result.multi_hand_landmarks:
            points = np.array(
                [
                    [
                        landmark.x * width,
                        landmark.y * height,
                        landmark.z * width,
                    ]
                    for landmark in hand_landmarks.landmark
                ],
                dtype=np.float32,
            )
            hands_px.append(points)

        return hands_px

    def close(self) -> None:
        self.hands.close()


def match_hand_to_arm(
    hands_px: list[np.ndarray],
    arm: ArmFeatures,
    max_match_ratio: float,
    wrist_deadband_angle: float,
) -> HandFeatures:
    if (
        not arm.valid
        or not arm.side_ok
        or arm.elbow_xy is None
        or arm.wrist_xy is None
    ):
        return HandFeatures(valid=False)

    best_hand: Optional[np.ndarray] = None
    best_distance = float("inf")

    for hand in hands_px:
        hand_wrist = hand[0, :2]
        match_distance = distance(hand_wrist, arm.wrist_xy) / max(
            arm.body_scale,
            1.0,
        )
        if match_distance < best_distance:
            best_distance = match_distance
            best_hand = hand

    if best_hand is None or best_distance > max_match_ratio:
        return HandFeatures(valid=False, match_distance=best_distance)

    hand_wrist = best_hand[0, :2]
    middle_mcp = best_hand[9, :2]
    index_mcp = best_hand[5, :2]
    pinky_mcp = best_hand[17, :2]

    forearm_vector = arm.wrist_xy - arm.elbow_xy
    hand_vector = middle_mcp - hand_wrist
    angle = signed_angle_degrees(forearm_vector, hand_vector)

    if angle <= -wrist_deadband_angle:
        state = "NEG"
    elif angle >= wrist_deadband_angle:
        state = "POS"
    else:
        state = "CENTER"

    palm_size = distance(index_mcp, pinky_mcp) / max(arm.body_scale, 1.0)

    return HandFeatures(
        valid=True,
        angle=angle,
        state=state,
        match_distance=best_distance,
        palm_size=palm_size,
        wrist_xy=hand_wrist.copy(),
        middle_mcp_xy=middle_mcp.copy(),
        landmarks_px=best_hand,
    )


class HybridActiveArmSelector:
    """
    Lock one active arm using:
      - YOLO wrist translation, or
      - MediaPipe hand-angle change.

    This lets wrist-only waving select an arm even when the YOLO wrist point
    barely moves.
    """

    def __init__(
        self,
        mode: str,
        arm_motion_threshold: float,
        hand_motion_threshold: float,
        dominance_ratio: float,
        select_frames: int,
        lock_timeout: float,
        invalid_timeout: float,
        motion_alpha: float,
    ) -> None:
        self.mode = mode
        self.arm_motion_threshold = arm_motion_threshold
        self.hand_motion_threshold = hand_motion_threshold
        self.dominance_ratio = dominance_ratio
        self.select_frames = select_frames
        self.lock_timeout = lock_timeout
        self.invalid_timeout = invalid_timeout
        self.motion_alpha = motion_alpha

        self.previous_wrist: Dict[str, Optional[np.ndarray]] = {
            "LEFT": None,
            "RIGHT": None,
        }
        self.previous_hand_angle: Dict[str, Optional[float]] = {
            "LEFT": None,
            "RIGHT": None,
        }
        self.arm_motion_score = {"LEFT": 0.0, "RIGHT": 0.0}
        self.hand_motion_score = {"LEFT": 0.0, "RIGHT": 0.0}

        self.candidate: Optional[str] = None
        self.candidate_frames = 0
        self.locked_arm: Optional[str] = (
            mode.upper() if mode in {"left", "right"} else None
        )
        self.locked_at = time.monotonic() if self.locked_arm else None
        self.last_valid_at = time.monotonic()

    def reset(self) -> None:
        self.previous_wrist = {"LEFT": None, "RIGHT": None}
        self.previous_hand_angle = {"LEFT": None, "RIGHT": None}
        self.arm_motion_score = {"LEFT": 0.0, "RIGHT": 0.0}
        self.hand_motion_score = {"LEFT": 0.0, "RIGHT": 0.0}
        self.candidate = None
        self.candidate_frames = 0
        self.locked_arm = (
            self.mode.upper() if self.mode in {"left", "right"} else None
        )
        self.locked_at = time.monotonic() if self.locked_arm else None
        self.last_valid_at = time.monotonic()

    def release(self) -> None:
        if self.mode == "auto":
            self.locked_arm = None
            self.locked_at = None
            self.candidate = None
            self.candidate_frames = 0

    def _update_scores(
        self,
        arm_name: str,
        arm: ArmFeatures,
        hand: HandFeatures,
    ) -> None:
        instant_arm_motion = 0.0
        previous_wrist = self.previous_wrist[arm_name]

        if (
            arm.valid
            and arm.side_ok
            and arm.wrist_xy is not None
            and previous_wrist is not None
        ):
            instant_arm_motion = float(
                np.linalg.norm(arm.wrist_xy - previous_wrist)
                / max(arm.body_scale, 1.0)
            )

        self.arm_motion_score[arm_name] = (
            self.motion_alpha * instant_arm_motion
            + (1.0 - self.motion_alpha) * self.arm_motion_score[arm_name]
        )

        if arm.valid and arm.side_ok and arm.wrist_xy is not None:
            self.previous_wrist[arm_name] = arm.wrist_xy.copy()

        instant_hand_motion = 0.0
        previous_angle = self.previous_hand_angle[arm_name]

        if hand.valid and previous_angle is not None:
            instant_hand_motion = (
                wrapped_angle_difference(hand.angle, previous_angle) / 90.0
            )

        self.hand_motion_score[arm_name] = (
            self.motion_alpha * instant_hand_motion
            + (1.0 - self.motion_alpha) * self.hand_motion_score[arm_name]
        )

        if hand.valid:
            self.previous_hand_angle[arm_name] = hand.angle

    def combined_score(self, arm_name: str) -> float:
        arm_component = self.arm_motion_score[arm_name] / max(
            self.arm_motion_threshold,
            1e-6,
        )
        hand_component = self.hand_motion_score[arm_name] / max(
            self.hand_motion_threshold,
            1e-6,
        )
        return max(arm_component, hand_component)

    def update(
        self,
        left_arm: ArmFeatures,
        right_arm: ArmFeatures,
        left_hand: HandFeatures,
        right_hand: HandFeatures,
        now: float,
    ) -> Optional[str]:
        self._update_scores("LEFT", left_arm, left_hand)
        self._update_scores("RIGHT", right_arm, right_hand)

        if self.mode in {"left", "right"}:
            self.locked_arm = self.mode.upper()
            return self.locked_arm

        if self.locked_arm is not None:
            selected_arm = left_arm if self.locked_arm == "LEFT" else right_arm
            selected_hand = (
                left_hand if self.locked_arm == "LEFT" else right_hand
            )

            if (
                (selected_arm.valid and selected_arm.side_ok)
                or selected_hand.valid
            ):
                self.last_valid_at = now

            timed_out = (
                self.locked_at is not None
                and now - self.locked_at > self.lock_timeout
            )
            invalid_too_long = now - self.last_valid_at > self.invalid_timeout

            if timed_out or invalid_too_long:
                self.release()
            else:
                return self.locked_arm

        left_score = self.combined_score("LEFT")
        right_score = self.combined_score("RIGHT")

        selected: Optional[str] = None
        if (
            left_score >= 1.0
            and left_score >= self.dominance_ratio * max(right_score, 1e-5)
        ):
            selected = "LEFT"
        elif (
            right_score >= 1.0
            and right_score >= self.dominance_ratio * max(left_score, 1e-5)
        ):
            selected = "RIGHT"

        if selected is None:
            self.candidate = None
            self.candidate_frames = 0
            return None

        if selected == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = selected
            self.candidate_frames = 1

        if self.candidate_frames >= self.select_frames:
            self.locked_arm = selected
            self.locked_at = now
            self.last_valid_at = now
            self.candidate = None
            self.candidate_frames = 0
            return self.locked_arm

        return None


def select_primary_person(
    result,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    boxes = result.boxes
    keypoints = result.keypoints

    if (
        boxes is None
        or keypoints is None
        or keypoints.data is None
        or len(boxes) == 0
    ):
        return None

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    all_keypoints = keypoints.data.cpu().numpy()

    best_index = 0
    best_score = -1.0

    for index, (box, confidence) in enumerate(
        zip(boxes_xyxy, confidences)
    ):
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        score = area * (0.5 + float(confidence))
        if score > best_score:
            best_score = score
            best_index = index

    return (
        boxes_xyxy[best_index],
        all_keypoints[best_index],
        float(confidences[best_index]),
    )


def draw_hand(
    image: np.ndarray,
    hand: HandFeatures,
    color: Tuple[int, int, int],
) -> None:
    if not hand.valid or hand.landmarks_px is None:
        return

    for point in hand.landmarks_px:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 2, color, -1)

    if hand.wrist_xy is not None and hand.middle_mcp_xy is not None:
        p1 = tuple(hand.wrist_xy.astype(int))
        p2 = tuple(hand.middle_mcp_xy.astype(int))
        cv2.line(image, p1, p2, color, 2)


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-person COME recognizer V4.1: whole-arm beckoning "
            "+ MediaPipe wrist wave."
        )
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)

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

    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--hand-detection-conf", type=float, default=0.50)
    parser.add_argument("--hand-tracking-conf", type=float, default=0.50)
    parser.add_argument("--hand-match-ratio", type=float, default=0.65)

    parser.add_argument("--wrist-deadband-angle", type=float, default=10.0)
    parser.add_argument("--wrist-gesture-window", type=float, default=3.0)
    parser.add_argument("--wrist-cooldown", type=float, default=2.0)
    parser.add_argument("--wrist-min-stroke-angle", type=float, default=7.0)
    parser.add_argument("--wrist-min-strokes", type=int, default=2)
    parser.add_argument("--wrist-smoothing-alpha", type=float, default=0.60)
    parser.add_argument("--gesture-min-interval", type=float, default=0.18)
    parser.add_argument("--signal-max-gap", type=float, default=0.70)
    parser.add_argument("--status-hold", type=float, default=1.5)
    parser.add_argument(
        "--wrist-action",
        choices=("wave", "come"),
        default="wave",
        help=(
            "'wave': emit WRIST_WAVE separately; "
            "'come': map wrist waving to COME."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device is None:
        device: int | str = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Arm mode: {args.arm}")
    print(f"Wrist action mapping: {args.wrist_action.upper()}")
    print("Keys: Q/Esc quit, R reset.")

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

    arm_recognizers = {
        "LEFT": ContinuousOscillationRecognizer(
            gesture_window=args.arm_gesture_window,
            cooldown=args.arm_cooldown,
            min_stroke=args.arm_min_stroke_ratio,
            min_strokes=args.arm_min_strokes,
            smoothing_alpha=args.arm_smoothing_alpha,
            min_trigger_interval=args.gesture_min_interval,
            max_signal_gap=args.signal_max_gap,
            circular=False,
        ),
        "RIGHT": ContinuousOscillationRecognizer(
            gesture_window=args.arm_gesture_window,
            cooldown=args.arm_cooldown,
            min_stroke=args.arm_min_stroke_ratio,
            min_strokes=args.arm_min_strokes,
            smoothing_alpha=args.arm_smoothing_alpha,
            min_trigger_interval=args.gesture_min_interval,
            max_signal_gap=args.signal_max_gap,
            circular=False,
        ),
    }
    wrist_recognizers = {
        "LEFT": ContinuousOscillationRecognizer(
            gesture_window=args.wrist_gesture_window,
            cooldown=args.wrist_cooldown,
            min_stroke=args.wrist_min_stroke_angle,
            min_strokes=args.wrist_min_strokes,
            smoothing_alpha=args.wrist_smoothing_alpha,
            min_trigger_interval=args.gesture_min_interval,
            max_signal_gap=args.signal_max_gap,
            circular=True,
        ),
        "RIGHT": ContinuousOscillationRecognizer(
            gesture_window=args.wrist_gesture_window,
            cooldown=args.wrist_cooldown,
            min_stroke=args.wrist_min_stroke_angle,
            min_strokes=args.wrist_min_strokes,
            smoothing_alpha=args.wrist_smoothing_alpha,
            min_trigger_interval=args.gesture_min_interval,
            max_signal_gap=args.signal_max_gap,
            circular=True,
        ),
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

    status_until = 0.0
    detected_action = "NONE"
    detected_source = ""
    detected_arm = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.", file=sys.stderr)
                break

            now = time.monotonic()

            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            output = result.plot()
            selected_person = select_primary_person(result)
            hands_px = hand_detector.process(frame)

            if selected_person is not None:
                box, keypoints, box_confidence = selected_person
                x1, y1, x2, y2 = map(int, box)

                left_arm = extract_arm_features(
                    keypoints,
                    "left",
                    args.min_kp_conf,
                    args.far_ratio,
                    args.near_ratio,
                    args.far_angle,
                    args.near_angle,
                    args.cross_body_margin,
                )
                right_arm = extract_arm_features(
                    keypoints,
                    "right",
                    args.min_kp_conf,
                    args.far_ratio,
                    args.near_ratio,
                    args.far_angle,
                    args.near_angle,
                    args.cross_body_margin,
                )

                left_hand = match_hand_to_arm(
                    hands_px,
                    left_arm,
                    args.hand_match_ratio,
                    args.wrist_deadband_angle,
                )
                right_hand = match_hand_to_arm(
                    hands_px,
                    right_arm,
                    args.hand_match_ratio,
                    args.wrist_deadband_angle,
                )

                active_arm = selector.update(
                    left_arm,
                    right_arm,
                    left_hand,
                    right_hand,
                    now,
                )

                arm_triggered_by = {
                    "LEFT": False,
                    "RIGHT": False,
                }
                wrist_triggered_by = {
                    "LEFT": False,
                    "RIGHT": False,
                }

                arms_by_name = {
                    "LEFT": left_arm,
                    "RIGHT": right_arm,
                }
                hands_by_name = {
                    "LEFT": left_hand,
                    "RIGHT": right_hand,
                }

                if active_arm is None:
                    # AUTO pre-buffer:
                    # Preserve the beginning of each motion while the selector
                    # decides which arm is dominant. Do not emit both arms;
                    # this only avoids losing the first outward/inward stroke.
                    for arm_name in ("LEFT", "RIGHT"):
                        candidate_arm = arms_by_name[arm_name]
                        candidate_hand = hands_by_name[arm_name]

                        if candidate_arm.valid and candidate_arm.side_ok:
                            arm_triggered_by[arm_name] = (
                                arm_recognizers[arm_name].update(
                                    candidate_arm.reach_ratio,
                                    now,
                                )
                            )

                        if candidate_hand.valid:
                            wrist_triggered_by[arm_name] = (
                                wrist_recognizers[arm_name].update(
                                    candidate_hand.angle,
                                    now,
                                )
                            )

                    # A full gesture may finish before the motion selector has
                    # formally locked. In that rare case, accept only the side
                    # with the clearly larger current motion score.
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
                            key=selector.combined_score,
                        )
                        other_arm = (
                            "RIGHT"
                            if active_arm == "LEFT"
                            else "LEFT"
                        )
                        arm_recognizers[other_arm].reset()
                        wrist_recognizers[other_arm].reset()
                else:
                    inactive = "RIGHT" if active_arm == "LEFT" else "LEFT"

                    # Once locked, only the selected arm may continue.
                    arm_recognizers[inactive].reset()
                    wrist_recognizers[inactive].reset()

                    selected_arm = arms_by_name[active_arm]
                    selected_hand = hands_by_name[active_arm]

                    if selected_arm.valid and selected_arm.side_ok:
                        arm_triggered_by[active_arm] = (
                            arm_recognizers[active_arm].update(
                                selected_arm.reach_ratio,
                                now,
                            )
                        )

                    if selected_hand.valid:
                        wrist_triggered_by[active_arm] = (
                            wrist_recognizers[active_arm].update(
                                selected_hand.angle,
                                now,
                            )
                        )

                arm_triggered = (
                    active_arm is not None
                    and arm_triggered_by[active_arm]
                )
                wrist_triggered = (
                    active_arm is not None
                    and wrist_triggered_by[active_arm]
                )

                if arm_triggered and active_arm is not None:
                    action = "COME"
                    source = "ARM_BECKON"
                    detected_action = "COME"
                    detected_source = "ARM_BECKON"
                    detected_arm = active_arm
                    status_until = now + args.status_hold

                    print(
                        json.dumps(
                            {
                                "type": "command",
                                "action": action,
                                "source": source,
                                "arm": active_arm,
                                "timestamp": time.time(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    selector.release()

                elif wrist_triggered and active_arm is not None:
                    action = (
                        "COME"
                        if args.wrist_action == "come"
                        else "WRIST_WAVE"
                    )
                    source = "WRIST_WAVE"
                    detected_action = action
                    detected_source = "WRIST_WAVE"
                    detected_arm = active_arm
                    status_until = now + args.status_hold

                    print(
                        json.dumps(
                            {
                                "type": "command",
                                "action": action,
                                "source": source,
                                "arm": active_arm,
                                "timestamp": time.time(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    selector.release()

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    3,
                )

                draw_hand(output, left_hand, (255, 0, 255))
                draw_hand(output, right_hand, (0, 255, 255))

                active_text = active_arm or "NONE"
                left_marker = "*" if active_arm == "LEFT" else " "
                right_marker = "*" if active_arm == "RIGHT" else " "

                left_text = (
                    f"{left_marker}LEFT "
                    f"arm={left_arm.state:<7} "
                    f"reach={left_arm.reach_ratio:.2f} "
                    f"A={arm_recognizers['LEFT'].history_text()} "
                    f"| angle={left_hand.angle:+.0f} "
                    f"W={wrist_recognizers['LEFT'].history_text()} "
                    f"match={left_hand.match_distance:.2f}"
                )
                right_text = (
                    f"{right_marker}RIGHT "
                    f"arm={right_arm.state:<7} "
                    f"reach={right_arm.reach_ratio:.2f} "
                    f"A={arm_recognizers['RIGHT'].history_text()} "
                    f"| angle={right_hand.angle:+.0f} "
                    f"W={wrist_recognizers['RIGHT'].history_text()} "
                    f"match={right_hand.match_distance:.2f}"
                )

                cv2.rectangle(
                    output,
                    (0, 0),
                    (output.shape[1], 112),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    output,
                    (
                        f"Primary conf={box_confidence:.2f} "
                        f"| ACTIVE={active_text} "
                        f"| score L={selector.combined_score('LEFT'):.1f} "
                        f"R={selector.combined_score('RIGHT'):.1f} "
                        f"| wrist->{args.wrist_action.upper()}"
                    ),
                    (18, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.61,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    left_text,
                    (18, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    right_text,
                    (18, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    (
                        "A_hist: whole-arm FAR/NEAR | "
                        "A/W: two completed reversals required; one-way extension cannot trigger"
                    ),
                    (18, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(
                    output,
                    (0, 0),
                    (output.shape[1], 58),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    output,
                    "No person detected",
                    (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Permanent action-status panel.
            if now >= status_until:
                detected_action = "NONE"
                detected_source = ""
                detected_arm = ""

            panel_color = (
                (0, 150, 0)
                if detected_action != "NONE"
                else (40, 40, 40)
            )
            cv2.rectangle(
                output,
                (0, output.shape[0] - 92),
                (output.shape[1], output.shape[0]),
                panel_color,
                -1,
            )

            cv2.putText(
                output,
                f"DETECTED: {detected_action}",
                (24, output.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.18,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            detail = (
                f"source={detected_source} arm={detected_arm}"
                if detected_action != "NONE"
                else "Whole-arm beckon or wrist wave | Q quit | R reset"
            )
            cv2.putText(
                output,
                detail,
                (26, output.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Single-person COME Recognizer V4.1",
                output,
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                for recognizer in arm_recognizers.values():
                    recognizer.reset()
                for recognizer in wrist_recognizers.values():
                    recognizer.reset()
                selector.reset()
                status_until = 0.0
                detected_action = "NONE"
                detected_source = ""
                detected_arm = ""
                print("[RESET] all histories cleared.")

    finally:
        cap.release()
        hand_detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

"""Vendored GestureEngine from x2-vision-agent for ROS 2 integration."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from emo_robot_perception.multi_person_come_tracker_v1 import (
    GestureEvent,
    PersonGestureState,
    PersonObservation,
    assign_hands_uniquely,
    draw_person_overlay,
    event_source_is_eligible,
    extract_arm_features,
    make_person_state,
    person_motion_ownership_score,
    update_person_gesture,
)
from emo_robot_perception.single_person_come_recognizer_v1 import MediaPipeHandDetector


@dataclass
class GestureEngineConfig:
    """Public configuration for :class:`GestureEngine`.

    ``device="auto"`` selects CUDA device 0 when PyTorch can see CUDA and
    otherwise uses the CPU. This works on both Jetson ARM64 and desktop hosts.
    """

    model_path: str = str(Path(__file__).with_name("yolo26n-pose.pt"))
    tracker: str = "botsort.yaml"
    device: str = "auto"
    imgsz: int = 640
    person_confidence: float = 0.30
    iou: float = 0.60

    hand_backend: str = "auto"  # auto / mediapipe / none
    hand_surface_mode: str = "strict"  # strict / assist / off
    max_hands: int = 4
    hand_detection_confidence: float = 0.50
    hand_tracking_confidence: float = 0.50

    debug_overlay: bool = False
    track_state_timeout: float = 2.0
    action_cooldown: float = 1.8


@dataclass(frozen=True)
class TrackedPerson:
    """A person visible in the current frame."""

    track_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    center_x: float
    center_y: float


@dataclass(frozen=True)
class DetectedGesture:
    """A temporally confirmed gesture attributed to one tracked person."""

    track_id: int
    action: str
    source: str
    arm: str
    score: float
    person_confidence: float
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int]
    ownership_score: float
    wrist_travel_px: float
    wrist_travel_ratio: float
    hand_surface: str
    hand_surface_score: Optional[float]
    motion_source: str
    timestamp: float
    radial_fraction: Optional[float] = None
    bbox_height_ratio: float = 0.0
    pose_fallback: bool = False
    quality_band: str = "NORMAL"

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class GestureFrameTimings:
    """Wall-clock timing breakdown for one processed frame."""

    yolo_track_ms: float
    yolo_preprocess_ms: Optional[float]
    onnx_inference_ms: Optional[float]
    yolo_postprocess_ms: Optional[float]
    pose_extract_ms: float
    mediapipe_ms: float
    hand_assign_ms: float
    gesture_logic_ms: float
    overlay_ms: float
    result_pack_ms: float
    engine_total_ms: float


@dataclass
class GestureFrameResult:
    """Result returned for every processed frame."""

    people: list[TrackedPerson]
    events: list[DetectedGesture]
    annotated_frame: Optional[np.ndarray]
    inference_ms: float
    timings: GestureFrameTimings


def _algorithm_defaults(config: GestureEngineConfig) -> SimpleNamespace:
    """Build the argument namespace expected by the original recognizers."""
    values = {
        "tracker": config.tracker,
        "device": config.device,
        "imgsz": config.imgsz,
        "conf": config.person_confidence,
        "iou": config.iou,
        "debug_overlay": config.debug_overlay,
        "min_kp_conf": 0.40,
        "far_ratio": 1.30,
        "near_ratio": 0.72,
        "far_angle": 135.0,
        "near_angle": 90.0,
        "cross_body_margin": 0.22,
        "arm_gesture_window": 3.5,
        "arm_cooldown": 2.5,
        "arm_min_stroke_ratio": 0.16,
        "arm_min_strokes": 2,
        "arm_smoothing_alpha": 0.50,
        "disable_mid_come_adaptive": False,
        "mid_come_max_box_height": 0.50,
        "mid_come_min_stroke_ratio": 0.085,
        "mid_come_allow_cross_body": True,
        "mid_come_min_strokes": 2,
        "mid_come_min_trigger_interval": 0.10,
        "wave_window": 0.90,
        "wave_motion_hold": 0.18,
        "wave_static_hold": 0.0,
        "wave_cooldown": 0.80,
        "wave_min_lateral_range": 0.075,
        "wave_min_direction_changes": 2,
        "wave_direction_delta": 0.008,
        "wave_lateral_dominance": 1.25,
        "wave_down_reset": 0.20,
        "wave_history_gap_reset": 0.18,
        "wave_min_kp_conf": 0.25,
        "arm": "auto",
        "arm_motion_threshold": 0.018,
        "hand_motion_threshold": 0.050,
        "motion_dominance": 1.10,
        "arm_select_frames": 1,
        "arm_lock_timeout": 5.0,
        "arm_invalid_timeout": 0.8,
        "motion_alpha": 0.45,
        "hand_match_ratio": 0.65,
        "wrist_deadband_angle": 10.0,
        "wrist_gesture_window": 3.0,
        "wrist_cooldown": 2.0,
        "wrist_min_stroke_angle": 7.0,
        "wrist_min_strokes": 2,
        "wrist_smoothing_alpha": 0.60,
        "palm_ema_alpha": 0.30,
        "hand_surface_mode": config.hand_surface_mode,
        "hand_surface_window": 0.60,
        "hand_surface_min_samples": 3,
        "hand_surface_sample_threshold": 0.06,
        "hand_surface_threshold": 0.12,
        "hand_surface_consensus": 0.60,
        "invert_hand_surface": False,
        "disable_pose_fallback": False,
        "pose_fallback_max_box_height": 0.50,
        "pose_only_hand_surface_max_box_height": 0.42,
        "pose_fallback_min_wave_score": 1.25,
        "pose_fallback_min_come_score": 1.20,
        "pose_fallback_min_wave_candidate_score": 1.35,
        "degraded_pose_fallback_min_wave_score": 1.75,
        "degraded_pose_fallback_min_come_score": 1.35,
        "degraded_pose_fallback_wave_max_radial_fraction": 0.40,
        "degraded_pose_fallback_come_min_radial_fraction": 0.35,
        "gesture_min_interval": 0.18,
        "signal_max_gap": 0.70,
        "status_hold": 2.2,
        "action_cooldown": config.action_cooldown,
        "ownership_window": 1.20,
        "min_gesture_person_conf": 0.50,
        "min_gesture_hard_person_conf": 0.35,
        "min_gesture_box_height": 0.14,
        "min_gesture_hard_box_height": 0.06,
        "min_gesture_track_age": 0.45,
        "min_gesture_seen_frames": 5,
        "min_gesture_wrist_travel_px": 18.0,
        "min_gesture_wrist_travel_ratio": 0.18,
        "ownership_dominance_ratio": 0.60,
        "degraded_gesture_min_track_age": 0.90,
        "degraded_gesture_seen_frames": 10,
        "degraded_min_gesture_wrist_travel_px": 10.0,
        "degraded_min_gesture_wrist_travel_ratio": 0.30,
        "degraded_ownership_dominance_ratio": 0.85,
        "align_deadband": 0.08,
        "track_state_timeout": config.track_state_timeout,
        "target_release_timeout": 3.0,
        "print_target_json": False,
    }
    return SimpleNamespace(**values)


class GestureEngine:
    """Run multi-person WAVE/COME recognition on caller-provided BGR frames.

    The engine owns the YOLO tracker, MediaPipe detector and temporal person
    states. It does not open a camera, display a window, print JSON or depend
    on ROS. Call frames in chronological order from a single thread.
    """

    def __init__(
        self,
        config: Optional[GestureEngineConfig] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or GestureEngineConfig()
        self._log = log
        self._validate_config()
        self.device = self._resolve_device(self.config.device)
        self._args = _algorithm_defaults(self.config)
        self._args.device = self.device

        model_path = Path(self.config.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        self._emit(f"Loading pose model: {model_path}")
        self._emit(f"Inference device: {self.device}")
        self._model = YOLO(str(model_path))
        self._hand_detector: Optional[MediaPipeHandDetector] = None
        self._initialize_hand_backend()
        self._person_states: Dict[int, PersonGestureState] = {}
        self._closed = False

    def _validate_config(self) -> None:
        if self.config.hand_backend not in {"auto", "mediapipe", "none"}:
            raise ValueError("hand_backend must be auto, mediapipe or none")
        if self.config.hand_surface_mode not in {"strict", "assist", "off"}:
            raise ValueError(
                "hand_surface_mode must be strict, assist or off"
            )
        if self.config.imgsz <= 0:
            raise ValueError("imgsz must be greater than zero")

    @staticmethod
    def _resolve_device(requested: str) -> int | str:
        if requested.lower() == "auto":
            return 0 if torch.cuda.is_available() else "cpu"
        if requested.isdigit():
            return int(requested)
        return requested

    def _initialize_hand_backend(self) -> None:
        if self.config.hand_backend == "none":
            self._args.hand_surface_mode = "off"
            self._emit("MediaPipe hands disabled; pose-only gesture mode")
            return

        try:
            self._hand_detector = MediaPipeHandDetector(
                max_hands=self.config.max_hands,
                detection_confidence=(
                    self.config.hand_detection_confidence
                ),
                tracking_confidence=self.config.hand_tracking_confidence,
            )
            self._emit("MediaPipe hand backend enabled")
        except RuntimeError:
            if self.config.hand_backend == "mediapipe":
                raise
            self._args.hand_surface_mode = "off"
            self._emit(
                "MediaPipe is unavailable; falling back to pose-only mode"
            )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
        annotate: bool = False,
    ) -> GestureFrameResult:
        """Process one BGR frame and return tracked people and new events."""
        if self._closed:
            raise RuntimeError("GestureEngine is closed")
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("frame_bgr must be an HxWx3 NumPy BGR image")

        now = time.monotonic() if timestamp is None else float(timestamp)
        frame_height, frame_width = frame_bgr.shape[:2]
        started = time.perf_counter()

        results = self._model.track(
            source=frame_bgr,
            persist=True,
            tracker=self._args.tracker,
            conf=self._args.conf,
            iou=self._args.iou,
            imgsz=self._args.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        yolo_done = time.perf_counter()

        people: Dict[int, PersonObservation] = {}
        person_arms = {}

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
                    center_x=float(((x1 + x2) / 2.0) / frame_width),
                    center_y=float(((y1 + y2) / 2.0) / frame_height),
                    bbox_height_ratio=float(bbox_height / frame_height),
                    bbox_area_ratio=float(
                        (bbox_width * bbox_height)
                        / (frame_width * frame_height)
                    ),
                    keypoints=keypoints,
                )
                person_arms[track_id] = {
                    "LEFT": extract_arm_features(
                        keypoints,
                        "left",
                        self._args.min_kp_conf,
                        self._args.far_ratio,
                        self._args.near_ratio,
                        self._args.far_angle,
                        self._args.near_angle,
                        self._args.cross_body_margin,
                    ),
                    "RIGHT": extract_arm_features(
                        keypoints,
                        "right",
                        self._args.min_kp_conf,
                        self._args.far_ratio,
                        self._args.near_ratio,
                        self._args.far_angle,
                        self._args.near_angle,
                        self._args.cross_body_margin,
                    ),
                }
        pose_extract_done = time.perf_counter()

        hands_px = (
            self._hand_detector.process(frame_bgr)
            if self._hand_detector is not None
            else []
        )
        mediapipe_done = time.perf_counter()
        hand_assignments = assign_hands_uniquely(
            hands_px=hands_px,
            person_arms=person_arms,
            max_match_ratio=self._args.hand_match_ratio,
            wrist_deadband_angle=self._args.wrist_deadband_angle,
            invert_hand_surface=self._args.invert_hand_surface,
        )
        hand_assign_done = time.perf_counter()

        candidate_events: list[GestureEvent] = []
        for track_id, person in people.items():
            state = self._person_states.get(track_id)
            if state is None:
                state = make_person_state(track_id, self._args, now)
                self._person_states[track_id] = state

            state.last_seen = now
            state.seen_frames += 1
            event = update_person_gesture(
                state=state,
                person=person,
                keypoints=person.keypoints,
                arms=person_arms[track_id],
                hands=hand_assignments[track_id],
                args=self._args,
                now=now,
            )
            if event is not None:
                candidate_events.append(event)

        ownership_by_person = {}
        for track_id, state in self._person_states.items():
            if track_id not in people:
                continue
            ownership_by_person[track_id] = (
                person_motion_ownership_score(
                    state=state,
                    now=now,
                    args=self._args,
                )[0]
            )
        global_max_ownership = max(
            ownership_by_person.values(),
            default=0.0,
        )

        accepted_events: list[GestureEvent] = []
        for event in candidate_events:
            person = people.get(event.track_id)
            state = self._person_states.get(event.track_id)
            if person is None or state is None:
                continue

            eligible, reason = event_source_is_eligible(
                event=event,
                person=person,
                state=state,
                global_max_ownership=global_max_ownership,
                now=now,
                args=self._args,
            )
            if eligible:
                accepted_events.append(event)
                state.begin_action_cooldown(
                    action=event.action,
                    now=now,
                    cooldown_seconds=self._args.action_cooldown,
                )
            else:
                self._emit(
                    f"Suppressed {event.action} from ID {event.track_id}: "
                    f"{reason}"
                )
                state.arm_recognizers[event.arm].reset()
                state.wrist_recognizers[event.arm].reset()
                state.wave_recognizer.reset_arm(event.arm)
                state.last_event_until = 0.0
                state.last_event_action = "NONE"
                state.last_event_source = ""
                state.last_event_arm = ""

        stale_ids = [
            track_id
            for track_id, state in self._person_states.items()
            if now - state.last_seen >= self._args.track_state_timeout
        ]
        for track_id in stale_ids:
            del self._person_states[track_id]
        gesture_logic_done = time.perf_counter()

        annotated_frame = None
        if annotate:
            if self.config.debug_overlay:
                try:
                    annotated_frame = result.plot(
                        boxes=False,
                        labels=False,
                    )
                except TypeError:
                    annotated_frame = result.plot()
            else:
                annotated_frame = frame_bgr.copy()

            for track_id, person in people.items():
                draw_person_overlay(
                    output=annotated_frame,
                    person=person,
                    state=self._person_states[track_id],
                    arms=person_arms[track_id],
                    hands=hand_assignments[track_id],
                    target_id=None,
                    now=now,
                    deadband=self._args.align_deadband,
                    debug_overlay=self.config.debug_overlay,
                    args=self._args,
                )
        overlay_done = time.perf_counter()

        public_people = [
            TrackedPerson(
                track_id=person.track_id,
                confidence=person.confidence,
                bbox=person.bbox,
                center_x=person.center_x,
                center_y=person.center_y,
            )
            for person in people.values()
        ]
        public_events = [
            self._public_event(event, people[event.track_id])
            for event in accepted_events
        ]
        result_pack_done = time.perf_counter()
        engine_total_ms = (result_pack_done - started) * 1000.0
        timings = GestureFrameTimings(
            yolo_track_ms=(yolo_done - started) * 1000.0,
            yolo_preprocess_ms=self._result_speed_ms(
                result,
                "preprocess",
            ),
            onnx_inference_ms=self._result_speed_ms(
                result,
                "inference",
            ),
            yolo_postprocess_ms=self._result_speed_ms(
                result,
                "postprocess",
            ),
            pose_extract_ms=(pose_extract_done - yolo_done) * 1000.0,
            mediapipe_ms=(
                mediapipe_done - pose_extract_done
            ) * 1000.0,
            hand_assign_ms=(
                hand_assign_done - mediapipe_done
            ) * 1000.0,
            gesture_logic_ms=(
                gesture_logic_done - hand_assign_done
            ) * 1000.0,
            overlay_ms=(overlay_done - gesture_logic_done) * 1000.0,
            result_pack_ms=(result_pack_done - overlay_done) * 1000.0,
            engine_total_ms=engine_total_ms,
        )
        return GestureFrameResult(
            people=public_people,
            events=public_events,
            annotated_frame=annotated_frame,
            inference_ms=engine_total_ms,
            timings=timings,
        )

    @staticmethod
    def _result_speed_ms(result, key: str) -> Optional[float]:
        """Read an optional Ultralytics per-stage speed value."""
        speed = getattr(result, "speed", None)
        if not isinstance(speed, dict):
            return None
        try:
            value = float(speed[key])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(value) or value < 0.0:
            return None
        return value

    @staticmethod
    def _public_event(
        event: GestureEvent,
        person: PersonObservation,
    ) -> DetectedGesture:
        return DetectedGesture(
            track_id=event.track_id,
            action=event.action,
            source=event.source,
            arm=event.arm,
            score=float(event.score),
            person_confidence=person.confidence,
            center_x=person.center_x,
            center_y=person.center_y,
            bbox=person.bbox,
            ownership_score=event.ownership_score,
            wrist_travel_px=event.wrist_travel_px,
            wrist_travel_ratio=event.wrist_travel_ratio,
            hand_surface=event.hand_surface,
            hand_surface_score=event.hand_surface_score,
            motion_source=event.motion_source,
            radial_fraction=getattr(event, "radial_fraction", None),
            bbox_height_ratio=getattr(
                event,
                "bbox_height_ratio",
                person.bbox_height_ratio,
            ),
            pose_fallback=getattr(event, "pose_fallback", False),
            quality_band=getattr(event, "quality_band", "NORMAL"),
            timestamp=event.timestamp,
        )

    def reset(self) -> None:
        """Forget all temporal tracks and gesture histories."""
        self._person_states.clear()
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers is not None:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()

    def close(self) -> None:
        """Release optional native resources owned by the engine."""
        if self._closed:
            return
        if self._hand_detector is not None:
            self._hand_detector.close()
        self._closed = True

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    def __enter__(self) -> "GestureEngine":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

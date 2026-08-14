#!/usr/bin/env python3

"""ROS 2 wrapper for the vendored multi-person WAVE/COME GestureEngine."""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from emo_robot_interfaces.msg import GestureDetection
from emo_robot_perception.gesture_engine import (
    DetectedGesture,
    GestureEngine,
    GestureEngineConfig,
)


class GestureDetectorNode(Node):
    """Detect multi-person WAVE and COME events from ROS camera images."""

    _MODEL_FILENAMES = {
        "pt": "yolo26n-pose.pt",
        "onnx": "yolo26n-pose-fp16.onnx",
    }

    def __init__(self) -> None:
        super().__init__("gesture_detector")

        self.declare_parameter("image_topic", "/emo_robot/camera/image_raw")
        self.declare_parameter("model_backend", "pt")
        self.declare_parameter("model_path", "")
        self.declare_parameter("tracker", "botsort.yaml")
        self.declare_parameter("yolo_conf", 0.30)
        self.declare_parameter("yolo_iou", 0.60)
        self.declare_parameter("yolo_imgsz", 640)
        self.declare_parameter("yolo_device", "auto")
        self.declare_parameter("hand_backend", "mediapipe")
        self.declare_parameter("hand_surface_mode", "strict")
        self.declare_parameter("track_state_timeout_s", 2.0)
        self.declare_parameter("action_cooldown_s", 1.8)
        self.declare_parameter("debug_overlay", True)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("timing_log_interval_frames", 30)

        image_topic = str(self.get_parameter("image_topic").value)
        model_path = self._resolve_model_path(
            str(self.get_parameter("model_path").value),
            str(self.get_parameter("model_backend").value),
        )
        yolo_device = str(self.get_parameter("yolo_device").value)
        backend_name, execution_provider = self._validate_model_backend(
            model_path,
            yolo_device,
        )
        self._publish_debug = bool(
            self.get_parameter("publish_debug_image").value
        )
        requested_timing_interval = int(
            self.get_parameter("timing_log_interval_frames").value
        )
        self._timing_log_interval = max(1, requested_timing_interval)
        if requested_timing_interval < 1:
            self.get_logger().warning(
                "timing_log_interval_frames must be >= 1; using 1"
            )

        engine_config = GestureEngineConfig(
            model_path=model_path,
            tracker=str(self.get_parameter("tracker").value),
            device=yolo_device,
            imgsz=int(self.get_parameter("yolo_imgsz").value),
            person_confidence=float(
                self.get_parameter("yolo_conf").value
            ),
            iou=float(self.get_parameter("yolo_iou").value),
            hand_backend=str(
                self.get_parameter("hand_backend").value
            ),
            hand_surface_mode=str(
                self.get_parameter("hand_surface_mode").value
            ),
            debug_overlay=bool(
                self.get_parameter("debug_overlay").value
            ),
            track_state_timeout=float(
                self.get_parameter("track_state_timeout_s").value
            ),
            action_cooldown=float(
                self.get_parameter("action_cooldown_s").value
            ),
        )
        self._engine = GestureEngine(
            engine_config,
            log=lambda message: self.get_logger().info(
                f"GestureEngine: {message}"
            ),
        )

        self._bridge = CvBridge()
        self._gesture_pub = self.create_publisher(
            GestureDetection,
            "/emo_robot/perception/gesture_detection",
            10,
        )
        self._debug_pub = (
            self.create_publisher(
                Image,
                "/emo_robot/perception/debug_image",
                10,
            )
            if self._publish_debug
            else None
        )
        self.create_subscription(
            Image,
            image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )

        self._frame_count = 0
        self._timing_samples: list[dict[str, Optional[float]]] = []
        self._perf_window_started_at: Optional[float] = None
        self.get_logger().info(
            f"Gesture detector ready on {image_topic}; "
            f"model={model_path}, backend={backend_name}, "
            f"provider={execution_provider}, "
            f"hand_backend={engine_config.hand_backend}, "
            f"hand_surface_mode={engine_config.hand_surface_mode}"
        )

    @staticmethod
    def _resolve_model_path(raw: str, backend: str = "pt") -> str:
        """Resolve an explicit model or a packaged backend-specific model."""
        if raw.strip():
            configured = Path(raw).expanduser().resolve()
            if configured.is_file():
                return str(configured)
            raise FileNotFoundError(
                f"Configured YOLO model does not exist: {configured}"
            )

        normalized_backend = backend.strip().lower()
        try:
            model_filename = GestureDetectorNode._MODEL_FILENAMES[
                normalized_backend
            ]
        except KeyError as exc:
            supported = ", ".join(
                sorted(GestureDetectorNode._MODEL_FILENAMES)
            )
            raise ValueError(
                f"Unsupported model_backend {backend!r}; "
                f"expected one of: {supported}"
            ) from exc

        candidates = []
        try:
            package_share = Path(
                get_package_share_directory("emo_robot_perception")
            )
            candidates.append(package_share / "models" / model_filename)
        except Exception:
            pass

        candidates.append(
            Path(__file__).resolve().parents[1]
            / "models"
            / model_filename
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"YOLO pose model for backend {normalized_backend!r} "
            f"is not installed. Expected {model_filename}; "
            f"checked: {checked}"
        )

    @staticmethod
    def _validate_model_backend(
        model_path: str,
        requested_device: str,
    ) -> tuple[str, str]:
        """Validate the runtime selected by the model suffix.

        Ultralytics chooses its backend from the model extension. For ONNX
        on a requested GPU, fail early instead of allowing a silent CPU
        fallback when the Jetson-specific runtime is missing.
        """
        suffix = Path(model_path).suffix.lower()
        if suffix == ".pt":
            return "PyTorch", requested_device
        if suffix != ".onnx":
            return suffix.lstrip(".").upper() or "unknown", requested_device

        try:
            ort = import_module("onnxruntime")
        except ImportError as exc:
            raise RuntimeError(
                "An ONNX model was selected, but onnxruntime is not "
                "installed. Run scripts/install_onnxruntime_jetson.sh "
                "inside the emobot environment."
            ) from exc

        providers = list(ort.get_available_providers())
        normalized_device = requested_device.strip().lower()
        gpu_requested = normalized_device != "cpu"
        if normalized_device == "auto":
            try:
                torch = import_module("torch")
                gpu_requested = bool(torch.cuda.is_available())
            except ImportError:
                gpu_requested = False

        if gpu_requested and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "ONNX GPU inference was requested, but "
                "CUDAExecutionProvider is unavailable. Available "
                f"providers: {providers}. Refusing CPU fallback."
            )

        provider = (
            "CUDAExecutionProvider"
            if gpu_requested
            else "CPUExecutionProvider"
        )
        return f"ONNX Runtime {ort.__version__}", provider

    def _on_image(self, msg: Image) -> None:
        callback_started = time.perf_counter()
        self._frame_count += 1
        try:
            frame_bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return
        bridge_done = time.perf_counter()

        try:
            result = self._engine.process_frame(
                frame_bgr,
                timestamp=time.monotonic(),
                annotate=self._debug_pub is not None,
            )
        except Exception as exc:
            self.get_logger().error(f"GestureEngine frame failed: {exc}")
            self._publish_debug_frame(frame_bgr, msg)
            return

        event_publish_started = time.perf_counter()
        for event in result.events:
            self._publish_detection(event, msg)
        event_publish_done = time.perf_counter()

        debug_frame = (
            result.annotated_frame
            if result.annotated_frame is not None
            else frame_bgr
        )
        debug_publish_started = time.perf_counter()
        self._publish_debug_frame(debug_frame, msg)
        callback_done = time.perf_counter()

        self._record_timing_sample(
            result=result,
            callback_started=callback_started,
            callback_done=callback_done,
            cv_bridge_ms=(bridge_done - callback_started) * 1000.0,
            event_publish_ms=(
                event_publish_done - event_publish_started
            ) * 1000.0,
            debug_publish_ms=(
                callback_done - debug_publish_started
            ) * 1000.0,
        )

    def _record_timing_sample(
        self,
        result,
        callback_started: float,
        callback_done: float,
        cv_bridge_ms: float,
        event_publish_ms: float,
        debug_publish_ms: float,
    ) -> None:
        timings = result.timings
        sample: dict[str, Optional[float]] = {
            "callback_total_ms": (
                callback_done - callback_started
            ) * 1000.0,
            "cv_bridge_ms": cv_bridge_ms,
            "event_publish_ms": event_publish_ms,
            "debug_publish_ms": debug_publish_ms,
            "engine_total_ms": timings.engine_total_ms,
            "yolo_track_ms": timings.yolo_track_ms,
            "yolo_preprocess_ms": timings.yolo_preprocess_ms,
            "onnx_inference_ms": timings.onnx_inference_ms,
            "yolo_postprocess_ms": timings.yolo_postprocess_ms,
            "pose_extract_ms": timings.pose_extract_ms,
            "mediapipe_ms": timings.mediapipe_ms,
            "hand_assign_ms": timings.hand_assign_ms,
            "gesture_logic_ms": timings.gesture_logic_ms,
            "overlay_ms": timings.overlay_ms,
            "result_pack_ms": timings.result_pack_ms,
        }
        if self._perf_window_started_at is None:
            self._perf_window_started_at = callback_started
        self._timing_samples.append(sample)
        if len(self._timing_samples) < self._timing_log_interval:
            return

        elapsed = callback_done - self._perf_window_started_at
        processing_fps = (
            len(self._timing_samples) / elapsed
            if elapsed > 0.0
            else 0.0
        )
        summary = self._summarize_timing_samples(self._timing_samples)
        self._log_timing_summary(
            summary=summary,
            sample_count=len(self._timing_samples),
            processing_fps=processing_fps,
            result=result,
        )
        self._timing_samples.clear()
        self._perf_window_started_at = callback_done

    @staticmethod
    def _summarize_timing_samples(
        samples: list[dict[str, Optional[float]]],
    ) -> dict[str, tuple[float, float]]:
        """Return mean and P95 for every available timing metric."""
        summary: dict[str, tuple[float, float]] = {}
        keys = {key for sample in samples for key in sample}
        for key in keys:
            values = [
                float(sample[key])
                for sample in samples
                if sample.get(key) is not None
            ]
            if not values:
                continue
            summary[key] = (
                float(np.mean(values)),
                float(np.percentile(values, 95)),
            )
        return summary

    @staticmethod
    def _format_mean(
        summary: dict[str, tuple[float, float]],
        key: str,
    ) -> str:
        metric = summary.get(key)
        return f"{metric[0]:.1f}" if metric is not None else "n/a"

    def _log_timing_summary(
        self,
        summary: dict[str, tuple[float, float]],
        sample_count: int,
        processing_fps: float,
        result,
    ) -> None:
        callback_avg, callback_p95 = summary["callback_total_ms"]
        engine_avg, engine_p95 = summary["engine_total_ms"]

        def mean(key: str) -> str:
            return self._format_mean(summary, key)

        self.get_logger().info(
            f"Perf[{sample_count}]: processing={processing_fps:.1f} FPS "
            f"callback={callback_avg:.1f}/{callback_p95:.1f} ms "
            f"engine={engine_avg:.1f}/{engine_p95:.1f} ms "
            f"(avg/p95) people={len(result.people)} "
            f"events={len(result.events)} frame=#{self._frame_count}"
        )
        self.get_logger().info(
            "Engine avg ms: "
            f"yolo_track={mean('yolo_track_ms')} "
            f"[pre={mean('yolo_preprocess_ms')} "
            f"model={mean('onnx_inference_ms')} "
            f"post={mean('yolo_postprocess_ms')}] "
            f"pose={mean('pose_extract_ms')} "
            f"mediapipe={mean('mediapipe_ms')} "
            f"hand_assign={mean('hand_assign_ms')} "
            f"gesture={mean('gesture_logic_ms')} "
            f"overlay={mean('overlay_ms')} "
            f"pack={mean('result_pack_ms')}"
        )
        self.get_logger().info(
            "ROS avg ms: "
            f"cv_bridge={mean('cv_bridge_ms')} "
            f"event_publish={mean('event_publish_ms')} "
            f"debug_publish={mean('debug_publish_ms')}"
        )

    @staticmethod
    def _gesture_type_for_action(action: str) -> Optional[int]:
        if action == "WAVE":
            return GestureDetection.GESTURE_WAVE
        if action == "COME":
            return GestureDetection.GESTURE_COME
        return None

    def _publish_detection(
        self,
        event: DetectedGesture,
        source_msg: Image,
    ) -> bool:
        gesture_type = self._gesture_type_for_action(event.action)
        if gesture_type is None:
            self.get_logger().warning(
                f"Ignoring unsupported gesture action: {event.action}"
            )
            return False

        detection = GestureDetection()
        detection.header.stamp = source_msg.header.stamp
        detection.header.frame_id = source_msg.header.frame_id
        detection.gesture = gesture_type
        detection.track_id = int(event.track_id)
        detection.confidence = float(event.score)
        detection.center_x = float(event.center_x)
        detection.center_y = float(event.center_y)
        detection.valid = True
        self._gesture_pub.publish(detection)
        self.get_logger().info(
            f"Gesture detected: action={event.action} "
            f"track_id={event.track_id} arm={event.arm} "
            f"source={event.source} score={event.score:.2f}"
        )
        return True

    def _publish_debug_frame(
        self,
        frame: np.ndarray,
        source_msg: Image,
    ) -> None:
        if self._debug_pub is None:
            return

        debug_msg = self._bridge.cv2_to_imgmsg(frame, "bgr8")
        debug_msg.header.stamp = source_msg.header.stamp
        debug_msg.header.frame_id = source_msg.header.frame_id
        self._debug_pub.publish(debug_msg)

    def destroy_node(self) -> None:
        self._engine.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GestureDetectorNode()
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

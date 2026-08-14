"""Unit tests for ROS gesture event mapping and publication."""

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def _install_runtime_stubs() -> None:
    if "sensor_msgs.msg" not in sys.modules:
        sensor_msgs = ModuleType("sensor_msgs")
        sensor_msgs_msg = ModuleType("sensor_msgs.msg")

        class _Image:
            def __init__(self):
                self.header = SimpleNamespace(stamp=None, frame_id="")

        sensor_msgs_msg.Image = _Image
        sensor_msgs.msg = sensor_msgs_msg
        sys.modules["sensor_msgs"] = sensor_msgs
        sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    if "emo_robot_interfaces.msg" not in sys.modules:
        interfaces = ModuleType("emo_robot_interfaces")
        interfaces_msg = ModuleType("emo_robot_interfaces.msg")

        class _GestureDetection:
            GESTURE_NONE = 0
            GESTURE_WAVE = 1
            GESTURE_COME = 2

            def __init__(self):
                self.header = SimpleNamespace(stamp=None, frame_id="")
                self.track_id = 0
                self.gesture = self.GESTURE_NONE
                self.confidence = 0.0
                self.center_x = 0.0
                self.center_y = 0.0
                self.valid = False

        interfaces_msg.GestureDetection = _GestureDetection
        interfaces.msg = interfaces_msg
        sys.modules["emo_robot_interfaces"] = interfaces
        sys.modules["emo_robot_interfaces.msg"] = interfaces_msg

    if "rclpy" not in sys.modules:
        rclpy = ModuleType("rclpy")
        rclpy.init = Mock()
        rclpy.spin = Mock()
        rclpy.ok = Mock(return_value=False)
        rclpy.shutdown = Mock()

        rclpy_node = ModuleType("rclpy.node")
        rclpy_node.Node = type("Node", (), {})

        rclpy_qos = ModuleType("rclpy.qos")
        rclpy_qos.qos_profile_sensor_data = object()

        rclpy.node = rclpy_node
        rclpy.qos = rclpy_qos
        rclpy_qos.QoSHistoryPolicy = SimpleNamespace(KEEP_LAST=1)
        rclpy_qos.QoSProfile = type("QoSProfile", (), {})
        rclpy_qos.QoSReliabilityPolicy = SimpleNamespace(BEST_EFFORT=1)
        sys.modules["rclpy"] = rclpy
        sys.modules["rclpy.node"] = rclpy_node
        sys.modules["rclpy.qos"] = rclpy_qos

    if "ament_index_python.packages" not in sys.modules:
        ament_index_python = ModuleType("ament_index_python")
        ament_index_packages = ModuleType("ament_index_python.packages")
        ament_index_packages.get_package_share_directory = (
            lambda _package_name: ""
        )
        ament_index_packages.get_package_prefix = (
            lambda _package_name: ""
        )
        ament_index_python.packages = ament_index_packages
        sys.modules["ament_index_python"] = ament_index_python
        sys.modules["ament_index_python.packages"] = ament_index_packages

    if "cv_bridge" not in sys.modules:
        cv_bridge = ModuleType("cv_bridge")
        cv_bridge.CvBridge = type("CvBridge", (), {})
        sys.modules["cv_bridge"] = cv_bridge

    if "torch" not in sys.modules:
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch

    if "ultralytics" not in sys.modules:
        ultralytics = ModuleType("ultralytics")
        ultralytics.YOLO = type("YOLO", (), {})
        sys.modules["ultralytics"] = ultralytics


_install_runtime_stubs()

from sensor_msgs.msg import Image

from emo_robot_interfaces.msg import GestureDetection
from emo_robot_perception import gesture_detector_node as detector_module
from emo_robot_perception import gesture_engine as gesture_engine_module
from emo_robot_perception.gesture_detector_node import GestureDetectorNode
from emo_robot_perception.gesture_engine import (
    GestureEngine,
    GestureEngineConfig,
    GestureFrameTimings,
    _algorithm_defaults,
)


def _event(action: str, track_id: int = 7):
    return SimpleNamespace(
        action=action,
        track_id=track_id,
        score=1.25,
        center_x=0.35,
        center_y=0.45,
        arm="RIGHT",
        source="TEST",
    )


def test_packaged_backend_selects_onnx_and_pt(monkeypatch, tmp_path):
    package_share = tmp_path / "share"
    models = package_share / "models"
    models.mkdir(parents=True)
    onnx_model = models / "yolo26n-pose-fp16.onnx"
    pt_model = models / "yolo26n-pose.pt"
    onnx_model.touch()
    pt_model.touch()
    monkeypatch.setattr(
        detector_module,
        "get_package_share_directory",
        lambda _package_name: str(package_share),
    )

    assert GestureDetectorNode._resolve_model_path("", "onnx") == str(
        onnx_model
    )
    assert GestureDetectorNode._resolve_model_path("", "pt") == str(
        pt_model
    )


def test_explicit_model_path_overrides_backend(tmp_path):
    custom_model = tmp_path / "custom.onnx"
    custom_model.touch()

    resolved = GestureDetectorNode._resolve_model_path(
        str(custom_model),
        "invalid-but-ignored",
    )

    assert resolved == str(custom_model)


def test_invalid_model_backend_is_rejected():
    try:
        GestureDetectorNode._resolve_model_path("", "tensorrt")
    except ValueError as exc:
        assert "Unsupported model_backend" in str(exc)
        assert "onnx, pt" in str(exc)
    else:
        raise AssertionError("invalid model backend was not rejected")


def test_missing_packaged_backend_model_is_rejected(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        detector_module,
        "get_package_share_directory",
        lambda _package_name: str(tmp_path),
    )
    monkeypatch.setattr(
        detector_module,
        "__file__",
        str(tmp_path / "package" / "gesture_detector_node.py"),
    )

    try:
        GestureDetectorNode._resolve_model_path("", "onnx")
    except FileNotFoundError as exc:
        assert "backend 'onnx'" in str(exc)
        assert "yolo26n-pose-fp16.onnx" in str(exc)
    else:
        raise AssertionError("missing packaged model was not rejected")


def test_pt_backend_does_not_require_onnxruntime():
    backend, provider = GestureDetectorNode._validate_model_backend(
        "/tmp/yolo26n-pose.pt",
        "0",
    )

    assert backend == "PyTorch"
    assert provider == "0"


def test_onnx_backend_selects_cuda_provider(monkeypatch):
    ort = SimpleNamespace(
        __version__="1.20.2",
        get_available_providers=lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    modules = {"onnxruntime": ort, "torch": torch}
    monkeypatch.setattr(
        detector_module,
        "import_module",
        lambda name: modules[name],
    )

    backend, provider = GestureDetectorNode._validate_model_backend(
        "/tmp/yolo26n-pose-fp16.onnx",
        "auto",
    )

    assert backend == "ONNX Runtime 1.20.2"
    assert provider == "CUDAExecutionProvider"


def test_onnx_backend_refuses_gpu_to_cpu_fallback(monkeypatch):
    ort = SimpleNamespace(
        __version__="1.20.2",
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        detector_module,
        "import_module",
        lambda _name: ort,
    )

    try:
        GestureDetectorNode._validate_model_backend(
            "/tmp/yolo26n-pose-fp16.onnx",
            "0",
        )
    except RuntimeError as exc:
        assert "Refusing CPU fallback" in str(exc)
    else:
        raise AssertionError("missing CUDA provider was not rejected")


def test_action_mapping_supports_wave_and_come():
    assert (
        GestureDetectorNode._gesture_type_for_action("WAVE")
        == GestureDetection.GESTURE_WAVE
    )
    assert (
        GestureDetectorNode._gesture_type_for_action("COME")
        == GestureDetection.GESTURE_COME
    )
    assert GestureDetectorNode._gesture_type_for_action("OTHER") is None


def test_publish_detection_maps_engine_fields():
    node = SimpleNamespace(
        _gesture_pub=Mock(),
        get_logger=lambda: Mock(),
        _gesture_type_for_action=(
            GestureDetectorNode._gesture_type_for_action
        ),
    )
    source = Image()
    source.header.frame_id = "usb_camera_optical_frame"

    published = GestureDetectorNode._publish_detection(
        node,
        _event("COME"),
        source,
    )

    assert published is True
    message = node._gesture_pub.publish.call_args.args[0]
    assert message.gesture == GestureDetection.GESTURE_COME
    assert message.track_id == 7
    assert message.confidence == 1.25
    assert message.center_x == 0.35
    assert message.center_y == 0.45
    assert message.valid is True
    assert message.header.frame_id == "usb_camera_optical_frame"


def test_unknown_action_is_not_published():
    logger = Mock()
    node = SimpleNamespace(
        _gesture_pub=Mock(),
        get_logger=lambda: logger,
        _gesture_type_for_action=(
            GestureDetectorNode._gesture_type_for_action
        ),
    )

    published = GestureDetectorNode._publish_detection(
        node,
        _event("UNKNOWN"),
        Image(),
    )

    assert published is False
    node._gesture_pub.publish.assert_not_called()
    logger.warning.assert_called_once()


def test_one_frame_publishes_every_engine_event():
    events = [_event("WAVE", 3), _event("COME", 8)]
    result = SimpleNamespace(
        events=events,
        people=[object(), object()],
        inference_ms=25.0,
        annotated_frame=None,
    )
    node = SimpleNamespace(
        _frame_count=0,
        _perf_window_frames=0,
        _perf_window_started_at=None,
        _bridge=SimpleNamespace(
            imgmsg_to_cv2=lambda _msg, _encoding: np.zeros(
                (8, 8, 3), dtype=np.uint8
            )
        ),
        _engine=SimpleNamespace(
            process_frame=lambda *_args, **_kwargs: result
        ),
        _debug_pub=None,
        _publish_detection=Mock(),
        _publish_debug_frame=Mock(),
        _record_timing_sample=Mock(),
        get_logger=lambda: Mock(),
    )
    source = Image()

    GestureDetectorNode._on_image(node, source)

    assert node._publish_detection.call_count == 2
    assert node._publish_detection.call_args_list[0].args == (
        events[0],
        source,
    )
    assert node._publish_detection.call_args_list[1].args == (
        events[1],
        source,
    )
    node._publish_debug_frame.assert_called_once()
    node._record_timing_sample.assert_called_once()


def _timings(**overrides):
    values = {
        "yolo_track_ms": 40.0,
        "yolo_preprocess_ms": 2.0,
        "onnx_inference_ms": 30.0,
        "yolo_postprocess_ms": 3.0,
        "pose_extract_ms": 4.0,
        "mediapipe_ms": 35.0,
        "hand_assign_ms": 1.0,
        "gesture_logic_ms": 2.0,
        "overlay_ms": 5.0,
        "result_pack_ms": 1.0,
        "engine_total_ms": 88.0,
    }
    values.update(overrides)
    return GestureFrameTimings(**values)


def test_timing_summary_uses_available_values_and_p95():
    samples = [
        {
            "callback_total_ms": 100.0,
            "engine_total_ms": 80.0,
            "onnx_inference_ms": None,
        },
        {
            "callback_total_ms": 200.0,
            "engine_total_ms": 160.0,
            "onnx_inference_ms": None,
        },
    ]

    summary = GestureDetectorNode._summarize_timing_samples(samples)

    assert summary["callback_total_ms"] == (150.0, 195.0)
    assert summary["engine_total_ms"] == (120.0, 156.0)
    assert "onnx_inference_ms" not in summary
    assert GestureDetectorNode._format_mean(
        summary,
        "onnx_inference_ms",
    ) == "n/a"


def test_timing_window_logs_and_resets_at_interval():
    log_summary = Mock()
    node = SimpleNamespace(
        _timing_samples=[],
        _timing_log_interval=2,
        _perf_window_started_at=None,
        _frame_count=2,
        _summarize_timing_samples=(
            GestureDetectorNode._summarize_timing_samples
        ),
        _log_timing_summary=log_summary,
    )
    result = SimpleNamespace(
        timings=_timings(),
        people=[object()],
        events=[],
    )

    GestureDetectorNode._record_timing_sample(
        node,
        result,
        callback_started=0.0,
        callback_done=0.1,
        cv_bridge_ms=1.0,
        event_publish_ms=0.0,
        debug_publish_ms=2.0,
    )
    assert len(node._timing_samples) == 1
    log_summary.assert_not_called()

    GestureDetectorNode._record_timing_sample(
        node,
        result,
        callback_started=0.2,
        callback_done=0.3,
        cv_bridge_ms=1.5,
        event_publish_ms=0.0,
        debug_publish_ms=2.5,
    )

    log_summary.assert_called_once()
    assert node._timing_samples == []
    assert node._perf_window_started_at == 0.3
    assert log_summary.call_args.kwargs["sample_count"] == 2
    assert log_summary.call_args.kwargs["processing_fps"] == 2 / 0.3


def test_timing_summary_logs_three_lines_and_missing_speed_as_na():
    logger = Mock()
    node = SimpleNamespace(
        _frame_count=30,
        _format_mean=GestureDetectorNode._format_mean,
        get_logger=lambda: logger,
    )
    samples = [
        {
            "callback_total_ms": 100.0,
            "engine_total_ms": 90.0,
            "yolo_track_ms": 40.0,
            "yolo_preprocess_ms": None,
            "onnx_inference_ms": None,
            "yolo_postprocess_ms": None,
            "pose_extract_ms": 5.0,
            "mediapipe_ms": 35.0,
            "hand_assign_ms": 1.0,
            "gesture_logic_ms": 2.0,
            "overlay_ms": 6.0,
            "result_pack_ms": 1.0,
            "cv_bridge_ms": 1.0,
            "event_publish_ms": 0.0,
            "debug_publish_ms": 3.0,
        }
    ]
    summary = GestureDetectorNode._summarize_timing_samples(samples)
    result = SimpleNamespace(people=[object()], events=[])

    GestureDetectorNode._log_timing_summary(
        node,
        summary=summary,
        sample_count=30,
        processing_fps=10.0,
        result=result,
    )

    lines = [call.args[0] for call in logger.info.call_args_list]
    assert len(lines) == 3
    assert lines[0].startswith("Perf[30]:")
    assert lines[1].startswith("Engine avg ms:")
    assert "[pre=n/a model=n/a post=n/a]" in lines[1]
    assert lines[2].startswith("ROS avg ms:")


def test_algorithm_defaults_include_updated_gesture_args():
    args = _algorithm_defaults(GestureEngineConfig())

    assert args.wave_min_lateral_range == 0.075
    assert args.wave_min_direction_changes == 2
    assert args.wave_history_gap_reset == 0.18
    assert args.disable_mid_come_adaptive is False
    assert args.mid_come_min_stroke_ratio == 0.085
    assert args.disable_pose_fallback is False
    assert args.pose_fallback_max_box_height == 0.50
    assert args.pose_only_hand_surface_max_box_height == 0.42
    assert args.min_gesture_hard_person_conf == 0.35
    assert args.min_gesture_hard_box_height == 0.06
    assert args.degraded_gesture_seen_frames == 10
    assert args.degraded_ownership_dominance_ratio == 0.85


def test_public_event_preserves_updated_algorithm_fields():
    event = SimpleNamespace(
        track_id=9,
        action="WAVE",
        source="POSE_WAVE",
        arm="LEFT",
        score=1.75,
        ownership_score=0.91,
        wrist_travel_px=21.0,
        wrist_travel_ratio=0.33,
        hand_surface="UNKNOWN",
        hand_surface_score=None,
        motion_source="POSE",
        radial_fraction=0.27,
        bbox_height_ratio=0.12,
        pose_fallback=True,
        quality_band="DEGRADED",
        timestamp=123.4,
    )
    person = SimpleNamespace(
        confidence=0.77,
        center_x=0.41,
        center_y=0.52,
        bbox=(10, 20, 110, 220),
        bbox_height_ratio=0.12,
    )

    detected = GestureEngine._public_event(event, person)

    assert detected.track_id == 9
    assert detected.action == "WAVE"
    assert detected.radial_fraction == 0.27
    assert detected.bbox_height_ratio == 0.12
    assert detected.pose_fallback is True
    assert detected.quality_band == "DEGRADED"
    assert detected.to_dict()["pose_fallback"] is True


class _TensorLike:
    def __init__(self, value):
        self._value = np.asarray(value)

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value.tolist()

    def numpy(self):
        return self._value


def test_process_frame_passes_person_to_updated_algorithm(monkeypatch):
    captured = {}

    class FakeBoxes:
        id = _TensorLike([42])
        xyxy = _TensorLike([[10.0, 20.0, 110.0, 220.0]])
        conf = _TensorLike([0.88])

    class FakeKeypoints:
        data = _TensorLike(np.zeros((1, 17, 3), dtype=np.float32))

    class FakeResult:
        boxes = FakeBoxes()
        keypoints = FakeKeypoints()
        speed = {
            "preprocess": 1.5,
            "inference": 7.5,
            "postprocess": 2.5,
        }

    class FakeModel:
        def track(self, **_kwargs):
            return [FakeResult()]

    def fake_update_person_gesture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            track_id=42,
            action="COME",
            source="TEST",
            arm="RIGHT",
            score=1.2,
            ownership_score=0.8,
            wrist_travel_px=20.0,
            wrist_travel_ratio=0.3,
            hand_surface="BACK",
            hand_surface_score=0.5,
            motion_source="ARM",
            radial_fraction=0.6,
            bbox_height_ratio=0.5,
            pose_fallback=False,
            quality_band="NORMAL",
            timestamp=10.0,
        )

    monkeypatch.setattr(
        gesture_engine_module,
        "extract_arm_features",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        gesture_engine_module,
        "assign_hands_uniquely",
        lambda **_kwargs: {42: {"LEFT": None, "RIGHT": None}},
    )
    monkeypatch.setattr(
        gesture_engine_module,
        "make_person_state",
        lambda track_id, _args, _now: SimpleNamespace(
            track_id=track_id,
            last_seen=0.0,
            seen_frames=0,
            begin_action_cooldown=Mock(),
        ),
    )
    monkeypatch.setattr(
        gesture_engine_module,
        "update_person_gesture",
        fake_update_person_gesture,
    )
    monkeypatch.setattr(
        gesture_engine_module,
        "person_motion_ownership_score",
        lambda **_kwargs: (0.8, None),
    )
    monkeypatch.setattr(
        gesture_engine_module,
        "event_source_is_eligible",
        lambda **_kwargs: (True, ""),
    )

    engine = object.__new__(GestureEngine)
    engine.config = GestureEngineConfig()
    engine.device = "cpu"
    engine._args = _algorithm_defaults(engine.config)
    engine._model = FakeModel()
    engine._hand_detector = None
    engine._person_states = {}
    engine._closed = False
    engine._log = None
    monkeypatch.setattr(
        gesture_engine_module.time,
        "perf_counter",
        Mock(
            side_effect=[
                0.00,
                0.01,
                0.02,
                0.05,
                0.06,
                0.08,
                0.09,
                0.10,
            ]
        ),
    )

    result = engine.process_frame(
        np.zeros((400, 600, 3), dtype=np.uint8),
        timestamp=10.0,
    )

    assert captured["person"].track_id == 42
    assert captured["person"].bbox_height_ratio == 0.5
    assert result.events[0].track_id == 42
    assert result.inference_ms == 100.0
    assert result.timings.yolo_track_ms == 10.0
    assert result.timings.yolo_preprocess_ms == 1.5
    assert result.timings.onnx_inference_ms == 7.5
    assert result.timings.yolo_postprocess_ms == 2.5
    assert np.isclose(result.timings.pose_extract_ms, 10.0)
    assert np.isclose(result.timings.mediapipe_ms, 30.0)
    assert np.isclose(result.timings.hand_assign_ms, 10.0)
    assert np.isclose(result.timings.gesture_logic_ms, 20.0)
    assert np.isclose(result.timings.overlay_ms, 10.0)
    assert np.isclose(result.timings.result_pack_ms, 10.0)
    assert np.isclose(result.timings.engine_total_ms, 100.0)


def test_missing_ultralytics_speed_is_safe():
    result = SimpleNamespace(speed=None)

    assert GestureEngine._result_speed_ms(result, "inference") is None
    assert GestureEngine._result_speed_ms(
        SimpleNamespace(speed={"inference": "bad"}),
        "inference",
    ) is None

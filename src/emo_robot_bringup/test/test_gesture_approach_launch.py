"""Tests for gesture launch model-parameter precedence."""

import importlib.util
from pathlib import Path
import sys

from launch import LaunchContext

_LAUNCH_MODULE = None


def _load_launch_module():
    global _LAUNCH_MODULE
    if _LAUNCH_MODULE is not None:
        return _LAUNCH_MODULE

    for module_name in tuple(sys.modules):
        if module_name in {"ament_index_python", "rclpy", "launch_ros"} or (
            module_name.startswith(
                ("ament_index_python.", "rclpy.", "launch_ros.")
            )
        ):
            del sys.modules[module_name]

    launch_path = (
        Path(__file__).resolve().parents[1]
        / "launch"
        / "gesture_approach.launch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gesture_approach_launch",
        launch_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LAUNCH_MODULE = module
    return _LAUNCH_MODULE


def _context(**values):
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "model_backend": "",
            "model_path": "",
            "yolo_device": "",
            **values,
        }
    )
    return context


def test_empty_launch_arguments_do_not_override_yaml():
    module = _load_launch_module()

    overrides = module._model_override_values(_context())

    assert overrides == {}


def test_nonempty_launch_arguments_override_individually():
    module = _load_launch_module()

    overrides = module._model_override_values(
        _context(
            model_backend="pt",
            model_path="/tmp/custom.pt",
            yolo_device="cpu",
        )
    )

    assert overrides == {
        "model_backend": "pt",
        "model_path": "/tmp/custom.pt",
        "yolo_device": "cpu",
    }

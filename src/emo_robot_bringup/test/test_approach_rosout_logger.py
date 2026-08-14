"""Tests for the approach-pipeline /rosout text formatter."""

from rcl_interfaces.msg import Log

from emo_robot_bringup.approach_rosout_logger import (
    format_rosout_message,
    logger_is_selected,
    severity_name,
)


def test_selected_node_and_child_loggers_are_accepted():
    assert logger_is_selected("gesture_detector")
    assert logger_is_selected("/robot/person_tracker")
    assert logger_is_selected("approach_controller.safety")


def test_unrelated_loggers_are_rejected():
    assert not logger_is_selected("usb_camera")
    assert not logger_is_selected("interaction_responder")
    assert not logger_is_selected("approach_controller_backup")


def test_severity_names_include_unknown_values():
    assert severity_name(20) == "INFO"
    assert severity_name(40) == "ERROR"
    assert severity_name(77) == "LEVEL_77"


def test_formatted_message_is_single_line_and_keeps_identity():
    message = Log()
    message.stamp.sec = 1
    message.stamp.nanosec = 250_000_000
    message.level = 30
    message.name = "person_tracker"
    message.msg = "target lost\nwaiting for WAVE"

    result = format_rosout_message(message)

    assert "[WARN] [person_tracker]" in result
    assert "target lost\\nwaiting for WAVE" in result
    assert "\n" not in result

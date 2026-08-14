"""Tests for single-person WAVE/COME and LiDAR target fusion."""

from types import SimpleNamespace

import pytest

from emo_robot_interfaces.msg import GestureDetection
from emo_robot_perception.person_tracker_node import (
    GestureTrigger,
    LidarObservation,
    SinglePersonTargetFusion,
)


def lidar(
    when,
    *,
    valid=True,
    forward=2.0,
    lateral=0.2,
    angle=0.1,
    points=100,
):
    return LidarObservation(
        valid=valid,
        forward_m=forward,
        lateral_m=lateral,
        angle_rad=angle,
        point_count=points,
        received_s=when,
    )


def trigger(when, gesture=GestureDetection.GESTURE_WAVE, track_id=7):
    return GestureTrigger(
        track_id=track_id,
        confidence=0.9,
        gesture=gesture,
        received_s=when,
    )


@pytest.mark.parametrize(
    "gesture",
    [
        GestureDetection.GESTURE_WAVE,
        GestureDetection.GESTURE_COME,
    ],
)
def test_wave_and_come_match_recent_lidar_and_preserve_identity(gesture):
    fusion = SinglePersonTargetFusion()
    assert not fusion.accept_lidar(lidar(10.0)).valid

    target = fusion.accept_trigger(trigger(10.2, gesture))

    assert target.valid
    assert target.track_id == 7
    assert target.confidence == 0.9
    assert target.horizontal_error == 0.1
    assert target.lateral_m == 0.2
    assert target.distance_m == 2.0
    assert target.gesture_confirmed


@pytest.mark.parametrize(
    "gesture",
    [
        GestureDetection.GESTURE_WAVE,
        GestureDetection.GESTURE_COME,
    ],
)
def test_lidar_after_either_gesture_activates_within_match_window(gesture):
    fusion = SinglePersonTargetFusion()
    assert not fusion.accept_trigger(trigger(5.0, gesture)).valid

    target = fusion.accept_lidar(lidar(5.6, forward=1.8))

    assert target.valid
    assert target.distance_m == 1.8


@pytest.mark.parametrize(
    "valid,gesture,expected",
    [
        (True, GestureDetection.GESTURE_WAVE, True),
        (True, GestureDetection.GESTURE_COME, True),
        (False, GestureDetection.GESTURE_WAVE, False),
        (False, GestureDetection.GESTURE_COME, False),
        (True, GestureDetection.GESTURE_NONE, False),
        (True, 99, False),
    ],
)
def test_only_valid_wave_and_come_are_approach_gestures(
    valid,
    gesture,
    expected,
):
    assert (
        SinglePersonTargetFusion.is_approach_gesture(valid, gesture)
        is expected
    )


def test_invalid_or_sparse_lidar_does_not_activate():
    fusion = SinglePersonTargetFusion(min_target_points=50)
    fusion.accept_trigger(trigger(1.0))

    assert not fusion.accept_lidar(lidar(1.1, valid=False)).valid
    assert not fusion.accept_lidar(lidar(1.2, points=49)).valid
    assert not fusion.current(2.1).valid
    assert fusion.last_error == "no valid LiDAR target near the gesture event"


def test_target_jump_enters_lost_state_and_reacquires_same_session():
    fusion = SinglePersonTargetFusion(
        max_forward_jump_m=0.75,
        max_lateral_jump_m=0.5,
        reacquire_timeout_s=2.0,
    )
    fusion.accept_trigger(trigger(1.0, GestureDetection.GESTURE_COME))
    assert fusion.accept_lidar(lidar(1.1)).valid

    target = fusion.accept_lidar(lidar(1.2, forward=3.0))

    assert not target.valid
    assert target.track_id == 7
    assert target.gesture_confirmed
    assert fusion.active
    assert "jumped" in fusion.last_error
    reacquired = fusion.accept_lidar(lidar(1.3, forward=3.0))
    assert reacquired.valid
    assert reacquired.track_id == 7


def test_lidar_watchdog_allows_reacquire_before_session_timeout():
    fusion = SinglePersonTargetFusion(
        lidar_timeout_s=0.4,
        reacquire_timeout_s=2.0,
        tracking_session_timeout_s=5.5,
    )
    fusion.accept_trigger(trigger(10.0))
    assert fusion.accept_lidar(lidar(10.1)).valid

    lost = fusion.current(10.51)
    assert not lost.valid
    assert lost.track_id == 7
    assert lost.gesture_confirmed
    assert fusion.last_error == "LiDAR target timed out"
    assert fusion.active
    reacquired = fusion.accept_lidar(lidar(10.8, lateral=0.4))
    assert reacquired.valid
    assert reacquired.track_id == 7
    assert reacquired.lateral_m == 0.4

    fusion.accept_trigger(
        trigger(20.0, GestureDetection.GESTURE_COME, track_id=8)
    )
    assert fusion.accept_lidar(lidar(20.1)).valid
    assert not fusion.current(25.5).valid
    assert fusion.last_error == "tracking session expired"


def test_lost_target_releases_when_reacquire_timeout_elapses():
    fusion = SinglePersonTargetFusion(
        lidar_timeout_s=0.4,
        reacquire_timeout_s=1.0,
        tracking_session_timeout_s=5.5,
    )
    fusion.accept_trigger(trigger(10.0))
    assert fusion.accept_lidar(lidar(10.1)).valid

    lost = fusion.current(10.51)
    assert not lost.valid
    assert lost.track_id == 7
    assert lost.gesture_confirmed
    assert fusion.active
    released = fusion.current(11.51)
    assert not released.valid
    assert released.track_id == -1
    assert not released.gesture_confirmed

    assert not fusion.active
    assert fusion.last_error == "LiDAR target reacquire timed out"


def test_mixed_repeated_gestures_do_not_replace_or_extend_active_session():
    fusion = SinglePersonTargetFusion(
        lidar_timeout_s=0.4,
        tracking_session_timeout_s=2.0,
    )
    fusion.accept_trigger(
        trigger(1.0, GestureDetection.GESTURE_WAVE, track_id=3)
    )
    assert fusion.accept_lidar(lidar(1.1)).track_id == 3

    repeated = fusion.accept_trigger(
        trigger(1.2, GestureDetection.GESTURE_COME, track_id=99)
    )

    assert repeated.valid
    assert repeated.track_id == 3
    assert fusion.accept_lidar(lidar(2.9)).track_id == 3
    assert not fusion.current(3.0).valid
    assert fusion.last_error == "tracking session expired"


class _CapturingFusion:
    def __init__(self):
        self.triggers = []

    def is_approach_gesture(self, valid, gesture):
        return SinglePersonTargetFusion.is_approach_gesture(valid, gesture)

    def accept_trigger(self, trigger_value):
        self.triggers.append(trigger_value)
        return SimpleNamespace(valid=False)


class _SilentLogger:
    def info(self, _message):
        pass


class _GestureCallbackHarness:
    def __init__(self):
        self._fusion = _CapturingFusion()
        self._last_source_header = None

    def get_logger(self):
        return _SilentLogger()

    def _publish_result(self, _result, _header):
        raise AssertionError("an invalid fusion result must not be published")


@pytest.mark.parametrize(
    "valid,gesture,accepted",
    [
        (True, GestureDetection.GESTURE_WAVE, True),
        (True, GestureDetection.GESTURE_COME, True),
        (False, GestureDetection.GESTURE_WAVE, False),
        (True, GestureDetection.GESTURE_NONE, False),
        (True, 99, False),
    ],
)
def test_node_callback_accepts_only_valid_wave_and_come(
    valid,
    gesture,
    accepted,
):
    from emo_robot_perception.person_tracker_node import PersonTrackerNode

    harness = _GestureCallbackHarness()
    message = SimpleNamespace(
        valid=valid,
        gesture=gesture,
        track_id=42,
        confidence=0.8,
    )

    PersonTrackerNode._on_gesture(harness, message)

    assert bool(harness._fusion.triggers) is accepted
    if accepted:
        captured = harness._fusion.triggers[0]
        assert captured.gesture == gesture
        assert captured.track_id == 42
        assert captured.confidence == pytest.approx(0.8)

"""Unit tests for ROS-independent LiDAR target estimation."""

import math

import numpy as np
import pytest

from emo_robot_perception.lidar_target_estimator import (
    LidarTargetEstimator,
    LidarTargetEstimatorConfig,
)


def _config(**overrides):
    values = {
        "forward_min_m": 0.4,
        "forward_max_m": 5.0,
        "vertical_min_m": -1.0,
        "vertical_max_m": 1.0,
        "lateral_abs_max_m": 2.0,
        "lateral_bin_size_m": 0.1,
        "cluster_half_width_m": 0.2,
        "min_cluster_points": 3,
        "smoothing_window": 5,
        "tracking_enabled": True,
        "candidate_count": 5,
        "track_min_cluster_points": 3,
        "max_tracking_forward_jump_m": 1.0,
        "max_tracking_lateral_jump_m": 0.8,
        "max_tracking_angle_jump_deg": 18.0,
        "lost_grace_frames": 5,
    }
    values.update(overrides)
    return LidarTargetEstimatorConfig(**values)


def _cluster(forward, lateral, count=3, vertical=0.0):
    offsets = np.linspace(-0.02, 0.02, count, dtype=np.float32)
    return np.column_stack(
        (
            np.full(count, forward, dtype=np.float32),
            np.full(count, vertical, dtype=np.float32),
            np.full(count, lateral, dtype=np.float32) + offsets,
        )
    )


def test_invalid_and_out_of_roi_points_are_filtered():
    estimator = LidarTargetEstimator(_config())
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [1.0, 0.0, 3.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = estimator.estimate(points)

    assert result.valid is False
    assert result.filtered_points.shape == (1, 3)


def test_too_few_points_returns_invalid_result():
    estimator = LidarTargetEstimator(_config(min_cluster_points=4))

    result = estimator.estimate(_cluster(2.0, 0.35, count=3))

    assert result.valid is False
    assert result.point_count == 0


def test_cluster_median_produces_distance_offset_and_angle():
    estimator = LidarTargetEstimator(_config())
    points = np.vstack(
        (
            _cluster(2.0, 0.55, count=5),
            np.asarray([[3.0, 0.0, -1.0]], dtype=np.float32),
        )
    )

    result = estimator.estimate(points)

    assert result.valid is True
    assert result.forward_m == pytest.approx(2.0)
    assert result.lateral_m == pytest.approx(0.55)
    assert result.angle_rad == pytest.approx(math.atan2(0.55, 2.0))
    assert result.point_count == 5
    assert result.cluster_points.shape == (5, 3)


def test_five_sample_median_smooths_target():
    estimator = LidarTargetEstimator(_config(tracking_enabled=False))

    estimates = [
        estimator.estimate(_cluster(forward, 0.05))
        for forward in (1.0, 1.1, 4.0, 1.2, 1.3)
    ]

    assert estimates[-1].forward_m == pytest.approx(1.2)


def test_invalid_frame_clears_smoothing_history():
    estimator = LidarTargetEstimator(_config(tracking_enabled=False))
    estimator.estimate(_cluster(1.0, 0.05))
    invalid = estimator.estimate(np.empty((0, 3), dtype=np.float32))
    after_loss = estimator.estimate(_cluster(3.0, 0.05))

    assert invalid.valid is False
    assert after_loss.forward_m == pytest.approx(3.0)


def test_initial_tracking_target_uses_densest_candidate():
    estimator = LidarTargetEstimator(_config(smoothing_window=1))
    points = np.vstack(
        (
            _cluster(2.0, 0.35, count=3),
            _cluster(1.5, -0.75, count=6),
        )
    )

    result = estimator.estimate(points)

    assert result.valid is True
    assert result.forward_m == pytest.approx(1.5)
    assert result.lateral_m == pytest.approx(-0.75)
    assert result.point_count == 6


def test_tracking_selects_near_candidate_when_densest_candidate_jumps():
    estimator = LidarTargetEstimator(
        _config(
            min_cluster_points=5,
            track_min_cluster_points=3,
            smoothing_window=1,
            max_tracking_lateral_jump_m=0.5,
            max_tracking_angle_jump_deg=20.0,
        )
    )
    estimator.estimate(_cluster(2.0, 0.20, count=5))
    points = np.vstack(
        (
            _cluster(2.1, 0.25, count=3),
            _cluster(2.0, 1.50, count=12),
        )
    )

    result = estimator.estimate(points)

    assert result.valid is True
    assert result.forward_m == pytest.approx(2.1)
    assert result.lateral_m == pytest.approx(0.25)
    assert result.point_count == 3


def test_tracking_holds_previous_target_until_lost_grace_expires():
    estimator = LidarTargetEstimator(
        _config(
            smoothing_window=1,
            lost_grace_frames=2,
            max_tracking_lateral_jump_m=0.4,
        )
    )
    initial = estimator.estimate(_cluster(2.0, 0.10, count=5))
    jumped = _cluster(2.0, 1.20, count=8)

    first_loss = estimator.estimate(jumped)
    second_loss = estimator.estimate(jumped)
    expired = estimator.estimate(jumped)

    assert initial.valid is True
    assert first_loss.valid is True
    assert second_loss.valid is True
    assert first_loss.forward_m == pytest.approx(initial.forward_m)
    assert first_loss.lateral_m == pytest.approx(initial.lateral_m)
    assert expired.valid is False


def test_tracking_min_points_can_be_lower_than_initial_min_points():
    estimator = LidarTargetEstimator(
        _config(
            min_cluster_points=5,
            track_min_cluster_points=3,
            smoothing_window=1,
        )
    )

    too_weak_to_start = estimator.estimate(_cluster(2.0, 0.10, count=3))
    started = estimator.estimate(_cluster(2.0, 0.10, count=5))
    tracked = estimator.estimate(_cluster(2.1, 0.12, count=3))

    assert too_weak_to_start.valid is False
    assert started.valid is True
    assert tracked.valid is True
    assert tracked.point_count == 3


def test_tracking_can_be_disabled_for_legacy_global_densest_behavior():
    estimator = LidarTargetEstimator(
        _config(
            tracking_enabled=False,
            min_cluster_points=3,
            smoothing_window=1,
        )
    )
    estimator.estimate(_cluster(2.0, 0.20, count=5))
    points = np.vstack(
        (
            _cluster(2.1, 0.25, count=3),
            _cluster(2.0, 1.50, count=12),
        )
    )

    result = estimator.estimate(points)

    assert result.valid is True
    assert result.lateral_m == pytest.approx(1.50)
    assert result.point_count == 12


def test_default_axes_match_x2_recorded_data_convention():
    estimator = LidarTargetEstimator(_config())
    points = np.asarray(
        [
            [2.0, 0.0, 0.53],
            [2.0, 0.0, 0.55],
            [2.0, 0.0, 0.57],
        ],
        dtype=np.float32,
    )

    result = estimator.estimate(points)

    assert result.valid is True
    assert result.forward_m == pytest.approx(2.0)
    assert result.lateral_m == pytest.approx(0.55)


def test_axis_names_must_be_unique():
    with pytest.raises(ValueError, match="must differ"):
        LidarTargetEstimator(
            _config(lateral_axis="x"),
        )


def test_bad_input_shape_is_rejected():
    estimator = LidarTargetEstimator(_config())

    with pytest.raises(ValueError, match="shape"):
        estimator.estimate(np.ones((4, 2), dtype=np.float32))

"""ROS-independent LiDAR target estimation for prototype navigation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque

import numpy as np


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class LidarTargetEstimatorConfig:
    """Configuration for ROI filtering and lateral-density clustering."""

    forward_axis: str = "x"
    lateral_axis: str = "z"
    vertical_axis: str = "y"
    forward_min_m: float = 0.4
    forward_max_m: float = 10.0
    vertical_min_m: float = -1.62
    vertical_max_m: float = 0.38
    lateral_abs_max_m: float = 4.0
    lateral_bin_size_m: float = 0.10
    cluster_half_width_m: float = 0.25
    min_cluster_points: int = 50
    smoothing_window: int = 5
    tracking_enabled: bool = True
    candidate_count: int = 5
    track_min_cluster_points: int = 25
    max_tracking_forward_jump_m: float = 1.0
    max_tracking_lateral_jump_m: float = 0.8
    max_tracking_angle_jump_deg: float = 18.0
    lost_grace_frames: int = 5

    def validate(self) -> None:
        """Raise ValueError when configuration is unsafe or ambiguous."""
        axes = (
            self.forward_axis.lower(),
            self.lateral_axis.lower(),
            self.vertical_axis.lower(),
        )
        if any(axis not in _AXIS_INDEX for axis in axes):
            raise ValueError("axis parameters must be x, y, or z")
        if len(set(axes)) != 3:
            raise ValueError("forward, lateral, and vertical axes must differ")
        if self.forward_min_m >= self.forward_max_m:
            raise ValueError("forward_min_m must be less than forward_max_m")
        if self.vertical_min_m >= self.vertical_max_m:
            raise ValueError("vertical_min_m must be less than vertical_max_m")
        if self.lateral_abs_max_m <= 0.0:
            raise ValueError("lateral_abs_max_m must be greater than zero")
        if self.lateral_bin_size_m <= 0.0:
            raise ValueError("lateral_bin_size_m must be greater than zero")
        if self.cluster_half_width_m <= 0.0:
            raise ValueError("cluster_half_width_m must be greater than zero")
        if self.min_cluster_points <= 0:
            raise ValueError("min_cluster_points must be greater than zero")
        if self.smoothing_window <= 0:
            raise ValueError("smoothing_window must be greater than zero")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be greater than zero")
        if self.track_min_cluster_points <= 0:
            raise ValueError(
                "track_min_cluster_points must be greater than zero"
            )
        if (
            self.max_tracking_forward_jump_m <= 0.0
            or self.max_tracking_lateral_jump_m <= 0.0
            or self.max_tracking_angle_jump_deg <= 0.0
        ):
            raise ValueError("tracking jump limits must be greater than zero")
        if self.lost_grace_frames < 0:
            raise ValueError("lost_grace_frames must be non-negative")


@dataclass(frozen=True)
class LidarTargetEstimate:
    """One target estimate plus point sets used for visualization."""

    valid: bool = False
    forward_m: float = 0.0
    lateral_m: float = 0.0
    angle_rad: float = 0.0
    point_count: int = 0
    filtered_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        repr=False,
        compare=False,
    )
    cluster_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _ClusterCandidate:
    """One lateral-density cluster candidate before smoothing."""

    forward_m: float
    lateral_m: float
    angle_rad: float
    point_count: int
    peak_count: int
    cluster_points: np.ndarray = field(repr=False, compare=False)


class LidarTargetEstimator:
    """Track a lateral-density cluster inside a configurable 3-D ROI."""

    def __init__(self, config: LidarTargetEstimatorConfig) -> None:
        config.validate()
        self.config = config
        self._forward_index = _AXIS_INDEX[config.forward_axis.lower()]
        self._lateral_index = _AXIS_INDEX[config.lateral_axis.lower()]
        self._vertical_index = _AXIS_INDEX[config.vertical_axis.lower()]
        self._history: Deque[tuple[float, float]] = deque(
            maxlen=config.smoothing_window,
        )
        self._last_raw_target: _ClusterCandidate | None = None
        self._last_estimate: LidarTargetEstimate | None = None
        self._lost_frames = 0

    def reset(self) -> None:
        """Clear tracking and smoothing history after target release."""
        self._history.clear()
        self._last_raw_target = None
        self._last_estimate = None
        self._lost_frames = 0

    def estimate(self, xyz_points: np.ndarray) -> LidarTargetEstimate:
        """Estimate a target from an array whose columns are x, y, z."""
        points = np.asarray(xyz_points, dtype=np.float32)
        if points.size == 0:
            return self._handle_tracking_loss()
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("xyz_points must have shape (N, 3) or wider")
        points = points[:, :3]

        filtered = self._filter_points(points)
        if filtered.size == 0:
            return self._handle_tracking_loss(filtered_points=filtered)

        if not self.config.tracking_enabled:
            return self._estimate_legacy(filtered)

        candidates = self._find_candidates(filtered)
        selected = self._select_candidate(candidates)
        if selected is None:
            return self._handle_tracking_loss(filtered_points=filtered)
        return self._accept_candidate(selected, filtered)

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Return finite points inside the configured 3-D ROI."""
        forward = points[:, self._forward_index]
        lateral = points[:, self._lateral_index]
        vertical = points[:, self._vertical_index]
        cfg = self.config
        valid_mask = (
            np.isfinite(points).all(axis=1)
            & (forward > cfg.forward_min_m)
            & (forward < cfg.forward_max_m)
            & (vertical > cfg.vertical_min_m)
            & (vertical < cfg.vertical_max_m)
            & (np.abs(lateral) < cfg.lateral_abs_max_m)
        )
        return points[valid_mask]

    def _estimate_legacy(self, filtered: np.ndarray) -> LidarTargetEstimate:
        """Preserve the original global densest-cluster behavior."""
        candidates = self._find_candidates(
            filtered,
            candidate_count=1,
            min_cluster_points=self.config.min_cluster_points,
            require_peak_min=True,
        )
        if not candidates:
            self.reset()
            return LidarTargetEstimate(filtered_points=filtered)
        return self._accept_candidate(candidates[0], filtered)

    def _find_candidates(
        self,
        filtered: np.ndarray,
        *,
        candidate_count: int | None = None,
        min_cluster_points: int | None = None,
        require_peak_min: bool = False,
    ) -> list[_ClusterCandidate]:
        """Return non-overlapping lateral-density candidates by bin count."""
        cfg = self.config
        if candidate_count is None:
            candidate_count = cfg.candidate_count
        if min_cluster_points is None:
            min_cluster_points = (
                cfg.track_min_cluster_points
                if self._last_raw_target is not None
                else cfg.min_cluster_points
            )

        filtered_lateral = filtered[:, self._lateral_index]
        edges = np.arange(
            -cfg.lateral_abs_max_m,
            cfg.lateral_abs_max_m + cfg.lateral_bin_size_m,
            cfg.lateral_bin_size_m,
            dtype=np.float64,
        )
        histogram, edges = np.histogram(filtered_lateral, bins=edges)
        candidates: list[_ClusterCandidate] = []
        used_peaks: list[float] = []
        for peak_index in np.argsort(histogram)[::-1]:
            peak_index = int(peak_index)
            peak_count = int(histogram[peak_index])
            if peak_count <= 0:
                break
            if require_peak_min and peak_count < min_cluster_points:
                break
            lateral_peak = 0.5 * (edges[peak_index] + edges[peak_index + 1])
            if any(
                abs(lateral_peak - used_peak) <= cfg.cluster_half_width_m
                for used_peak in used_peaks
            ):
                continue
            cluster_mask = (
                (filtered_lateral >= lateral_peak - cfg.cluster_half_width_m)
                & (filtered_lateral <= lateral_peak + cfg.cluster_half_width_m)
            )
            cluster = filtered[cluster_mask]
            if cluster.shape[0] < min_cluster_points:
                continue
            raw_forward = float(np.median(cluster[:, self._forward_index]))
            raw_lateral = float(np.median(cluster[:, self._lateral_index]))
            raw_angle = math.atan2(raw_lateral, max(raw_forward, 0.05))
            candidates.append(
                _ClusterCandidate(
                    forward_m=raw_forward,
                    lateral_m=raw_lateral,
                    angle_rad=raw_angle,
                    point_count=int(cluster.shape[0]),
                    peak_count=peak_count,
                    cluster_points=cluster,
                )
            )
            used_peaks.append(lateral_peak)
            if len(candidates) >= candidate_count:
                break
        return candidates

    def _select_candidate(
        self,
        candidates: list[_ClusterCandidate],
    ) -> _ClusterCandidate | None:
        """Choose the densest initial target or closest gated track update."""
        if not candidates:
            return None
        previous = self._last_raw_target
        if previous is None:
            return candidates[0]

        best: tuple[float, int, _ClusterCandidate] | None = None
        max_angle_jump_rad = math.radians(
            self.config.max_tracking_angle_jump_deg
        )
        for candidate in candidates:
            forward_delta = abs(candidate.forward_m - previous.forward_m)
            lateral_delta = abs(candidate.lateral_m - previous.lateral_m)
            angle_delta = abs(candidate.angle_rad - previous.angle_rad)
            if (
                forward_delta > self.config.max_tracking_forward_jump_m
                or lateral_delta > self.config.max_tracking_lateral_jump_m
                or angle_delta > max_angle_jump_rad
            ):
                continue
            score = (
                forward_delta / self.config.max_tracking_forward_jump_m
                + lateral_delta / self.config.max_tracking_lateral_jump_m
                + angle_delta / max_angle_jump_rad
            )
            item = (score, -candidate.point_count, candidate)
            if best is None or item[:2] < best[:2]:
                best = item
        if best is None:
            return None
        return best[2]

    def _accept_candidate(
        self,
        candidate: _ClusterCandidate,
        filtered: np.ndarray,
    ) -> LidarTargetEstimate:
        """Update tracking state and return a smoothed target estimate."""
        self._history.append((candidate.forward_m, candidate.lateral_m))
        self._last_raw_target = candidate
        self._lost_frames = 0
        smoothed = np.asarray(self._history, dtype=np.float64)
        target_forward = float(np.median(smoothed[:, 0]))
        target_lateral = float(np.median(smoothed[:, 1]))
        target_angle = math.atan2(
            target_lateral,
            max(target_forward, 0.05),
        )

        estimate = LidarTargetEstimate(
            valid=True,
            forward_m=target_forward,
            lateral_m=target_lateral,
            angle_rad=target_angle,
            point_count=candidate.point_count,
            filtered_points=filtered,
            cluster_points=candidate.cluster_points,
        )
        self._last_estimate = estimate
        return estimate

    def _handle_tracking_loss(
        self,
        *,
        filtered_points: np.ndarray | None = None,
    ) -> LidarTargetEstimate:
        """Temporarily hold the last target before declaring invalid."""
        if filtered_points is None:
            filtered_points = np.empty((0, 3), dtype=np.float32)
        if (
            self.config.tracking_enabled
            and self._last_estimate is not None
            and self._lost_frames < self.config.lost_grace_frames
        ):
            self._lost_frames += 1
            return LidarTargetEstimate(
                valid=True,
                forward_m=self._last_estimate.forward_m,
                lateral_m=self._last_estimate.lateral_m,
                angle_rad=self._last_estimate.angle_rad,
                point_count=self._last_estimate.point_count,
                filtered_points=filtered_points,
                cluster_points=self._last_estimate.cluster_points,
            )
        self.reset()
        return LidarTargetEstimate(filtered_points=filtered_points)

    def project_forward_lateral(
        self,
        xyz_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return configured forward/lateral columns for visualization."""
        points = np.asarray(xyz_points)
        if points.size == 0:
            empty = np.empty((0,), dtype=np.float32)
            return empty, empty
        return (
            points[:, self._forward_index],
            points[:, self._lateral_index],
        )

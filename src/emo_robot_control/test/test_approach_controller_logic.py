"""Tests for fail-closed two-phase single-person approach decisions."""

import math
from types import SimpleNamespace

import pytest

from aimdk_msgs.msg import McAction, McActionStatus

from emo_robot_control.approach_controller_node import (
    ApproachControllerNode,
    ApproachPolicy,
    is_approach_mode_running,
)
from emo_robot_motion.mc_action_state_client import McActionState


@pytest.mark.parametrize(
    "action",
    [McAction.STAND_DEFAULT, McAction.LOCOMOTION_DEFAULT],
)
def test_running_stand_or_locomotion_mode_is_allowed(action):
    state = McActionState(
        action=action,
        description="allowed",
        status=McActionStatus.RUNNING,
        response_code=0,
    )

    assert is_approach_mode_running(state)


@pytest.mark.parametrize(
    "description",
    ["STAND_DEFAULT", "LOCOMOTION_DEFAULT", " stand_default "],
)
def test_running_mode_description_accepts_aimdk_zero_action(description):
    state = McActionState(
        action=0,
        description=description,
        status=McActionStatus.RUNNING,
        response_code=0,
    )

    assert is_approach_mode_running(state)


@pytest.mark.parametrize(
    "state",
    [
        None,
        McActionState(
            action=McAction.STAND_DEFAULT,
            description="transition",
            status=McActionStatus.TRANSITION,
            response_code=0,
        ),
        McActionState(
            action=-999,
            description="other",
            status=McActionStatus.RUNNING,
            response_code=0,
        ),
        McActionState(
            action=0,
            description="PASSIVE_DEFAULT",
            status=McActionStatus.RUNNING,
            response_code=0,
        ),
        McActionState(
            action=McAction.LOCOMOTION_DEFAULT,
            description="response error",
            status=McActionStatus.RUNNING,
            response_code=1,
        ),
    ],
)
def test_transition_other_and_error_states_are_rejected(state):
    assert not is_approach_mode_running(state)


def test_person_target_invalid_message_can_still_mark_active_session():
    active_lost = SimpleNamespace(
        valid=False,
        gesture_confirmed=True,
        track_id=7,
    )
    released = SimpleNamespace(
        valid=False,
        gesture_confirmed=False,
        track_id=-1,
    )

    assert ApproachControllerNode._message_session_active(active_lost)
    assert not ApproachControllerNode._message_session_active(released)


def make_aligned(policy, start_s=0.1):
    decisions = []
    for index in range(policy.align_stable_frames):
        decisions.append(
            policy.decide(2.0, 0.1, math.radians(2.0), start_s + index * 0.1)
        )
    return decisions


def test_align_left_and_right_commands_keep_forward_zero():
    left_policy = ApproachPolicy()
    left_policy.reset(0.0)
    left = left_policy.decide(2.0, 0.3, math.radians(12.0), 0.1)

    right_policy = ApproachPolicy()
    right_policy.reset(0.0)
    right = right_policy.decide(2.0, -0.3, math.radians(-12.0), 0.1)

    assert (left.forward, left.lateral, left.angular) == (0.0, 0.2, 0.1)
    assert (right.forward, right.lateral, right.angular) == (
        0.0,
        -0.2,
        -0.1,
    )


def test_lateral_direction_sign_can_be_calibrated_without_changing_turn():
    policy = ApproachPolicy(lateral_direction_sign=-1.0)
    policy.reset(0.0)

    decision = policy.decide(2.0, 0.3, math.radians(12.0), 0.1)

    assert decision.lateral == -0.2
    assert decision.angular == 0.1


def test_axes_stop_independently_inside_each_alignment_tolerance():
    policy = ApproachPolicy()
    policy.reset(0.0)

    lateral_only = policy.decide(2.0, 0.3, math.radians(2.0), 0.1)
    angular_only = policy.decide(2.0, 0.1, math.radians(10.0), 0.2)

    assert (lateral_only.lateral, lateral_only.angular) == (0.2, 0.0)
    assert (angular_only.lateral, angular_only.angular) == (0.0, 0.1)


def test_four_aligned_frames_emit_zero_before_straight_approach():
    policy = ApproachPolicy(align_stable_frames=4)
    policy.reset(0.0)

    aligned = make_aligned(policy)

    assert all(
        (decision.forward, decision.lateral, decision.angular)
        == (0.0, 0.0, 0.0)
        for decision in aligned
    )
    assert not any(decision.transition for decision in aligned[:3])
    assert aligned[3].transition == ApproachPolicy.APPROACH
    straight = policy.decide(2.0, 0.1, math.radians(2.0), 0.5)
    assert (straight.forward, straight.lateral, straight.angular) == (
        0.2,
        0.0,
        0.0,
    )


@pytest.mark.parametrize(
    "lateral,angle",
    [
        (0.46, math.radians(2.0)),
        (0.1, math.radians(14.1)),
    ],
)
def test_approach_drift_emits_zero_then_returns_to_align(lateral, angle):
    policy = ApproachPolicy()
    policy.reset(0.0)
    make_aligned(policy)

    decision = policy.decide(2.0, lateral, angle, 0.5)

    assert decision.transition == ApproachPolicy.ALIGN
    assert (decision.forward, decision.lateral, decision.angular) == (
        0.0,
        0.0,
        0.0,
    )
    assert policy.phase == ApproachPolicy.ALIGN


def test_large_target_angle_fails_closed():
    policy = ApproachPolicy(max_target_angle_deg=35.0)
    policy.reset(0.0)

    decision = policy.decide(2.0, 0.2, math.radians(35.1), 0.1)

    assert (decision.forward, decision.lateral, decision.angular) == (
        0.0,
        0.0,
        0.0,
    )
    assert "angle" in decision.fault


def test_arrival_requires_three_consecutive_frames_and_stays_zero():
    policy = ApproachPolicy(stop_distance_m=1.3, arrival_frames=3)
    policy.reset(0.0)

    first = policy.decide(1.3, 0.0, 0.0, 0.1)
    second = policy.decide(1.2, 0.0, 0.0, 0.2)
    third = policy.decide(1.1, 0.0, 0.0, 0.3)

    assert not first.arrived and first.forward == 0.0
    assert not second.arrived and second.forward == 0.0
    assert third.arrived and third.forward == 0.0


def test_arrival_counter_resets_when_target_moves_away():
    policy = ApproachPolicy(stop_distance_m=1.3, arrival_frames=3)
    policy.reset(0.0)
    policy.decide(1.2, 0.0, 0.0, 0.1)
    policy.decide(1.2, 0.0, 0.0, 0.2)

    moving = policy.decide(1.5, 0.3, math.radians(10.0), 0.3)
    after_reset = policy.decide(1.2, 0.0, 0.0, 0.4)

    assert moving.forward == 0.0
    assert not after_reset.arrived


def test_align_timeout_fails_closed():
    policy = ApproachPolicy(align_timeout_s=5.0)
    policy.reset(10.0)

    decision = policy.decide(2.0, 0.3, math.radians(10.0), 15.0)

    assert "ALIGN timeout" in decision.fault


def test_cumulative_approach_timeout_survives_realign():
    policy = ApproachPolicy(max_approach_duration_s=5.0)
    policy.reset(0.0)
    make_aligned(policy)
    assert policy.decide(2.0, 0.0, 0.0, 2.4).forward == 0.2
    assert policy.decide(2.0, 0.5, 0.0, 2.5).transition == policy.ALIGN
    make_aligned(policy, start_s=2.6)

    decision = policy.decide(2.0, 0.0, 0.0, 5.8)

    assert "cumulative APPROACH" in decision.fault


def test_restart_align_preserves_task_and_approach_timers():
    policy = ApproachPolicy(
        max_approach_duration_s=5.0,
        max_total_duration_s=8.0,
    )
    policy.reset(0.0)
    make_aligned(policy)
    assert policy.decide(2.0, 0.0, 0.0, 2.4).forward == 0.2

    policy.restart_align(2.5)
    make_aligned(policy, start_s=2.6)

    approach_timeout = policy.decide(2.0, 0.0, 0.0, 6.0)
    policy.restart_align(6.1)
    total_timeout = policy.decide(2.0, 0.0, 0.0, 8.0)

    assert "cumulative APPROACH" in approach_timeout.fault
    assert "total duration" in total_timeout.fault


def test_total_task_timeout_fails_closed():
    policy = ApproachPolicy(max_total_duration_s=8.0)
    policy.reset(2.0)

    decision = policy.decide(2.0, 0.3, 0.1, 10.0)

    assert "total duration" in decision.fault


@pytest.mark.parametrize(
    "distance,lateral,angle,now_s",
    [
        (0.0, 0.0, 0.0, 0.1),
        (-1.0, 0.0, 0.0, 0.1),
        (float("nan"), 0.0, 0.0, 0.1),
        (2.0, float("inf"), 0.0, 0.1),
        (2.0, 0.0, float("inf"), 0.1),
    ],
)
def test_invalid_target_values_fail_closed(distance, lateral, angle, now_s):
    policy = ApproachPolicy()
    policy.reset(0.0)

    decision = policy.decide(distance, lateral, angle, now_s)

    assert (decision.forward, decision.lateral, decision.angular) == (
        0.0,
        0.0,
        0.0,
    )
    assert decision.fault


def test_unsafe_configuration_is_rejected():
    with pytest.raises(ValueError):
        ApproachPolicy(forward_velocity=-0.2)
    with pytest.raises(ValueError):
        ApproachPolicy(align_lateral_velocity=0.1)
    with pytest.raises(ValueError):
        ApproachPolicy(align_angular_velocity=-0.1)
    with pytest.raises(ValueError):
        ApproachPolicy(lateral_direction_sign=0.0)
    with pytest.raises(ValueError):
        ApproachPolicy(align_angle_deg=15.0, realign_angle_deg=14.0)
    with pytest.raises(ValueError):
        ApproachPolicy(align_lateral_m=0.5, realign_lateral_m=0.45)


class _SilentLogger:
    def info(self, _message):
        pass


class _PreApproachHarness:
    _SNAPSHOT_TURN = ApproachControllerNode._SNAPSHOT_TURN
    _SETTLE = ApproachControllerNode._SETTLE
    _REACQUIRE = ApproachControllerNode._REACQUIRE
    _BLIND_APPROACH = ApproachControllerNode._BLIND_APPROACH
    _POLICY_STATES = ApproachControllerNode._POLICY_STATES

    _snapshot_turn_duration_s = (
        ApproachControllerNode._snapshot_turn_duration_s
    )
    _blind_approach_duration_s = (
        ApproachControllerNode._blind_approach_duration_s
    )
    _target_summary = ApproachControllerNode._target_summary
    _approach_snapshot_summary = (
        ApproachControllerNode._approach_snapshot_summary
    )

    def __init__(self):
        self._policy = ApproachPolicy()
        self._policy.reset(0.0)
        self._snapshot_turn_enabled = True
        self._snapshot_turn_min_s = 0.2
        self._snapshot_turn_max_s = 2.5
        self._snapshot_turn_direction_sign = 1.0
        self._settle_duration_s = 0.6
        self._reacquire_timeout_s = 3.0
        self._target_timeout_s = 0.4
        self._motion_enabled = False
        self._last_mode_check_s = 0.0
        self._snapshot_distance_m = 2.0
        self._snapshot_lateral_m = 0.3
        self._snapshot_angle_rad = math.radians(20.0)
        self._approach_snapshot_distance_m = 0.0
        self._approach_snapshot_lateral_m = 0.0
        self._approach_snapshot_angle_rad = 0.0
        self._blind_approach_started_s = 0.0
        self._blind_approach_deadline_s = 0.0
        self._blind_arrival_count = 0
        self._state = "IDLE"
        self._state_deadline_s = 0.0
        self._target = None
        self._target_received_s = 0.0
        self.commands = []
        self.stop_reasons = []

    def get_logger(self):
        return _SilentLogger()

    def _send_velocity(self, forward, lateral, angular):
        self.commands.append((forward, lateral, angular))

    def _stop_task(self, reason, fault=False):
        self.stop_reasons.append((reason, fault))

    def _query_mode(self, _kind):
        raise AssertionError("mode query is not expected in this test")


@pytest.mark.parametrize(
    "angle_rad,expected_s",
    [
        (0.001, 0.2),
        (0.1, 1.0),
        (1.0, 2.5),
    ],
)
def test_snapshot_turn_duration_is_clamped(angle_rad, expected_s):
    harness = _PreApproachHarness()

    duration_s = harness._snapshot_turn_duration_s(angle_rad)

    assert duration_s == pytest.approx(expected_s)


def test_small_snapshot_angle_bypasses_snapshot_turn():
    harness = _PreApproachHarness()
    harness._snapshot_angle_rad = math.radians(5.0)

    ApproachControllerNode._enter_first_control_state(harness, 10.0)

    assert harness._state == ApproachPolicy.ALIGN
    assert harness._state_deadline_s == 0.0


@pytest.mark.parametrize(
    "angle_rad,direction_sign,expected_angular",
    [
        (math.radians(20.0), 1.0, 0.1),
        (math.radians(-20.0), 1.0, -0.1),
        (math.radians(20.0), -1.0, -0.1),
    ],
)
def test_snapshot_turn_uses_only_signed_angular_velocity(
    angle_rad,
    direction_sign,
    expected_angular,
):
    harness = _PreApproachHarness()
    harness._snapshot_angle_rad = angle_rad
    harness._snapshot_turn_direction_sign = direction_sign
    ApproachControllerNode._enter_first_control_state(harness, 0.0)

    ApproachControllerNode._run_pre_approach_tick(harness, 0.1)

    assert harness._state == harness._SNAPSHOT_TURN
    assert harness.commands[-1] == (0.0, 0.0, expected_angular)


def test_snapshot_settle_reacquire_then_enters_align_with_zero_commands():
    harness = _PreApproachHarness()
    ApproachControllerNode._enter_first_control_state(harness, 0.0)
    snapshot_deadline_s = harness._state_deadline_s

    ApproachControllerNode._run_pre_approach_tick(
        harness,
        snapshot_deadline_s,
    )
    assert harness._state == harness._SETTLE
    assert harness.commands[-1] == (0.0, 0.0, 0.0)

    settle_deadline_s = harness._state_deadline_s
    ApproachControllerNode._run_pre_approach_tick(
        harness,
        settle_deadline_s,
    )
    assert harness._state == harness._REACQUIRE
    assert harness.commands[-1] == (0.0, 0.0, 0.0)

    harness._target = SimpleNamespace(
        distance_m=1.8,
        lateral_m=0.1,
        horizontal_error=math.radians(2.0),
    )
    harness._target_received_s = settle_deadline_s + 0.1
    ApproachControllerNode._run_pre_approach_tick(
        harness,
        settle_deadline_s + 0.1,
    )

    assert harness._state == ApproachPolicy.ALIGN
    assert harness.commands[-1] == (0.0, 0.0, 0.0)
    assert not harness.stop_reasons


def test_approach_snapshot_enters_blind_approach_and_ignores_lidar_loss():
    harness = _PreApproachHarness()
    harness._policy = ApproachPolicy(
        forward_velocity=0.2,
        stop_distance_m=1.3,
        max_approach_duration_s=35.0,
        max_total_duration_s=45.0,
    )
    harness._policy.reset(0.0)
    harness._target = SimpleNamespace(
        distance_m=5.3,
        lateral_m=0.2,
        horizontal_error=math.radians(2.0),
    )
    harness._target_received_s = 10.0

    ApproachControllerNode._enter_blind_approach(harness, 10.0)

    assert harness._state == harness._BLIND_APPROACH
    assert harness._approach_snapshot_distance_m == pytest.approx(5.3)
    assert harness._blind_approach_deadline_s == pytest.approx(30.0)

    harness._target = None
    ApproachControllerNode._run_blind_approach_tick(harness, 11.0)

    assert harness.commands[-1] == (0.2, 0.0, 0.0)
    assert not harness.stop_reasons


def test_blind_approach_stops_on_snapshot_time_or_live_stop_distance():
    timed = _PreApproachHarness()
    timed._policy = ApproachPolicy(
        forward_velocity=0.2,
        stop_distance_m=1.3,
        max_approach_duration_s=35.0,
        max_total_duration_s=45.0,
    )
    timed._policy.reset(0.0)
    timed._target = SimpleNamespace(
        distance_m=2.3,
        lateral_m=0.0,
        horizontal_error=0.0,
    )
    ApproachControllerNode._enter_blind_approach(timed, 10.0)
    ApproachControllerNode._run_blind_approach_tick(timed, 15.0)

    live_stop = _PreApproachHarness()
    live_stop._policy = ApproachPolicy(arrival_frames=3)
    live_stop._policy.reset(10.0)
    live_stop._state = live_stop._BLIND_APPROACH
    live_stop._blind_approach_started_s = 10.0
    live_stop._blind_approach_deadline_s = 20.0
    live_stop._target = SimpleNamespace(distance_m=1.2)
    live_stop._target_received_s = 11.0
    for now_s in (11.0, 11.1, 11.2):
        ApproachControllerNode._run_blind_approach_tick(live_stop, now_s)

    assert timed.stop_reasons == [
        ("approach snapshot travel time elapsed", False)
    ]
    assert live_stop.stop_reasons == [("stop distance confirmed", False)]


@pytest.mark.parametrize(
    "target,expected_reason",
    [
        (None, "cannot snapshot approach without target"),
        (
            SimpleNamespace(
                distance_m=-1.0,
                lateral_m=0.0,
                horizontal_error=0.0,
            ),
            "approach snapshot target is invalid",
        ),
        (
            SimpleNamespace(
                distance_m=2.0,
                lateral_m=math.nan,
                horizontal_error=0.0,
            ),
            "approach snapshot target is invalid",
        ),
        (
            SimpleNamespace(
                distance_m=2.0,
                lateral_m=0.0,
                horizontal_error=math.inf,
            ),
            "approach snapshot target is invalid",
        ),
    ],
)
def test_invalid_blind_approach_snapshot_fails_closed(
    target,
    expected_reason,
):
    harness = _PreApproachHarness()
    harness._target = target

    ApproachControllerNode._enter_blind_approach(harness, 10.0)

    assert harness.stop_reasons == [(expected_reason, True)]
    assert not harness.commands


def test_blind_approach_enforces_35_second_and_45_second_limits():
    duration_limit = _PreApproachHarness()
    duration_limit._policy = ApproachPolicy(
        max_approach_duration_s=35.0,
        max_total_duration_s=45.0,
    )
    duration_limit._policy.reset(10.0)
    duration_limit._state = duration_limit._BLIND_APPROACH
    duration_limit._blind_approach_started_s = 10.0
    duration_limit._blind_approach_deadline_s = 100.0

    ApproachControllerNode._run_blind_approach_tick(duration_limit, 45.0)

    assert duration_limit.stop_reasons == [
        ("maximum cumulative BLIND_APPROACH duration elapsed", True)
    ]

    total_limit = _PreApproachHarness()
    total_limit._policy = ApproachPolicy(
        max_approach_duration_s=35.0,
        max_total_duration_s=45.0,
    )
    total_limit._policy.reset(0.0)
    total_limit._state = total_limit._BLIND_APPROACH
    total_limit._blind_approach_started_s = 20.0
    total_limit._blind_approach_deadline_s = 100.0

    ApproachControllerNode._run_blind_approach_tick(total_limit, 45.0)

    assert total_limit.stop_reasons == [
        ("maximum total duration elapsed", True)
    ]


def test_reacquire_timeout_and_total_timeout_fail_closed():
    reacquire = _PreApproachHarness()
    reacquire._state = reacquire._REACQUIRE
    reacquire._state_deadline_s = 3.0
    ApproachControllerNode._run_pre_approach_tick(reacquire, 3.0)

    total = _PreApproachHarness()
    total._state = total._SNAPSHOT_TURN
    total._state_deadline_s = 20.0
    ApproachControllerNode._run_pre_approach_tick(total, 8.0)

    assert reacquire.commands[-1] == (0.0, 0.0, 0.0)
    assert reacquire.stop_reasons == [
        ("person target reacquire timed out", True)
    ]
    assert total.stop_reasons == [
        ("maximum total duration elapsed", True)
    ]


class _TargetCallbackHarness:
    _POLICY_STATES = ApproachControllerNode._POLICY_STATES
    _message_session_active = staticmethod(
        ApproachControllerNode._message_session_active
    )

    def __init__(self, state):
        self._state = state
        self._target = object()
        self._awaiting_release = True
        self.stop_reasons = []

    def _stop_task(self, reason, fault=False):
        self.stop_reasons.append((reason, fault))

    def _release_for_next_wave(self):
        raise AssertionError("release is not expected in this test")


def test_invalid_target_is_tolerated_before_align_but_stops_policy_states():
    invalid = SimpleNamespace(
        valid=False,
        gesture_confirmed=False,
        track_id=-1,
    )

    for state in (
        "VERIFY_MODE",
        "REGISTERING",
        ApproachControllerNode._SNAPSHOT_TURN,
        ApproachControllerNode._SETTLE,
        ApproachControllerNode._REACQUIRE,
        ApproachControllerNode._BLIND_APPROACH,
    ):
        harness = _TargetCallbackHarness(state)
        ApproachControllerNode._on_target(harness, invalid)
        assert harness._target is None
        assert not harness.stop_reasons

    for state in (ApproachPolicy.ALIGN, ApproachPolicy.APPROACH):
        harness = _TargetCallbackHarness(state)
        ApproachControllerNode._on_target(harness, invalid)
        assert harness._target is None
        assert harness.stop_reasons == [
            ("person target became invalid", True)
        ]

"""Tests for allow-listed real preset-motion sequencing."""

from emo_robot_agent.executor import SkillExecutor
from emo_robot_agent.plan import SkillPlan
from emo_robot_agent.planner import DemoPlanner
from emo_robot_agent.speech_feedback import RealSpeechFeedback
from emo_robot_agent.skill_catalog import SKILLS_BY_NAME
from emo_robot_agent.skills import (
    IntroducePlaceholderSkill,
    PresetMotionSkill,
    SkillRegistry,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class _Timer:
    def __init__(self, callback):
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback()


class _Node:
    def __init__(self):
        self.timers = []

    def create_timer(self, _period_s, callback):
        timer = _Timer(callback)
        self.timers.append(timer)
        return timer

    def destroy_timer(self, _timer):
        return True


class _State:
    action = 200
    status = 100
    description = "STAND_DEFAULT"

    def __init__(self, running=True):
        self.running = running

    def is_running(self, expected_action):
        return self.running and expected_action == 200


class _StateClient:
    def __init__(self, success=True, running=True):
        self.success = success
        self.running = running
        self.calls = 0

    def get_state_async(self, callback):
        self.calls += 1
        if self.success:
            callback(True, _State(self.running), "")
        else:
            callback(False, None, "state query failed")


class _MotionClient:
    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def play_async(
        self,
        area,
        motion_id,
        interrupt=False,
        done_callback=None,
    ):
        self.calls.append((area, motion_id, interrupt))
        done_callback(self.success)


class _SpeechClient:
    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def say_async(self, text, domain, done_callback):
        self.calls.append((text, domain))
        done_callback(self.success)


def _skill(name, node, motion, state, logger=None):
    return PresetMotionSkill(
        node=node,
        logger=logger or _Logger(),
        spec=SKILLS_BY_NAME[name],
        motion_client=motion,
        state_client=state,
        stand_action=200,
        wait_s=0.1,
    )


def test_sdk_motion_mapping_is_allow_listed():
    expected = {
        "wave_right": (2, 1002),
        "wave_left": (1, 1002),
        "handshake_right": (2, 1003),
        "handshake_left": (1, 1003),
        "raise_hand_right": (2, 1001),
        "raise_hand_left": (1, 1001),
        "flying_kiss_right": (2, 1004),
        "flying_kiss_left": (1, 1004),
        "bow": (11, 3001),
        "clap": (11, 3017),
        "hug": (11, 3008),
        "salute_right": (2, 1013),
        "salute_left": (1, 1013),
        "heart_both": (3, 1007),
        "heart_right": (2, 1007),
        "heart_left": (1, 1007),
        "cheer": (11, 3011),
        "flat_raise_both": (3, 1010),
        "flat_raise_right": (2, 1010),
        "flat_raise_left": (1, 1010),
        "bye": (11, 3031),
        "dynamic_wave": (11, 3007),
        "high_five_right": (2, 1008),
        "high_five_left": (1, 1008),
        "cross_arms": (11, 3009),
        "chest_wave_right": (2, 1011),
        "chest_wave_left": (1, 1011),
        "scratch_head": (11, 3024),
        "scratch_butt": (11, 3025),
    }

    assert {
        name: (SKILLS_BY_NAME[name].area, SKILLS_BY_NAME[name].motion_id)
        for name in expected
    } == expected


def test_motion_disabled_is_a_dry_run_without_sdk_clients():
    results = []
    skill = PresetMotionSkill(
        node=_Node(),
        logger=_Logger(),
        spec=SKILLS_BY_NAME["wave_right"],
        motion_client=None,
        state_client=None,
        stand_action=200,
    )

    skill.execute({}, lambda success, detail: results.append((success, detail)))

    assert results == [
        (
            True,
            "wave_right dry-run: motion_enabled=false area=2 motion=1002",
        )
    ]


def test_non_stand_mode_rejects_motion_request():
    motion = _MotionClient()
    results = []

    _skill("wave_right", _Node(), motion, _StateClient(running=False)).execute(
        {},
        lambda success, detail: results.append((success, detail)),
    )

    assert motion.calls == []
    assert results[0][0] is False
    assert "STAND_DEFAULT/RUNNING" in results[0][1]


def test_service_failure_finishes_skill_without_wait_timer():
    node = _Node()
    results = []

    _skill(
        "wave_right",
        node,
        _MotionClient(success=False),
        _StateClient(),
    ).execute({}, lambda success, detail: results.append((success, detail)))

    assert node.timers == []
    assert results == [(False, "wave_right preset request failed")]


def test_wave_bow_clap_waits_and_runs_strictly_in_order():
    node = _Node()
    logger = _Logger()
    motion = _MotionClient()
    state = _StateClient()
    registry = SkillRegistry()
    for name in ("wave_right", "bow", "clap"):
        registry.register(name, _skill(name, node, motion, state, logger))

    statuses = []
    spoken = []
    finished = []
    executor = SkillExecutor(
        registry,
        lambda skill, status, detail: statuses.append((skill, status, detail)),
        lambda text, done: (spoken.append(text), done(True, "speech accepted")),
    )
    executor.execute(
        SkillPlan.from_dict(
            {
                "tasks": ["wave_right", "bow", "clap"],
                "speech_text": "动作完成。",
            }
        ),
        lambda success, detail: finished.append((success, detail)),
    )

    assert motion.calls == [(2, 1002, False)]
    assert spoken == []
    node.timers.pop(0).fire()
    assert motion.calls == [(2, 1002, False), (11, 3001, False)]
    node.timers.pop(0).fire()
    assert motion.calls == [
        (2, 1002, False),
        (11, 3001, False),
        (11, 3017, False),
    ]
    assert spoken == []
    node.timers.pop(0).fire()

    assert state.calls == 3
    assert spoken == ["动作完成。"]
    assert finished == [(True, "all skills and speech completed")]
    assert ("wave_right", "succeeded") in [item[:2] for item in statuses]
    assert ("bow", "succeeded") in [item[:2] for item in statuses]
    assert ("clap", "succeeded") in [item[:2] for item in statuses]


def test_deprecated_ambiguous_skill_names_are_not_allow_listed():
    for name in ("wave", "handshake", "salute", "heart", "raise_hand"):
        assert name not in SKILLS_BY_NAME


def test_demo_planner_uses_defaults_and_disambiguates_directions():
    planner = DemoPlanner()

    assert [
        task.skill for task in planner.plan("挥手、鞠躬并鼓掌").tasks
    ] == ["wave_right", "bow", "clap"]
    assert [task.skill for task in planner.plan("左手挥手").tasks] == [
        "wave_left"
    ]
    assert [task.skill for task in planner.plan("左手挥手、右手挥手").tasks] == [
        "wave_left",
        "wave_right",
    ]
    assert [task.skill for task in planner.plan("比心").tasks] == [
        "heart_both"
    ]
    assert [task.skill for task in planner.plan("平举").tasks] == [
        "flat_raise_both"
    ]


def test_demo_planner_separates_applause_from_high_five():
    planner = DemoPlanner()

    assert [task.skill for task in planner.plan("请鼓掌").tasks] == ["clap"]
    assert [task.skill for task in planner.plan("请左手击掌").tasks] == [
        "high_five_left"
    ]


def test_introduce_uses_only_plan_speech_text_once():
    registry = SkillRegistry()
    registry.register("introduce", IntroducePlaceholderSkill(_Logger()))
    speech_client = _SpeechClient()
    feedback = RealSpeechFeedback(speech_client)
    finished = []
    executor = SkillExecutor(
        registry,
        speech_callback=feedback.execute,
    )
    executor.execute(
        SkillPlan.from_dict(
            {
                "tasks": ["introduce"],
                "speech_text": "你好，我是小智。",
            }
        ),
        lambda success, detail: finished.append((success, detail)),
    )

    assert speech_client.calls == [("你好，我是小智。", "emo_robot")]
    assert finished == [(True, "all skills and speech completed")]


def test_real_tts_failure_fails_plan():
    speech_client = _SpeechClient(success=False)
    feedback = RealSpeechFeedback(speech_client)
    finished = []
    executor = SkillExecutor(
        SkillRegistry(),
        speech_callback=feedback.execute,
    )
    executor.execute(
        SkillPlan.from_dict({"tasks": [], "speech_text": "测试"}),
        lambda success, detail: finished.append((success, detail)),
    )

    assert finished == [(False, "plan speech TTS request failed")]


def test_empty_plan_without_speech_is_rejected():
    try:
        SkillPlan.from_dict({"tasks": [], "speech_text": ""})
    except ValueError as exc:
        assert "task or speech_text" in str(exc)
    else:
        raise AssertionError("empty plan should have been rejected")

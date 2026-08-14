"""Planner-visible skill metadata based on the AimDK preset-motion table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SkillSpec:
    """Static metadata shared by planning and skill registration."""

    name: str
    description: str
    area: int
    motion_id: int
    wait_s: float = 0.0


SKILL_CATALOG: Tuple[SkillSpec, ...] = (
    SkillSpec("approach", "接近用户；用于邀请机器人来到面前。", 0, 0),
    SkillSpec("wave_right", "右手挥手；未指定左右手的挥手默认选此技能。", 2, 1002, 4.0),
    SkillSpec("wave_left", "左手挥手；仅在用户明确要求左手时选择。", 1, 1002, 4.0),
    SkillSpec("handshake_right", "右手握手；未指定左右手的握手默认选此技能。", 2, 1003, 6.0),
    SkillSpec("handshake_left", "左手握手；仅在用户明确要求左手时选择。", 1, 1003, 6.0),
    SkillSpec("raise_hand_right", "举起右手；未指定左右手的举手默认选此技能。", 2, 1001, 4.0),
    SkillSpec("raise_hand_left", "举起左手；仅在用户明确要求左手时选择。", 1, 1001, 4.0),
    SkillSpec("flying_kiss_right", "右手飞吻；未指定左右手的飞吻默认选此技能。", 2, 1004, 4.0),
    SkillSpec("flying_kiss_left", "左手飞吻；仅在用户明确要求左手时选择。", 1, 1004, 4.0),
    SkillSpec("clap", "双手鼓掌表达祝贺和赞赏；不要与单手击掌混淆。", 11, 3017, 5.0),
    SkillSpec("salute_right", "右手敬礼；未指定左右手的敬礼默认选此技能。", 2, 1013, 4.0),
    SkillSpec("salute_left", "左手敬礼；仅在用户明确要求左手时选择。", 1, 1013, 4.0),
    SkillSpec("heart_both", "双手比心；未指定单手时，比心默认选此技能。", 3, 1007, 5.0),
    SkillSpec("heart_right", "右手比心；仅在用户明确要求右手时选择。", 2, 1007, 5.0),
    SkillSpec("heart_left", "左手比心；仅在用户明确要求左手时选择。", 1, 1007, 5.0),
    SkillSpec("hug", "拥抱动作；用于安慰、鼓励和表达关心。", 11, 3008, 6.0),
    SkillSpec("cheer", "加油动作；用于鼓励、庆祝和提升情绪。", 11, 3011, 5.0),
    SkillSpec("flat_raise_both", "双手平举；未指定单手时，平举默认选此技能。", 3, 1010, 4.0),
    SkillSpec("flat_raise_right", "右手平举；仅在用户明确要求右手时选择。", 2, 1010, 4.0),
    SkillSpec("flat_raise_left", "左手平举；仅在用户明确要求左手时选择。", 1, 1010, 4.0),
    SkillSpec("bye", "拜拜动作；用于明确告别、再见和送别。", 11, 3031, 5.0),
    SkillSpec("dynamic_wave", "动感光波动作；用于明确要求动感光波。", 11, 3007, 5.0),
    SkillSpec("high_five_right", "右手击掌；未指定左右手的击掌默认选此技能。", 2, 1008, 5.0),
    SkillSpec("high_five_left", "左手击掌；仅在用户明确要求左手时选择。", 1, 1008, 5.0),
    SkillSpec("cross_arms", "双手在胸前打叉，表达拒绝或停止。", 11, 3009, 5.0),
    SkillSpec("chest_wave_right", "胸前右手挥手；用于明确要求胸前右手挥手。", 2, 1011, 4.0),
    SkillSpec("chest_wave_left", "胸前左手挥手；用于明确要求胸前左手挥手。", 1, 1011, 4.0),
    SkillSpec("bow", "鞠躬表达礼貌、欢迎、感谢或道歉。", 11, 3001, 5.0),
    SkillSpec("scratch_head", "挠头动作；用于困惑、思考或不好意思。", 11, 3024, 5.0),
    SkillSpec("scratch_butt", "抓屁股动作；仅在用户明确要求时选择。", 11, 3025, 5.0),
    SkillSpec("introduce", "介绍机器人自己；用于自我介绍。", 0, 0),
)

SKILLS_BY_NAME: Dict[str, SkillSpec] = {
    skill.name: skill for skill in SKILL_CATALOG
}
PLANNER_SKILL_NAMES = tuple(skill.name for skill in SKILL_CATALOG)
MOTION_SKILLS = tuple(
    skill for skill in SKILL_CATALOG if skill.area and skill.motion_id
)

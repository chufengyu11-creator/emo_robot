"""Planner backends, TCP transport, and LLM output validation."""

from __future__ import annotations

import json
import re
import socket
import time
from typing import Callable, Protocol
import uuid

from emo_robot_agent.plan import ALLOWED_SKILLS, SkillPlan
from emo_robot_agent.skill_catalog import SKILL_CATALOG


_PROMPT_SKILLS = "\n\n".join(
    f"{skill.name}:\n{skill.description}" for skill in SKILL_CATALOG
)
PROMPT_TEMPLATE = """你是机器人任务规划器。

你的任务是根据用户意图、情绪和交互场景，选择最合适的机器人技能。
不要只根据用户是否说出动作关键词来选择技能。

你需要理解：
- 用户当前状态
- 用户情绪
- 用户交互目的
- 当前场景

可以选择一个或多个技能，并按照合理的执行顺序排列。
同时根据用户输入、用户情绪和所选技能，生成简短、自然、温暖的中文回复。
回复放入 speech_text，不要在 speech_text 中解释规划过程。

只能从以下技能选择：

{skills}

示例：

用户：
"今天不开心"

输出：
{{
 "tasks":[
   "hug"
 ],
 "speech_text":"别难过，我陪着你。"
}}

用户：
"我最近压力很大"

输出：
{{
 "tasks":[
   "cheer"
 ],
 "speech_text":"辛苦了，加油，你一定可以。"
}}

用户：
"欢迎一下新朋友"

输出：
{{
 "tasks":[
   "wave_right",
   "introduce"
 ],
 "speech_text":"欢迎新朋友，我是小智。"
}}

用户：
"谢谢你的帮助"

输出：
{{
 "tasks":[
   "bow"
 ],
 "speech_text":"不用客气，很高兴帮到你。"
}}


用户指令:
{text}

只输出JSON:

{{
"tasks":[],
"speech_text":""
}}

禁止输出解释。""".replace("{skills}", _PROMPT_SKILLS)


class PlannerBackend(Protocol):
    """Backend contract used by the ROS planner node."""

    def plan(self, text: str) -> SkillPlan:
        """Return a validated plan for user text."""


def parse_llm_plan(text: str) -> SkillPlan:
    """Extract and validate a plan from a JSON-only LLM response."""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1)
    return SkillPlan.from_json(cleaned, ALLOWED_SKILLS)


class DemoPlanner:
    """Deterministic offline backend for the first-phase demo."""

    _PHRASES = {
        "approach": (
            "走到我面前",
            "走过来",
            "靠近我",
            "过来",
            "approach",
        ),
        "wave_right": (
            "右手挥手", "右手招手", "挥手", "招手", "欢迎", "wave"
        ),
        "wave_left": ("左手挥手", "左手招手", "left wave"),
        "handshake_right": ("右手握手", "握手", "handshake"),
        "handshake_left": ("左手握手", "left handshake"),
        "raise_hand_right": (
            "右手举手", "右手抬手", "举手", "抬手", "raise hand"
        ),
        "raise_hand_left": ("左手举手", "左手抬手", "left raise hand"),
        "flying_kiss_right": ("右手飞吻", "飞吻", "flying kiss"),
        "flying_kiss_left": ("左手飞吻", "left flying kiss"),
        "bow": ("鞠躬", "感谢", "谢谢", "道歉", "bow"),
        "clap": ("鼓掌", "拍手", "clap"),
        "hug": (
            "拥抱",
            "抱一下",
            "安慰",
            "不开心",
            "不高兴",
            "难过",
            "情绪低落",
            "疲惫",
            "有点累",
            "hug",
        ),
        "salute_right": ("右手敬礼", "敬礼", "salute"),
        "salute_left": ("左手敬礼", "left salute"),
        "heart_both": ("双手比心", "比心", "爱心", "heart"),
        "heart_right": ("右手比心", "right heart"),
        "heart_left": ("左手比心", "left heart"),
        "cheer": (
            "加油",
            "欢呼",
            "庆祝",
            "鼓励",
            "压力很大",
            "压力大",
            "失落",
            "提升情绪",
            "cheer",
        ),
        "flat_raise_both": ("双手平举", "平举", "flat raise"),
        "flat_raise_right": ("右手平举", "right flat raise"),
        "flat_raise_left": ("左手平举", "left flat raise"),
        "bye": ("挥手告别", "告别", "再见", "拜拜", "goodbye"),
        "dynamic_wave": ("动感光波", "dynamic wave"),
        "high_five_right": ("右手击掌", "击掌", "high five"),
        "high_five_left": ("左手击掌", "left high five"),
        "cross_arms": ("双手打叉", "胸前打叉", "打叉", "cross arms"),
        "chest_wave_right": ("胸前右手挥手", "right chest wave"),
        "chest_wave_left": ("胸前左手挥手", "left chest wave"),
        "scratch_head": ("挠头", "scratch head"),
        "scratch_butt": ("抓屁股", "scratch butt"),
        "introduce": (
            "介绍自己",
            "自我介绍",
            "新朋友",
            "客人",
            "introduce yourself",
        ),
    }

    _SPEECH_BY_SKILL = {
        "approach": "好的，我来靠近你。",
        "wave_right": "你好，很高兴见到你。",
        "wave_left": "你好，我用左手向你打招呼。",
        "handshake_right": "你好，很高兴认识你。",
        "handshake_left": "好的，我用左手和你握手。",
        "raise_hand_right": "好的，我举起右手示意。",
        "raise_hand_left": "好的，我举起左手示意。",
        "flying_kiss_right": "送你一个飞吻。",
        "flying_kiss_left": "送你一个左手飞吻。",
        "bow": "不用客气，很高兴帮到你。",
        "clap": "真棒，为你鼓掌！",
        "hug": "别难过，我陪着你。",
        "salute_right": "向你敬礼！",
        "salute_left": "我用左手向你敬礼！",
        "heart_both": "送你一颗大大的爱心。",
        "heart_right": "送你一颗右手爱心。",
        "heart_left": "送你一颗左手爱心。",
        "cheer": "辛苦了，加油，你一定可以。",
        "flat_raise_both": "好的，我把双手平举。",
        "flat_raise_right": "好的，我把右手平举。",
        "flat_raise_left": "好的，我把左手平举。",
        "bye": "再见，期待下次见面。",
        "dynamic_wave": "看我的动感光波！",
        "high_five_right": "来，击个掌！",
        "high_five_left": "来，用左手击个掌！",
        "cross_arms": "好的，我用双手打叉。",
        "chest_wave_right": "你好，我在胸前挥动右手。",
        "chest_wave_left": "你好，我在胸前挥动左手。",
        "scratch_head": "让我挠挠头想一想。",
        "scratch_butt": "好的，执行抓屁股动作。",
        "introduce": "你好，我是小智，很高兴认识你。",
    }

    def plan(self, text: str) -> SkillPlan:
        source = text.strip().lower()
        if not source:
            raise ValueError("user text must not be empty")

        candidates = []
        for skill, phrases in self._PHRASES.items():
            for phrase in phrases:
                for match in re.finditer(re.escape(phrase), source):
                    candidates.append(
                        (match.start(), match.end(), skill, phrase)
                    )
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))

        selected = []
        selected_skills = set()
        for candidate in candidates:
            start, end, skill, _phrase = candidate
            overlaps = any(
                start < chosen_end and end > chosen_start
                for chosen_start, chosen_end, _skill, _text in selected
            )
            if overlaps or skill in selected_skills:
                continue
            selected.append(candidate)
            selected_skills.add(skill)

        selected.sort(key=lambda item: item[0])
        if not selected:
            raise ValueError("no supported skill found in user text")
        tasks = [item[2] for item in selected]
        if "wave_right" in tasks and "introduce" in tasks:
            speech_text = "欢迎新朋友，我是小智。"
        else:
            speech_text = self._SPEECH_BY_SKILL[tasks[0]]
        return SkillPlan.from_dict(
            {"tasks": tasks, "speech_text": speech_text}
        )


class TcpJsonPromptClient:
    """Send one JSON prompt request and receive one JSON response over TCP."""

    _RESPONSE_KEYS = ("response", "text", "content", "result")

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 15.0,
        max_response_bytes: int = 1024 * 1024,
        connector: Callable = socket.create_connection,
    ) -> None:
        if not host:
            raise ValueError("LLM host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("LLM port must be in the range 1..65535")
        if timeout_s <= 0.0:
            raise ValueError("LLM timeout_s must be greater than zero")
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._max_response_bytes = max_response_bytes
        self._connector = connector

    def complete(self, prompt: str) -> str:
        """Return model text from the TCP JSON response."""
        request_id = str(uuid.uuid4())
        payload = (
            json.dumps(
                {
                    "type": "prompt_request",
                    "request_id": request_id,
                    "prompt": prompt,
                    "sent_at": time.time(),
                    "source": "robot_prompt_client",
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        with self._connector(
            (self._host, self._port),
            self._timeout_s,
        ) as connection:
            connection.settimeout(self._timeout_s)
            connection.sendall(payload)
            with connection.makefile("r", encoding="utf-8") as stream:
                response_line = stream.readline(self._max_response_bytes + 1)

        if not response_line:
            raise ValueError("LLM relay closed without a response")
        if len(response_line.encode("utf-8")) > self._max_response_bytes:
            raise ValueError("LLM response is too large")
        response = json.loads(response_line)
        if isinstance(response, dict):
            response_request_id = response.get("request_id")
            if response_request_id and response_request_id != request_id:
                raise ValueError("LLM response request_id does not match")
            if response.get("ok") is False:
                detail = response.get("error", "unknown relay error")
                raise ValueError(f"LLM relay failed: {detail}")
        if isinstance(response, dict) and "tasks" in response:
            return json.dumps(response, ensure_ascii=False)
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content
            for key in self._RESPONSE_KEYS:
                if key not in response:
                    continue
                value = response[key]
                if isinstance(value, str):
                    return value
                return json.dumps(value, ensure_ascii=False)
        raise ValueError("LLM response has no supported text field")


class TcpLlmPlanner:
    """Build the constrained Chinese prompt and validate the LLM plan."""

    def __init__(self, client: TcpJsonPromptClient) -> None:
        self._client = client

    def plan(self, text: str) -> SkillPlan:
        source = text.strip()
        if not source:
            raise ValueError("user text must not be empty")
        response_text = self._client.complete(
            PROMPT_TEMPLATE.format(text=source)
        )
        # parse_llm_plan performs JSON parsing and skill allow-list validation.
        plan = parse_llm_plan(response_text)
        if not plan.speech_text:
            raise ValueError("LLM plan must include non-empty speech_text")
        return plan

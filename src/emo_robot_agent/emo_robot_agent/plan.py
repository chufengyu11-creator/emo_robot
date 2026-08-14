"""Small JSON plan model shared by the planner and executor."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List

from emo_robot_agent.skill_catalog import PLANNER_SKILL_NAMES


ALLOWED_SKILLS = PLANNER_SKILL_NAMES


@dataclass(frozen=True)
class SkillTask:
    """One registry skill invocation."""

    skill: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "SkillTask":
        if isinstance(value, str):
            return cls(skill=value.strip())
        if isinstance(value, dict):
            skill = value.get("skill", value.get("name", ""))
            parameters = value.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError("task parameters must be a JSON object")
            return cls(skill=str(skill).strip(), parameters=parameters)
        raise ValueError("each task must be a skill name or object")


@dataclass(frozen=True)
class SkillPlan:
    """Validated sequential plan."""

    tasks: List[SkillTask]
    speech_text: str = ""

    @classmethod
    def from_dict(
        cls,
        value: Dict[str, Any],
        allowed_skills: Iterable[str] = ALLOWED_SKILLS,
    ) -> "SkillPlan":
        raw_tasks = value.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("plan.tasks must be a list")
        if len(raw_tasks) > 10:
            raise ValueError("plan contains too many tasks (maximum 10)")

        allowed = set(allowed_skills)
        tasks = [SkillTask.from_value(item) for item in raw_tasks]
        for task in tasks:
            if task.skill not in allowed:
                raise ValueError(f"unknown skill: {task.skill!r}")
        speech_text = value.get("speech_text", "")
        if not isinstance(speech_text, str):
            raise ValueError("plan.speech_text must be a string")
        speech_text = speech_text.strip()
        if not tasks and not speech_text:
            raise ValueError("plan must contain a task or speech_text")
        return cls(tasks=tasks, speech_text=speech_text)

    @classmethod
    def from_json(
        cls,
        text: str,
        allowed_skills: Iterable[str] = ALLOWED_SKILLS,
    ) -> "SkillPlan":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("plan must be a JSON object")
        return cls.from_dict(value, allowed_skills)

    def to_dict(self) -> Dict[str, Any]:
        values = []
        for task in self.tasks:
            if task.parameters:
                values.append(
                    {"skill": task.skill, "parameters": task.parameters}
                )
            else:
                values.append(task.skill)
        return {"tasks": values, "speech_text": self.speech_text}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

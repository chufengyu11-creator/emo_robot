"""Small allow-listed skill registry."""

from __future__ import annotations

from typing import Dict, Iterable

from emo_robot_agent.skills.base import Skill


class SkillRegistry:
    """Map planner-visible names to trusted skill adapters."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    @property
    def names(self) -> Iterable[str]:
        return tuple(self._skills)

    def register(self, name: str, skill: Skill) -> None:
        if not name:
            raise ValueError("skill name must not be empty")
        if name in self._skills:
            raise ValueError(f"skill already registered: {name}")
        self._skills[name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"skill is not registered: {name}") from exc

"""Simulation-only skills that never call robot or TTS interfaces."""

from __future__ import annotations

from typing import Any, Dict

from emo_robot_agent.skill_catalog import SkillSpec
from emo_robot_agent.skills.base import SkillDone


class SimulationApproachSkill:
    """Log the deferred approach behavior and complete immediately."""

    def __init__(self, logger) -> None:
        self._logger = logger

    def execute(
        self,
        parameters: Dict[str, Any],
        done_callback: SkillDone,
    ) -> None:
        del parameters
        detail = (
            "SIMULATION skill=approach placeholder="
            "target binding is not implemented"
        )
        self._logger.info(detail)
        done_callback(True, detail)


class SimulationMotionSkill:
    """Log one catalogued preset motion without calling AimDK."""

    def __init__(self, logger, spec: SkillSpec) -> None:
        self._logger = logger
        self._spec = spec

    def execute(
        self,
        parameters: Dict[str, Any],
        done_callback: SkillDone,
    ) -> None:
        del parameters
        detail = (
            f"SIMULATION skill={self._spec.name} "
            f"action={self._spec.name} "
            f"description={self._spec.description} "
            f"area={self._spec.area} motion={self._spec.motion_id}"
        )
        self._logger.info(detail)
        done_callback(True, detail)

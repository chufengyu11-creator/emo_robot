"""Sequential executor independent from ROS transport details."""

from __future__ import annotations

import json
from typing import Callable, Optional

from emo_robot_agent.plan import SkillPlan
from emo_robot_agent.skills.registry import SkillRegistry


StatusCallback = Callable[[str, str, str], None]
PlanDone = Callable[[bool, str], None]
SpeechCallback = Callable[[str, Callable[[bool, str], None]], None]


class SkillExecutor:
    """Execute one validated plan at a time in list order."""

    def __init__(
        self,
        registry: SkillRegistry,
        status_callback: Optional[StatusCallback] = None,
        speech_callback: Optional[SpeechCallback] = None,
    ) -> None:
        self._registry = registry
        self._status_callback = status_callback
        self._speech_callback = speech_callback
        self._plan: Optional[SkillPlan] = None
        self._index = 0
        self._speech_started = False
        self._done_callback: Optional[PlanDone] = None

    @property
    def busy(self) -> bool:
        return self._plan is not None

    def execute(self, plan: SkillPlan, done_callback: PlanDone) -> None:
        if self.busy:
            done_callback(False, "executor is busy")
            return
        self._plan = plan
        self._index = 0
        self._speech_started = False
        self._done_callback = done_callback
        self._run_next()

    def _emit(self, skill: str, state: str, detail: str) -> None:
        if self._status_callback:
            self._status_callback(skill, state, detail)

    def _run_next(self) -> None:
        assert self._plan is not None
        if self._index >= len(self._plan.tasks):
            self._run_speech_or_finish()
            return

        task = self._plan.tasks[self._index]
        try:
            skill = self._registry.get(task.skill)
        except KeyError as exc:
            self._finish(False, str(exc))
            return

        task_detail = (
            f"task {self._index + 1}/{len(self._plan.tasks)} "
            f"parameters={json.dumps(task.parameters, ensure_ascii=False)}"
        )
        self._emit(task.skill, "running", task_detail)

        def _on_done(success: bool, detail: str) -> None:
            if self._plan is None:
                return
            self._emit(
                task.skill,
                "succeeded" if success else "failed",
                detail,
            )
            if not success:
                self._finish(False, detail)
                return
            self._index += 1
            self._run_next()

        try:
            skill.execute(task.parameters, _on_done)
        except Exception as exc:
            self._emit(task.skill, "failed", str(exc))
            self._finish(False, f"{task.skill} raised: {exc}")

    def _run_speech_or_finish(self) -> None:
        assert self._plan is not None
        text = self._plan.speech_text
        if not text:
            self._finish(True, "all skills completed")
            return
        if self._speech_started:
            return
        if self._speech_callback is None:
            self._finish(False, "speech feedback handler is unavailable")
            return

        self._speech_started = True
        self._emit("speech", "running", f"speech_text={text}")

        def _on_speech_done(success: bool, detail: str) -> None:
            if self._plan is None:
                return
            self._emit(
                "speech",
                "succeeded" if success else "failed",
                detail,
            )
            if success:
                self._finish(True, "all skills and speech completed")
            else:
                self._finish(False, detail)

        try:
            self._speech_callback(text, _on_speech_done)
        except Exception as exc:
            self._emit("speech", "failed", str(exc))
            self._finish(False, f"speech feedback raised: {exc}")

    def _finish(self, success: bool, detail: str) -> None:
        callback = self._done_callback
        self._plan = None
        self._done_callback = None
        if callback:
            callback(success, detail)

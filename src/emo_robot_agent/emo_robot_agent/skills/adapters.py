"""Adapters from registry skills to existing emo_robot clients."""

from __future__ import annotations

from typing import Any, Dict

from emo_robot_agent.skill_catalog import SkillSpec
from emo_robot_agent.skills.base import SkillDone


class ApproachPlaceholderSkill:
    """First-phase placeholder until language target binding is designed."""

    def __init__(self, logger) -> None:
        self._logger = logger

    def execute(
        self,
        parameters: Dict[str, Any],
        done_callback: SkillDone,
    ) -> None:
        del parameters
        message = "approach deferred: no language target binding in phase 1"
        self._logger.warning(message)
        done_callback(True, message)


class PresetMotionSkill:
    """Run one allow-listed AimDK preset motion after a stand-mode check."""

    def __init__(
        self,
        node,
        logger,
        spec: SkillSpec,
        motion_client=None,
        state_client=None,
        stand_action: int = 0,
        wait_s: float | None = None,
    ) -> None:
        resolved_wait_s = spec.wait_s if wait_s is None else float(wait_s)
        if resolved_wait_s <= 0.0:
            raise ValueError("preset motion wait_s must be greater than zero")
        self._node = node
        self._logger = logger
        self._spec = spec
        self._motion_client = motion_client
        self._state_client = state_client
        self._stand_action = int(stand_action)
        self._wait_s = resolved_wait_s

    def execute(
        self,
        parameters: Dict[str, Any],
        done_callback: SkillDone,
    ) -> None:
        del parameters
        if self._motion_client is None:
            message = (
                f"{self._spec.name} dry-run: motion_enabled=false "
                f"area={self._spec.area} motion={self._spec.motion_id}"
            )
            self._logger.warning(message)
            done_callback(True, message)
            return
        if self._state_client is None:
            done_callback(False, "motion state client is unavailable")
            return

        def _on_state(success, state, error: str) -> None:
            if not success or state is None:
                done_callback(
                    False,
                    error or "failed to query robot motion state",
                )
                return
            if not state.is_running(self._stand_action):
                done_callback(
                    False,
                    "preset motion requires STAND_DEFAULT/RUNNING: "
                    f"action={state.action} status={state.status} "
                    f"description={state.description}",
                )
                return
            self._request_motion(done_callback)

        self._state_client.get_state_async(_on_state)

    def _request_motion(self, done_callback: SkillDone) -> None:
        """Submit the preset request and wait before completing the skill."""

        def _done(success: bool) -> None:
            if not success:
                done_callback(
                    False,
                    f"{self._spec.name} preset request failed",
                )
                return

            timer_holder = {"timer": None, "finished": False}

            def _after_wait() -> None:
                if timer_holder["finished"]:
                    return
                timer_holder["finished"] = True
                timer = timer_holder["timer"]
                if timer is not None:
                    timer.cancel()
                    try:
                        self._node.destroy_timer(timer)
                    except Exception:
                        pass
                done_callback(
                    True,
                    f"{self._spec.name} preset accepted; "
                    f"waited {self._wait_s:.1f}s",
                )

            timer_holder["timer"] = self._node.create_timer(
                self._wait_s,
                _after_wait,
            )
            self._logger.info(
                f"Preset motion accepted: skill={self._spec.name} "
                f"area={self._spec.area} motion={self._spec.motion_id}; "
                f"waiting {self._wait_s:.1f}s before the next skill."
            )

        self._motion_client.play_async(
            area=self._spec.area,
            motion_id=self._spec.motion_id,
            interrupt=False,
            done_callback=_done,
        )


class IntroducePlaceholderSkill:
    """Represent introduction intent; plan speech performs the only TTS."""

    def __init__(self, logger) -> None:
        self._logger = logger

    def execute(
        self,
        parameters: Dict[str, Any],
        done_callback: SkillDone,
    ) -> None:
        del parameters
        detail = "introduce intent accepted; speech_text handles TTS"
        self._logger.info(detail)
        done_callback(True, detail)

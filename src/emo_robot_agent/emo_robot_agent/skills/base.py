"""Common skill adapter protocol."""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol


SkillDone = Callable[[bool, str], None]


class Skill(Protocol):
    """Minimal asynchronous skill contract."""

    def execute(
        self,
        parameters: Dict[str, Any],
        done_callback: SkillDone,
    ) -> None:
        """Start the skill and invoke done_callback exactly once."""

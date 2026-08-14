"""Thread-safe Agent execution guard for microphone acceptance."""

from __future__ import annotations
import threading
import time
from typing import Callable


class AgentGuard:
    """Block microphone input during planning, execution, and cooldown."""

    def __init__(
        self,
        planner_guard_s: float,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if planner_guard_s < 0 or cooldown_s < 0:
            raise ValueError("guard durations must not be negative")
        self._planner_guard_s = planner_guard_s
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._agent_busy = False
        self._blocked_until = 0.0

    def blocked(self) -> bool:
        with self._lock:
            return self._agent_busy or self._clock() < self._blocked_until

    def text_published(self) -> None:
        with self._lock:
            self._blocked_until = self._clock() + self._planner_guard_s

    def agent_state(self, state: str) -> bool:
        """Update state and return True when a plan reached a terminal state."""
        with self._lock:
            if state == "running":
                self._agent_busy = True
                return False
            if state in ("plan_succeeded", "plan_failed"):
                self._agent_busy = False
                self._blocked_until = self._clock() + self._cooldown_s
                return True
        return False

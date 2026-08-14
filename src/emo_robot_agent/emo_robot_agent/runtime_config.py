"""Validation for independently selectable motion and TTS runtimes."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeModes:
    """Validated Agent runtime selection and motion safety gates."""

    motion_mode: str
    tts_mode: str
    motion_enabled: bool
    confirmation: str

    @classmethod
    def create(
        cls,
        motion_mode: str,
        tts_mode: str,
        motion_enabled: bool,
        confirmation: str,
    ) -> "RuntimeModes":
        motion = motion_mode.strip().lower()
        tts = tts_mode.strip().lower()
        valid = {"simulation", "real"}
        if motion not in valid:
            raise ValueError("motion_mode must be simulation or real")
        if tts not in valid:
            raise ValueError("tts_mode must be simulation or real")
        if motion == "simulation" and motion_enabled:
            raise ValueError(
                "motion_enabled must be false when motion_mode=simulation"
            )
        if (
            motion == "real"
            and motion_enabled
            and confirmation != "I_UNDERSTAND"
        ):
            raise ValueError(
                "confirmation must be exactly I_UNDERSTAND when real "
                "preset motion is enabled"
            )
        return cls(motion, tts, bool(motion_enabled), confirmation)

    @property
    def real_motion(self) -> bool:
        return self.motion_mode == "real"

    @property
    def real_tts(self) -> bool:
        return self.tts_mode == "real"

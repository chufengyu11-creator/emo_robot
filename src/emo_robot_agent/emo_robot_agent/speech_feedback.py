"""Plan-level speech feedback adapters for simulation and real execution."""

from __future__ import annotations

from typing import Callable


SpeechDone = Callable[[bool, str], None]


class SimulationSpeechFeedback:
    """Log generated speech without creating or calling a TTS client."""

    def __init__(self, logger) -> None:
        self._logger = logger

    def execute(self, text: str, done_callback: SpeechDone) -> None:
        detail = f"SIMULATION TTS:\n{text}"
        self._logger.info(detail)
        done_callback(True, detail)


class RealSpeechFeedback:
    """Send generated speech through the existing SpeechClient."""

    def __init__(self, speech_client, domain: str = "emo_robot") -> None:
        self._speech_client = speech_client
        self._domain = domain

    def execute(self, text: str, done_callback: SpeechDone) -> None:
        def _done(success: bool) -> None:
            detail = (
                "plan speech TTS accepted"
                if success
                else "plan speech TTS request failed"
            )
            done_callback(success, detail)

        self._speech_client.say_async(
            text,
            domain=self._domain,
            done_callback=_done,
        )

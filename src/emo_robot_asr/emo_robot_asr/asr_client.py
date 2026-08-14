"""HTTP client for the remote SenseVoice transcription endpoint."""

from __future__ import annotations

import time
from typing import Callable, Optional

import requests


Sleep = Callable[[float], None]


class AsrClient:
    """Upload a WAV file and return the validated text field."""

    def __init__(
        self,
        url: str,
        timeout_s: float = 30.0,
        retry_count: int = 1,
        retry_interval_s: float = 0.5,
        session: Optional[requests.Session] = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("asr_url must use http or https")
        if timeout_s <= 0 or retry_count < 0 or retry_interval_s < 0:
            raise ValueError("invalid ASR retry parameters")
        self.url = url
        self.timeout_s = timeout_s
        self.retry_count = retry_count
        self.retry_interval_s = retry_interval_s
        self._session = session or requests.Session()
        self._sleep = sleep

    def transcribe(self, wav_data: bytes) -> str:
        """Return recognized text; retry transport and malformed responses."""
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_count + 1):
            try:
                response = self._session.post(
                    self.url,
                    files={"file": ("speech.wav", wav_data, "audio/wav")},
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("ASR response must be a JSON object")
                text = payload.get("text")
                if not isinstance(text, str):
                    raise ValueError("ASR response has no string text field")
                return text.strip()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.retry_count:
                    self._sleep(self.retry_interval_s)
        raise RuntimeError(f"ASR request failed: {last_error}")

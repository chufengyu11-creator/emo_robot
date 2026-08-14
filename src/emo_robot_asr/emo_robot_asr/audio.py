"""Pure-Python VAD segment assembly and WAV conversion."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Dict, Optional
import wave


@dataclass(frozen=True)
class SegmentResult:
    """Result of processing one AimDK VAD message."""

    state: str
    stream_id: int
    pcm: Optional[bytes] = None
    detail: str = ""


class VadSegmentAssembler:
    """Assemble independent PCM utterances for every AimDK stream ID."""

    def __init__(
        self,
        min_utterance_s: float = 0.25,
        max_utterance_s: float = 15.0,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
    ) -> None:
        if min_utterance_s < 0 or max_utterance_s <= min_utterance_s:
            raise ValueError("invalid utterance duration limits")
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        bytes_per_second = sample_rate * sample_width * channels
        self.min_bytes = int(min_utterance_s * bytes_per_second)
        self.max_bytes = int(max_utterance_s * bytes_per_second)
        self._buffers: Dict[int, bytearray] = {}

    def clear(self) -> None:
        self._buffers.clear()

    def process(
        self,
        stream_id: int,
        vad_state: int,
        audio_data: bytes,
    ) -> SegmentResult:
        data = bytes(audio_data)
        if vad_state == 1:
            self._buffers[stream_id] = bytearray(data)
            return self._check_size(stream_id, "recording")
        if vad_state == 2:
            if stream_id not in self._buffers:
                return SegmentResult("ignored", stream_id, detail="VAD=2 without start")
            self._buffers[stream_id].extend(data)
            return self._check_size(stream_id, "recording")
        if vad_state == 3:
            if stream_id not in self._buffers:
                return SegmentResult("ignored", stream_id, detail="VAD=3 without start")
            self._buffers[stream_id].extend(data)
            if len(self._buffers[stream_id]) > self.max_bytes:
                self._buffers.pop(stream_id, None)
                return SegmentResult("dropped", stream_id, detail="utterance too long")
            pcm = bytes(self._buffers.pop(stream_id))
            if len(pcm) < self.min_bytes:
                return SegmentResult("dropped", stream_id, detail="utterance too short")
            return SegmentResult("complete", stream_id, pcm=pcm)
        if vad_state == 0:
            existed = self._buffers.pop(stream_id, None) is not None
            detail = "incomplete utterance reset" if existed else "idle"
            return SegmentResult("dropped" if existed else "idle", stream_id, detail=detail)
        return SegmentResult("ignored", stream_id, detail=f"unknown VAD={vad_state}")

    def _check_size(self, stream_id: int, state: str) -> SegmentResult:
        if len(self._buffers[stream_id]) > self.max_bytes:
            self._buffers.pop(stream_id, None)
            return SegmentResult("dropped", stream_id, detail="utterance too long")
        return SegmentResult(state, stream_id)


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """Wrap raw interleaved PCM bytes in an in-memory WAV container."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()

import io
import wave

from emo_robot_asr.audio import VadSegmentAssembler, pcm_to_wav


def test_vad_assembles_independent_streams():
    assembler = VadSegmentAssembler(0.0, 1.0, sample_rate=10)
    assert assembler.process(1, 1, b"aa").state == "recording"
    assert assembler.process(2, 1, b"XX").state == "recording"
    assembler.process(1, 2, b"bb")
    first = assembler.process(1, 3, b"cc")
    second = assembler.process(2, 3, b"YY")
    assert first.pcm == b"aabbcc"
    assert second.pcm == b"XXYY"


def test_vad_rejects_invalid_sequences_and_resets():
    assembler = VadSegmentAssembler(0.0, 1.0, sample_rate=10)
    assert assembler.process(1, 2, b"data").state == "ignored"
    assert assembler.process(1, 3, b"data").state == "ignored"
    assembler.process(1, 1, b"data")
    reset = assembler.process(1, 0, b"")
    assert reset.state == "dropped"
    assert assembler.process(1, 3, b"").state == "ignored"


def test_vad_drops_short_and_long_utterances():
    assembler = VadSegmentAssembler(0.25, 0.5, sample_rate=10)
    assembler.process(1, 1, b"12")
    assert assembler.process(1, 3, b"34").detail == "utterance too short"
    assembler.process(1, 1, b"12345678")
    result = assembler.process(1, 2, b"901")
    assert result.state == "dropped"
    assert result.detail == "utterance too long"


def test_pcm_to_wav_has_expected_format_and_frames():
    pcm = b"\x01\x02" * 160
    encoded = pcm_to_wav(pcm)
    with wave.open(io.BytesIO(encoded), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(160) == pcm

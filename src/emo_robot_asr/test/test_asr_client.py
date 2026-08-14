import requests
import pytest

from emo_robot_asr.asr_client import AsrClient


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, files, timeout):
        self.calls.append((url, files, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_asr_client_uploads_named_wav_and_returns_trimmed_text():
    session = FakeSession([FakeResponse({"text": " 你好 "})])
    client = AsrClient("http://server/transcribe", session=session)
    assert client.transcribe(b"wav") == "你好"
    url, files, timeout = session.calls[0]
    assert url == "http://server/transcribe"
    assert files["file"] == ("speech.wav", b"wav", "audio/wav")
    assert timeout == 30.0


def test_asr_client_accepts_empty_text_without_retry():
    session = FakeSession([FakeResponse({"text": ""})])
    client = AsrClient("http://server/transcribe", session=session)
    assert client.transcribe(b"wav") == ""
    assert len(session.calls) == 1


def test_asr_client_retries_transport_failure():
    session = FakeSession([
        requests.ConnectionError("offline"),
        FakeResponse({"text": "ok"}),
    ])
    sleeps = []
    client = AsrClient(
        "http://server/transcribe",
        retry_count=1,
        retry_interval_s=0.5,
        session=session,
        sleep=sleeps.append,
    )
    assert client.transcribe(b"wav") == "ok"
    assert sleeps == [0.5]


@pytest.mark.parametrize("payload", [[], {}, {"text": 1}])
def test_asr_client_rejects_malformed_json(payload):
    session = FakeSession([FakeResponse(payload)])
    client = AsrClient(
        "http://server/transcribe", retry_count=0, session=session
    )
    with pytest.raises(RuntimeError, match="ASR request failed"):
        client.transcribe(b"wav")

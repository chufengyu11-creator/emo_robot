"""ROS 2 bridge from AimDK processed VAD audio to remote ASR text."""

from __future__ import annotations

import json
import queue
import threading
from typing import Optional

from aimdk_msgs.msg import ProcessedAudioOutput
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from emo_robot_asr.asr_client import AsrClient
from emo_robot_asr.audio import VadSegmentAssembler, pcm_to_wav
from emo_robot_asr.guard import AgentGuard


class VadAsrNode(Node):
    """Collect complete VAD utterances and publish final ASR text."""

    def __init__(self) -> None:
        super().__init__("vad_asr")
        defaults = {
            "audio_topic": "/agent/process_audio_output",
            "output_topic": "/emo_robot/agent/text_input",
            "agent_status_topic": "/emo_robot/agent/status",
            "status_topic": "/emo_robot/asr/status",
            "asr_url": (
                "http://111.56.189.29:8765/v1/audio/transcriptions"
            ),
            "request_timeout_s": 30.0,
            "min_utterance_s": 0.25,
            "max_utterance_s": 15.0,
            "retry_count": 1,
            "retry_interval_s": 0.5,
            "post_agent_cooldown_s": 1.0,
            "planner_guard_s": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._assembler = VadSegmentAssembler(
            min_utterance_s=float(
                self.get_parameter("min_utterance_s").value
            ),
            max_utterance_s=float(
                self.get_parameter("max_utterance_s").value
            ),
        )
        self._client = AsrClient(
            url=str(self.get_parameter("asr_url").value),
            timeout_s=float(self.get_parameter("request_timeout_s").value),
            retry_count=int(self.get_parameter("retry_count").value),
            retry_interval_s=float(
                self.get_parameter("retry_interval_s").value
            ),
        )
        self._cooldown_s = float(
            self.get_parameter("post_agent_cooldown_s").value
        )
        self._planner_guard_s = float(
            self.get_parameter("planner_guard_s").value
        )
        self._guard = AgentGuard(self._planner_guard_s, self._cooldown_s)

        self._text_publisher = self.create_publisher(
            String, str(self.get_parameter("output_topic").value), 10
        )
        self._status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=500,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            ProcessedAudioOutput,
            str(self.get_parameter("audio_topic").value),
            self._on_audio,
            qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("agent_status_topic").value),
            self._on_agent_status,
            10,
        )

        self._jobs: queue.Queue[Optional[tuple[int, bytes]]] = queue.Queue(
            maxsize=1
        )
        self._worker_busy = threading.Event()
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_main,
            name="emo_robot_asr_worker",
            daemon=True,
        )
        self._worker.start()
        self._publish_status("listening", detail="waiting for AimDK VAD")
        self.get_logger().info(
            "ASR ready: audio=%s output=%s"
            % (
                self.get_parameter("audio_topic").value,
                self.get_parameter("output_topic").value,
            )
        )

    def _is_blocked(self) -> bool:
        return self._guard.blocked()

    def _on_audio(self, message: ProcessedAudioOutput) -> None:
        stream_id = int(message.stream_id)
        if self._is_blocked():
            self._assembler.process(stream_id, 0, b"")
            return
        vad_state = int(message.audio_vad_state.value)
        result = self._assembler.process(
            stream_id, vad_state, bytes(message.audio_data)
        )
        if result.state == "recording" and vad_state == 1:
            self._publish_status("recording", stream_id)
        elif result.state == "dropped":
            self._publish_status("dropped", stream_id, result.detail)
        elif result.state == "complete" and result.pcm is not None:
            self._submit(stream_id, result.pcm)

    def _submit(self, stream_id: int, pcm: bytes) -> None:
        if self._worker_busy.is_set() or not self._jobs.empty():
            self._publish_status("dropped", stream_id, "ASR worker is busy")
            self.get_logger().warning("Dropping utterance: ASR worker is busy.")
            return
        try:
            self._jobs.put_nowait((stream_id, pcm))
        except queue.Full:
            self._publish_status("dropped", stream_id, "ASR queue is full")

    def _worker_main(self) -> None:
        while not self._stopping.is_set():
            job = self._jobs.get()
            if job is None:
                return
            stream_id, pcm = job
            self._worker_busy.set()
            self._publish_status("transcribing", stream_id)
            try:
                text = self._client.transcribe(pcm_to_wav(pcm))
                if not text:
                    self._publish_status(
                        "dropped", stream_id, "ASR returned empty text"
                    )
                    continue
                output = String()
                output.data = text
                self._text_publisher.publish(output)
                self._guard.text_published()
                self._publish_status("succeeded", stream_id, text=text)
                self.get_logger().info(f"ASR text: {text}")
            except Exception as exc:
                self._publish_status("failed", stream_id, str(exc))
                self.get_logger().error(str(exc))
            finally:
                self._worker_busy.clear()

    def _on_agent_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            state = payload.get("state")
        except (json.JSONDecodeError, AttributeError):
            self.get_logger().warning("Ignoring invalid Agent status JSON.")
            return
        if self._guard.agent_state(state):
            self._assembler.clear()

    def _publish_status(
        self,
        state: str,
        stream_id: Optional[int] = None,
        detail: str = "",
        text: str = "",
    ) -> None:
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "stream_id": stream_id,
                "detail": detail,
                "text": text,
            },
            ensure_ascii=False,
        )
        self._status_publisher.publish(message)

    def destroy_node(self) -> bool:
        if not self._stopping.is_set():
            self._stopping.set()
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=2.0)
            self._assembler.clear()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VadAsrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

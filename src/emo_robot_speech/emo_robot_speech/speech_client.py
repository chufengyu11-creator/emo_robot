"""Reusable TTS client for emo_robot.

Wraps the AimDK PlayTts ROS 2 service behind a simple call interface
so that any node can speak with a single line.

Usage:
    from emo_robot_speech import SpeechClient

    class MyNode(Node):
        def __init__(self):
            super().__init__("my_node")
            self.speech = SpeechClient(self)
        def on_event(self):
            self.speech.say_async("你好！")
"""

from __future__ import annotations

from typing import Callable, Optional

from rclpy.node import Node

from aimdk_msgs.srv import PlayTts


class SpeechClient:
    """Thin wrapper around the AimDK PlayTts ROS 2 service.

    The client is created once and reused for the lifetime of the
    owning node.  Calls never spin or block the owning node's executor.
    """

    def __init__(
        self,
        node: Node,
        request_timeout_s: float = 0.25,
        max_attempts: int = 8,
    ) -> None:
        """Create a non-blocking TTS client bound to *node*."""
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be greater than zero")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")

        self._node = node
        self._logger = node.get_logger()
        self._request_timeout_s = request_timeout_s
        self._max_attempts = max_attempts
        self._client = node.create_client(
            PlayTts,
            "/aimdk_5Fmsgs/srv/PlayTts",
        )
        self._logger.info("PlayTts async client created.")

    def say_async(
        self,
        text: str,
        domain: str = "emo_robot",
        priority: int = 6,
        interrupt: bool = True,
        done_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Request speech without blocking or spinning the executor.

        *done_callback* (if given) is invoked with ``True`` when the
        TTS request is accepted, or ``False`` after a response failure
        or after all timed attempts have failed.
        """
        if not text:
            self._logger.error("TTS text must not be empty.")
            if done_callback:
                done_callback(False)
            return
        if not 1 <= priority <= 10:
            self._logger.error("TTS priority must be in the range 1..10.")
            if done_callback:
                done_callback(False)
            return

        self._logger.info(f"📨 TTS (async): {text}")

        state = {
            "attempt": 0,
            "finished": False,
            "generation": 0,
            "timer": None,
        }

        def _cancel_timer() -> None:
            timer = state["timer"]
            if timer is None:
                return
            state["timer"] = None
            timer.cancel()
            try:
                self._node.destroy_timer(timer)
            except Exception:
                pass

        def _finish(success: bool) -> None:
            if state["finished"]:
                return
            state["finished"] = True
            state["generation"] += 1
            _cancel_timer()
            if done_callback:
                try:
                    done_callback(success)
                except Exception as exc:
                    self._logger.error(
                        f"TTS completion callback failed: {exc}"
                    )

        def _schedule_retry(reason: str) -> None:
            if state["finished"]:
                return
            state["attempt"] += 1
            if state["attempt"] >= self._max_attempts:
                self._logger.error("❌ TTS async: all retries exhausted.")
                _finish(False)
                return
            self._logger.warning(
                f"TTS async retry "
                f"{state['attempt'] + 1}/{self._max_attempts}: {reason}"
            )
            _try()

        def _try() -> None:
            if state["finished"]:
                return

            state["generation"] += 1
            generation = state["generation"]

            if not self._client.service_is_ready():
                def _on_service_wait_timeout() -> None:
                    if (
                        state["finished"]
                        or generation != state["generation"]
                    ):
                        return
                    _cancel_timer()
                    _schedule_retry("PlayTts service is not ready")

                state["timer"] = self._node.create_timer(
                    self._request_timeout_s,
                    _on_service_wait_timeout,
                )
                return

            req = PlayTts.Request()
            req.tts_req.text = text
            req.tts_req.domain = domain
            req.tts_req.trace_id = "emo_robot"
            req.tts_req.is_interrupted = interrupt
            req.tts_req.priority_weight = 0
            req.tts_req.priority_level.value = priority
            req.header.header.stamp = (
                self._node.get_clock().now().to_msg()
            )

            try:
                future = self._client.call_async(req)
            except Exception as exc:
                _schedule_retry(f"request could not be sent: {exc}")
                return

            def _on_timeout() -> None:
                if state["finished"] or generation != state["generation"]:
                    return
                if future.done():
                    return
                state["generation"] += 1
                _cancel_timer()
                future.cancel()
                _schedule_retry(
                    f"no response within {self._request_timeout_s:.2f}s"
                )

            state["timer"] = self._node.create_timer(
                self._request_timeout_s,
                _on_timeout,
            )

            def _on_response(fut) -> None:
                if state["finished"] or generation != state["generation"]:
                    return
                _cancel_timer()
                try:
                    resp = fut.result()
                except Exception as exc:
                    _schedule_retry(f"request failed: {exc}")
                    return
                if resp is None:
                    _schedule_retry("response is None")
                    return

                if resp.tts_resp.is_success:
                    self._logger.info("✅ TTS async: sent successfully.")
                    _finish(True)
                else:
                    self._logger.error("❌ TTS async: failed.")
                    _finish(False)

            future.add_done_callback(_on_response)

        _try()

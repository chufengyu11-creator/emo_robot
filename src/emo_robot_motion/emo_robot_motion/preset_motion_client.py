"""Reusable preset-motion client for emo_robot.

Wraps the AimDK SetMcPresetMotion ROS 2 service behind a small async
interface so other nodes can trigger preset motions without blocking or
spinning their executor.
"""

from __future__ import annotations

from typing import Callable, Optional

from rclpy.node import Node

from aimdk_msgs.msg import (
    CommonState,
    McControlArea,
    McPresetMotion,
    RequestHeader,
)
from aimdk_msgs.srv import SetMcPresetMotion


class PresetMotionClient:
    """Thin wrapper around the AimDK SetMcPresetMotion ROS 2 service."""

    def __init__(
        self,
        node: Node,
        request_timeout_s: float = 0.25,
        max_attempts: int = 8,
        service_name: str = "/aimdk_5Fmsgs/srv/SetMcPresetMotion",
    ) -> None:
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be greater than zero")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if not service_name:
            raise ValueError("service_name must not be empty")

        self._node = node
        self._logger = node.get_logger()
        self._request_timeout_s = request_timeout_s
        self._max_attempts = max_attempts
        self._client = node.create_client(
            SetMcPresetMotion,
            service_name,
        )
        self._logger.info("SetMcPresetMotion async client created.")

    def play_async(
        self,
        area: int,
        motion_id: int,
        interrupt: bool = False,
        done_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Request a preset motion without blocking the executor."""
        if area <= 0:
            self._logger.error("Preset motion area must be greater than zero.")
            self._finish_callback(done_callback, False)
            return
        if motion_id <= 0:
            self._logger.error(
                "Preset motion id must be greater than zero."
            )
            self._finish_callback(done_callback, False)
            return

        self._logger.info(
            f"Preset motion (async): area={area} motion={motion_id}"
        )

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
            self._finish_callback(done_callback, success)

        def _schedule_retry(reason: str) -> None:
            if state["finished"]:
                return
            state["attempt"] += 1
            if state["attempt"] >= self._max_attempts:
                self._logger.error(
                    "Preset motion async: all retries exhausted."
                )
                _finish(False)
                return
            self._logger.warning(
                f"Preset motion async retry "
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
                    _schedule_retry(
                        "SetMcPresetMotion service is not ready"
                    )

                state["timer"] = self._node.create_timer(
                    self._request_timeout_s,
                    _on_service_wait_timeout,
                )
                return

            req = SetMcPresetMotion.Request()
            req.header = RequestHeader()
            req.header.stamp = self._node.get_clock().now().to_msg()
            req.motion = McPresetMotion(value=motion_id)
            req.area = McControlArea(value=area)
            req.interrupt = interrupt

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

                code = resp.response.header.code
                state_value = resp.response.state.value
                task_id = resp.response.task_id
                if code == 0:
                    self._logger.info(
                        f"Preset motion OK task={task_id}"
                    )
                    _finish(True)
                elif state_value == CommonState.RUNNING:
                    self._logger.info(
                        f"Preset motion running task={task_id}"
                    )
                    _finish(True)
                else:
                    self._logger.error(
                        f"Preset motion failed code={code} task={task_id}"
                    )
                    _finish(False)

            future.add_done_callback(_on_response)

        _try()

    def _finish_callback(
        self,
        done_callback: Optional[Callable[[bool], None]],
        success: bool,
    ) -> None:
        if done_callback is None:
            return
        try:
            done_callback(success)
        except Exception as exc:
            self._logger.error(
                f"Preset motion completion callback failed: {exc}"
            )

"""Reusable asynchronous client for the AimDK GetMcAction service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from rclpy.node import Node

from aimdk_msgs.msg import CommonRequest, McActionStatus
from aimdk_msgs.srv import GetMcAction


@dataclass(frozen=True)
class McActionState:
    """Snapshot of the robot motion-controller action state."""

    action: int
    description: str
    status: int
    response_code: int

    def is_running(self, expected_action: int) -> bool:
        """Return whether the requested action is currently running."""
        return bool(
            self.response_code == 0
            and self.action == int(expected_action)
            and self.status == McActionStatus.RUNNING
        )


McActionStateCallback = Callable[
    [bool, Optional[McActionState], str],
    None,
]


class McActionStateClient:
    """Query GetMcAction asynchronously with bounded retry and timeout."""

    def __init__(
        self,
        node: Node,
        request_timeout_s: float = 0.25,
        max_attempts: int = 8,
        service_name: str = "/aimdk_5Fmsgs/srv/GetMcAction",
    ) -> None:
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be greater than zero")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if not service_name:
            raise ValueError("service_name must not be empty")

        self._node = node
        self._logger = node.get_logger()
        self._request_timeout_s = float(request_timeout_s)
        self._max_attempts = int(max_attempts)
        self._client = node.create_client(GetMcAction, service_name)

        self._active = False
        self._attempts = 0
        self._generation = 0
        self._timer = None
        self._future = None
        self._callback: Optional[McActionStateCallback] = None
        self._logger.info("GetMcAction async client created.")

    @property
    def active(self) -> bool:
        """Return whether a query is currently in progress."""
        return self._active

    def get_state_async(
        self,
        done_callback: McActionStateCallback,
    ) -> None:
        """Start one asynchronous state query.

        Only one query may be active at a time. A second request is rejected
        through its own callback without disturbing the in-flight query.
        """
        if not callable(done_callback):
            raise TypeError("done_callback must be callable")
        if self._active:
            self._invoke_callback(
                done_callback,
                False,
                None,
                "GetMcAction query is already in progress",
            )
            return

        self._active = True
        self._attempts = 0
        self._callback = done_callback
        self._try_query()

    def cancel(self) -> None:
        """Silently cancel the active query and ignore late responses."""
        if not self._active:
            return
        self._active = False
        self._generation += 1
        self._cancel_timer()
        future = self._future
        self._future = None
        self._callback = None
        if future is not None and not future.done():
            future.cancel()

    def _try_query(self) -> None:
        if not self._active:
            return
        if self._attempts >= self._max_attempts:
            self._complete(
                False,
                None,
                "GetMcAction retries exhausted",
            )
            return

        self._attempts += 1
        self._generation += 1
        generation = self._generation

        if not self._client.service_is_ready():
            self._timer = self._node.create_timer(
                self._request_timeout_s,
                lambda: self._retry(
                    generation,
                    "GetMcAction service is not ready",
                ),
            )
            return

        request = GetMcAction.Request()
        request.request = CommonRequest()
        request.request.header.stamp = (
            self._node.get_clock().now().to_msg()
        )
        try:
            future = self._client.call_async(request)
        except Exception as exc:
            self._retry(generation, f"request failed: {exc}")
            return

        self._future = future
        self._timer = self._node.create_timer(
            self._request_timeout_s,
            lambda: self._on_timeout(generation, future),
        )
        future.add_done_callback(
            lambda fut: self._on_response(generation, fut)
        )

    def _retry(self, generation: int, reason: str) -> None:
        if not self._active or generation != self._generation:
            return
        self._cancel_timer()
        if self._attempts >= self._max_attempts:
            self._complete(
                False,
                None,
                f"GetMcAction retries exhausted: {reason}",
            )
            return
        self._logger.warning(
            f"GetMcAction retry {self._attempts + 1}/"
            f"{self._max_attempts}: {reason}"
        )
        self._try_query()

    def _on_timeout(self, generation: int, future) -> None:
        if not self._active or generation != self._generation:
            return
        if future.done():
            return
        self._generation += 1
        self._cancel_timer()
        future.cancel()
        self._retry(
            self._generation,
            f"no response within {self._request_timeout_s:.2f}s",
        )

    def _on_response(self, generation: int, future) -> None:
        if not self._active or generation != self._generation:
            return
        self._cancel_timer()
        self._future = None
        try:
            response = future.result()
        except Exception as exc:
            self._retry(generation, f"response failed: {exc}")
            return
        if response is None:
            self._retry(generation, "response is None")
            return

        try:
            state = McActionState(
                action=int(response.info.current_action.value),
                description=str(response.info.action_desc),
                status=int(response.info.status.value),
                response_code=int(response.header.code),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self._retry(generation, f"invalid response: {exc}")
            return

        if state.response_code != 0:
            self._complete(
                False,
                state,
                f"GetMcAction returned code={state.response_code}",
            )
            return
        self._complete(True, state, "")

    def _cancel_timer(self) -> None:
        timer = self._timer
        if timer is None:
            return
        self._timer = None
        timer.cancel()
        try:
            self._node.destroy_timer(timer)
        except Exception:
            pass

    def _complete(
        self,
        success: bool,
        state: Optional[McActionState],
        error: str,
    ) -> None:
        if not self._active:
            return
        callback = self._callback
        self._active = False
        self._generation += 1
        self._cancel_timer()
        self._future = None
        self._callback = None
        if callback is not None:
            self._invoke_callback(
                callback,
                success,
                state,
                error,
            )

    def _invoke_callback(
        self,
        callback: McActionStateCallback,
        success: bool,
        state: Optional[McActionState],
        error: str,
    ) -> None:
        try:
            callback(success, state, error)
        except Exception as exc:
            self._logger.error(
                f"GetMcAction completion callback failed: {exc}"
            )

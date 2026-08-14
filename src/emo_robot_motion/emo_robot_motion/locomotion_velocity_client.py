"""Reusable locomotion velocity client for emo_robot.

This module wraps the AimDK input-source service and locomotion velocity
topic. It is intentionally only a client library; it does not create a
standalone ROS node or decide when the robot should move.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from rclpy.node import Node

from aimdk_msgs.msg import (
    CommonState,
    McInputAction,
    McLocomotionVelocity,
    MessageHeader,
)
from aimdk_msgs.srv import SetMcInputSource


class LocomotionVelocityClient:
    """Async AimDK locomotion velocity adapter.

    The client never publishes non-zero velocity until the input source
    registration has completed successfully.
    """

    _INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/SetMcInputSource"
    _VELOCITY_TOPIC = "/aima/mc/locomotion/velocity"

    def __init__(
        self,
        node: Node,
        source_name: str = "node",
        publish_period_s: float = 0.02,
        request_timeout_s: float = 0.25,
        max_attempts: int = 8,
    ) -> None:
        if not source_name:
            raise ValueError("source_name must not be empty")
        if publish_period_s <= 0.0:
            raise ValueError("publish_period_s must be greater than zero")
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be greater than zero")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")

        self._node = node
        self._logger = node.get_logger()
        self._source_name = source_name
        self._publish_period_s = publish_period_s
        self._request_timeout_s = request_timeout_s
        self._max_attempts = max_attempts

        self._publisher = node.create_publisher(
            McLocomotionVelocity,
            self._VELOCITY_TOPIC,
            10,
        )
        self._input_source_client = node.create_client(
            SetMcInputSource,
            self._INPUT_SOURCE_SERVICE,
        )

        self._forward_velocity = 0.0
        self._lateral_velocity = 0.0
        self._angular_velocity = 0.0
        self._registered = False
        self._registering = False
        self._unregistering = False
        self._publish_timer = None

        self._logger.info("Locomotion velocity client created.")

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def registering(self) -> bool:
        return self._registering

    def register_input_source_async(
        self,
        done_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Register the configured AimDK locomotion input source."""
        if self._registered:
            self._finish_callback(done_callback, True)
            return
        if self._registering:
            self._logger.warning("Input source registration already active.")
            self._finish_callback(done_callback, False)
            return

        self._registering = True
        self._logger.info("Registering locomotion input source.")

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
            self._registering = False
            self._registered = success
            _cancel_timer()
            self._finish_callback(done_callback, success)

        def _schedule_retry(reason: str) -> None:
            if state["finished"]:
                return
            state["attempt"] += 1
            if state["attempt"] >= self._max_attempts:
                self._logger.error(
                    "Input source registration: all retries exhausted."
                )
                _finish(False)
                return
            self._logger.warning(
                f"Input source registration retry "
                f"{state['attempt'] + 1}/{self._max_attempts}: {reason}"
            )
            _try()

        def _try() -> None:
            if state["finished"]:
                return

            state["generation"] += 1
            generation = state["generation"]

            if not self._input_source_client.service_is_ready():

                def _on_service_wait_timeout() -> None:
                    if (
                        state["finished"]
                        or generation != state["generation"]
                    ):
                        return
                    _cancel_timer()
                    _schedule_retry("SetMcInputSource service is not ready")

                state["timer"] = self._node.create_timer(
                    self._request_timeout_s,
                    _on_service_wait_timeout,
                )
                return

            req = SetMcInputSource.Request()
            req.request.header.stamp = self._node.get_clock().now().to_msg()
            req.action.value = McInputAction.INPUTACTION_ADD
            req.input_source.name = self._source_name
            req.input_source.priority = 40
            req.input_source.timeout = 1000

            try:
                future = self._input_source_client.call_async(req)
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
                if code == 0 or state_value == CommonState.RUNNING:
                    self._logger.info(
                        f"Input source registered task={task_id}"
                    )
                    _finish(True)
                else:
                    self._logger.error(
                        "Input source registration failed "
                        f"code={code} task={task_id}"
                    )
                    _finish(False)

            future.add_done_callback(_on_response)

        _try()

    def unregister_input_source_async(
        self,
        done_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Delete the configured AimDK locomotion input source."""
        if not self._registered:
            self._finish_callback(done_callback, True)
            return
        if self._unregistering:
            self._logger.warning(
                "Input source unregistration already active."
            )
            self._finish_callback(done_callback, False)
            return

        self._unregistering = True
        self._logger.info("Unregistering locomotion input source.")
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
            self._unregistering = False
            if success:
                self._registered = False
            _cancel_timer()
            self._finish_callback(done_callback, success)

        def _schedule_retry(reason: str) -> None:
            if state["finished"]:
                return
            state["attempt"] += 1
            if state["attempt"] >= self._max_attempts:
                self._logger.error(
                    "Input source unregistration: all retries exhausted."
                )
                _finish(False)
                return
            self._logger.warning(
                f"Input source unregistration retry "
                f"{state['attempt'] + 1}/{self._max_attempts}: {reason}"
            )
            _try()

        def _try() -> None:
            if state["finished"]:
                return

            state["generation"] += 1
            generation = state["generation"]
            if not self._input_source_client.service_is_ready():

                def _on_service_wait_timeout() -> None:
                    if (
                        state["finished"]
                        or generation != state["generation"]
                    ):
                        return
                    _cancel_timer()
                    _schedule_retry(
                        "SetMcInputSource service is not ready"
                    )

                state["timer"] = self._node.create_timer(
                    self._request_timeout_s,
                    _on_service_wait_timeout,
                )
                return

            req = SetMcInputSource.Request()
            req.request.header.stamp = self._node.get_clock().now().to_msg()
            req.action.value = McInputAction.INPUTACTION_DELETE
            req.input_source.name = self._source_name
            req.input_source.priority = 40
            req.input_source.timeout = 1000

            try:
                future = self._input_source_client.call_async(req)
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
                task_id = resp.response.task_id
                if code == 0:
                    self._logger.info(
                        f"Input source unregistered task={task_id}"
                    )
                    _finish(True)
                else:
                    self._logger.error(
                        "Input source unregistration failed "
                        f"code={code} task={task_id}"
                    )
                    _finish(False)

            future.add_done_callback(_on_response)

        _try()

    def start_publishing(self) -> None:
        """Start the periodic velocity publisher."""
        if self._publish_timer is not None:
            return
        if not self._registered:
            self._logger.warning(
                "Starting velocity publisher before input source "
                "registration; only zero velocity will be published."
            )
        self._publish_timer = self._node.create_timer(
            self._publish_period_s,
            self._publish_velocity,
        )

    def stop_publishing(self) -> None:
        """Stop the periodic velocity publisher."""
        if self._publish_timer is None:
            return
        timer = self._publish_timer
        self._publish_timer = None
        timer.cancel()
        try:
            self._node.destroy_timer(timer)
        except Exception:
            pass

    def set_velocity(
        self,
        forward: float,
        lateral: float,
        angular: float,
    ) -> None:
        """Set the desired velocity after validating AimDK thresholds."""
        self._forward_velocity = self._validate_velocity(
            "forward",
            forward,
            min_abs=0.2,
            max_abs=1.0,
        )
        self._lateral_velocity = self._validate_velocity(
            "lateral",
            lateral,
            min_abs=0.2,
            max_abs=1.0,
        )
        self._angular_velocity = self._validate_velocity(
            "angular",
            angular,
            min_abs=0.1,
            max_abs=1.0,
        )

    def stop(self) -> None:
        """Set zero velocity and publish one zero frame."""
        self._set_zero_velocity()
        self._publish_velocity(force_zero=True)

    def emergency_stop(self) -> None:
        """Publish several zero frames and stop the periodic publisher."""
        self._set_zero_velocity()
        for _ in range(3):
            self._publish_velocity(force_zero=True)
        self.stop_publishing()

    def _publish_velocity(self, force_zero: bool = False) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.source = self._source_name

        if force_zero or not self._registered:
            msg.forward_velocity = 0.0
            msg.lateral_velocity = 0.0
            msg.angular_velocity = 0.0
        else:
            msg.forward_velocity = self._forward_velocity
            msg.lateral_velocity = self._lateral_velocity
            msg.angular_velocity = self._angular_velocity

        self._publisher.publish(msg)

    @staticmethod
    def _validate_velocity(
        name: str,
        value: float,
        min_abs: float,
        max_abs: float,
    ) -> float:
        velocity = float(value)
        if not math.isfinite(velocity):
            raise ValueError(f"{name} velocity must be finite")
        magnitude = abs(velocity)
        if magnitude < 0.005:
            return 0.0
        if magnitude < min_abs or magnitude > max_abs:
            raise ValueError(
                f"{name} velocity must be 0 or ±({min_abs}..{max_abs})"
            )
        return velocity

    def _set_zero_velocity(self) -> None:
        self._forward_velocity = 0.0
        self._lateral_velocity = 0.0
        self._angular_velocity = 0.0

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
                f"Input source registration callback failed: {exc}"
            )

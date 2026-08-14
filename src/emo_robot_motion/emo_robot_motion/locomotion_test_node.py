#!/usr/bin/env python3

"""Safety-gated, short-duration AimDK locomotion test node."""

from __future__ import annotations

from dataclasses import dataclass
import signal
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from aimdk_msgs.msg import McAction

from emo_robot_motion.locomotion_velocity_client import (
    LocomotionVelocityClient,
)
from emo_robot_motion.mc_action_state_client import (
    McActionState,
    McActionStateClient,
)


@dataclass(frozen=True)
class LocomotionTestCommand:
    """Validated parameters for one short locomotion test."""

    motion_enabled: bool = False
    confirmation: str = ""
    forward_velocity: float = 0.0
    lateral_velocity: float = 0.0
    angular_velocity: float = 0.0
    duration_s: float = 2.0

    CONFIRMATION = "I_UNDERSTAND"
    MAX_DURATION_S = 5.0

    def validated_velocities(self) -> tuple[float, float, float]:
        """Validate safety gates and return normalized velocities."""
        duration = float(self.duration_s)
        if not 0.0 < duration <= self.MAX_DURATION_S:
            raise ValueError(
                f"duration_s must be greater than 0 and at most "
                f"{self.MAX_DURATION_S:.1f} seconds"
            )

        velocities = (
            LocomotionVelocityClient._validate_velocity(
                "forward",
                self.forward_velocity,
                min_abs=0.2,
                max_abs=1.0,
            ),
            LocomotionVelocityClient._validate_velocity(
                "lateral",
                self.lateral_velocity,
                min_abs=0.2,
                max_abs=1.0,
            ),
            LocomotionVelocityClient._validate_velocity(
                "angular",
                self.angular_velocity,
                min_abs=0.1,
                max_abs=1.0,
            ),
        )
        if not self.motion_enabled:
            return velocities
        if self.confirmation != self.CONFIRMATION:
            raise ValueError(
                "confirmation must be exactly I_UNDERSTAND when "
                "motion_enabled is true"
            )
        nonzero_axes = sum(abs(value) >= 0.005 for value in velocities)
        if nonzero_axes > 1:
            raise ValueError(
                "only one of forward_velocity, lateral_velocity, and "
                "angular_velocity may be non-zero"
            )
        return velocities


class LocomotionTestNode(Node):
    """Run one safety-gated, time-limited, single-axis motion test."""

    _SOURCE_NAME = "emo_robot_locomotion_test"
    _REQUEST_TIMEOUT_S = 0.25
    _MAX_ATTEMPTS = 8
    _ALLOWED_ACTIONS = (
        McAction.STAND_DEFAULT,
        McAction.LOCOMOTION_DEFAULT,
    )

    def __init__(self) -> None:
        super().__init__("locomotion_test")
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter("confirmation", "")
        self.declare_parameter("forward_velocity", 0.0)
        self.declare_parameter("lateral_velocity", 0.0)
        self.declare_parameter("angular_velocity", 0.0)
        self.declare_parameter("duration_s", 2.0)

        self._command = LocomotionTestCommand(
            motion_enabled=bool(
                self.get_parameter("motion_enabled").value
            ),
            confirmation=str(self.get_parameter("confirmation").value),
            forward_velocity=float(
                self.get_parameter("forward_velocity").value
            ),
            lateral_velocity=float(
                self.get_parameter("lateral_velocity").value
            ),
            angular_velocity=float(
                self.get_parameter("angular_velocity").value
            ),
            duration_s=float(self.get_parameter("duration_s").value),
        )
        self._velocities = self._command.validated_velocities()
        self._finished = False
        self._cleanup_started = False
        self._cleanup_success = True
        self._motion_timer = None
        self._mode_state_client: Optional[McActionStateClient] = None
        self._motion: Optional[LocomotionVelocityClient] = None

        if not self._command.motion_enabled:
            self.get_logger().warning(
                "Locomotion test is locked: motion_enabled=false. "
                "No input source was registered and no velocity was "
                "published."
            )
            self._finished = True
            return

        self._mode_state_client = McActionStateClient(
            self,
            request_timeout_s=self._REQUEST_TIMEOUT_S,
            max_attempts=self._MAX_ATTEMPTS,
        )
        self._motion = LocomotionVelocityClient(
            self,
            source_name=self._SOURCE_NAME,
            publish_period_s=0.02,
            request_timeout_s=self._REQUEST_TIMEOUT_S,
            max_attempts=self._MAX_ATTEMPTS,
        )
        self.get_logger().warning(
            "REAL ROBOT MOTION ARMED. The node will verify "
            "STAND_DEFAULT/RUNNING or LOCOMOTION_DEFAULT/RUNNING "
            "before registering its input source."
        )

    @property
    def finished(self) -> bool:
        return self._finished

    def start(self) -> None:
        """Begin the asynchronous mode-check and registration sequence."""
        if self._finished or self._cleanup_started:
            return
        self._mode_state_client.get_state_async(
            self._on_mode_state,
        )

    def _on_mode_state(
        self,
        success: bool,
        state: Optional[McActionState],
        error: str,
    ) -> None:
        if self._cleanup_started:
            return
        if not success or state is None:
            self._abort(error or "GetMcAction query failed")
            return
        if not any(
            state.is_running(action)
            for action in self._ALLOWED_ACTIONS
        ):
            self._abort(
                "Robot is not STAND_DEFAULT/RUNNING or "
                "LOCOMOTION_DEFAULT/RUNNING: "
                f"action={state.action}, status={state.status}, "
                f"code={state.response_code}"
            )
            return

        self.get_logger().info(
            "Robot mode verified for locomotion: "
            f"{state.description}/RUNNING."
        )
        self._motion.register_input_source_async(
            done_callback=self._on_input_source_registered,
        )

    def _on_input_source_registered(self, success: bool) -> None:
        if self._cleanup_started:
            if success:
                self._motion.emergency_stop()
                self._motion.unregister_input_source_async(
                    done_callback=self._finish_cleanup,
                )
            else:
                self._finish_cleanup(False)
            return
        if not success:
            self._abort("Locomotion input source registration failed")
            return

        forward, lateral, angular = self._velocities
        self._motion.set_velocity(forward, lateral, angular)
        self._motion.start_publishing()
        self._motion_timer = self.create_timer(
            self._command.duration_s,
            self._on_motion_timeout,
        )
        self.get_logger().warning(
            "Motion started: "
            f"forward={forward:.2f} m/s, "
            f"lateral={lateral:.2f} m/s, "
            f"angular={angular:.2f} rad/s, "
            f"duration={self._command.duration_s:.2f}s"
        )

    def _on_motion_timeout(self) -> None:
        self._cancel_motion_timer()
        self.request_stop("configured duration elapsed")

    def _cancel_motion_timer(self) -> None:
        timer = self._motion_timer
        if timer is None:
            return
        self._motion_timer = None
        timer.cancel()
        try:
            self.destroy_timer(timer)
        except Exception:
            pass

    def _abort(self, reason: str) -> None:
        self.get_logger().error(f"Locomotion test refused: {reason}")
        self.request_stop(reason, success=False)

    def request_stop(self, reason: str, success: bool = True) -> None:
        """Start idempotent zero-speed and input-source cleanup."""
        if self._cleanup_started:
            return
        self._cleanup_started = True
        self._cleanup_success = success
        if self._mode_state_client is not None:
            self._mode_state_client.cancel()
        self._cancel_motion_timer()

        self.get_logger().warning(f"Stopping locomotion test: {reason}")
        if self._motion is None:
            self._finish_cleanup(True)
            return

        self._motion.emergency_stop()
        if self._motion.registered:
            self._motion.unregister_input_source_async(
                done_callback=self._finish_cleanup,
            )
        elif self._motion.registering:
            self.get_logger().warning(
                "Waiting for in-flight input-source registration before "
                "final cleanup."
            )
        else:
            self._finish_cleanup(True)

    def _finish_cleanup(self, success: bool) -> None:
        if self._finished:
            return
        self._finished = True
        self._cleanup_success = self._cleanup_success and success
        if self._cleanup_success:
            self.get_logger().info(
                "Locomotion test stopped and input source released."
            )
        else:
            self.get_logger().error(
                "Locomotion cleanup did not complete successfully; "
                "the 1000 ms input-source timeout remains the fallback."
            )

    def destroy_node(self):
        if self._mode_state_client is not None:
            self._mode_state_client.cancel()
        if self._motion is not None and not self._cleanup_started:
            self._motion.emergency_stop()
            if self._motion.registered:
                self._motion.unregister_input_source_async()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    received_signal = {"number": None}
    previous_handlers = {}

    def _signal_handler(signum, _frame) -> None:
        received_signal["number"] = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _signal_handler)

        node = LocomotionTestNode()
        node.start()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
            signum = received_signal["number"]
            if signum is not None:
                received_signal["number"] = None
                node.request_stop(f"received signal {signum}")
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Locomotion test exception: {exc}")
            node.request_stop("unhandled exception", success=False)
    finally:
        if node is not None and not node.finished:
            node.request_stop("node shutdown")
            deadline = time.monotonic() + 3.0
            while (
                rclpy.ok()
                and not node.finished
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(node, timeout_sec=0.05)
        if node is not None:
            node.destroy_node()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

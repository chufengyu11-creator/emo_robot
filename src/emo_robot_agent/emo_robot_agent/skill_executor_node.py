"""ROS 2 JSON plan executor for the minimal language-agent demo."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from emo_robot_agent.executor import SkillExecutor
from emo_robot_agent.plan import SkillPlan
from emo_robot_agent.runtime_config import RuntimeModes
from emo_robot_agent.skill_catalog import MOTION_SKILLS
from emo_robot_agent.speech_feedback import (
    RealSpeechFeedback,
    SimulationSpeechFeedback,
)
from emo_robot_agent.skills import (
    ApproachPlaceholderSkill,
    IntroducePlaceholderSkill,
    SimulationApproachSkill,
    SimulationMotionSkill,
    SkillRegistry,
    PresetMotionSkill,
)


class SkillExecutorNode(Node):
    """Execute approach/wave/introduce plans sequentially."""

    def __init__(self) -> None:
        super().__init__("agent")
        self.declare_parameter("plan_topic", "/emo_robot/agent/skill_plan")
        self.declare_parameter("status_topic", "/emo_robot/agent/status")
        self.declare_parameter("motion_mode", "simulation")
        self.declare_parameter("tts_mode", "real")
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter("confirmation", "")
        for spec in MOTION_SKILLS:
            self.declare_parameter(
                f"motion_wait_{spec.name}_s",
                spec.wait_s,
            )
        self.declare_parameter("tts_domain", "emo_robot")

        modes = RuntimeModes.create(
            str(self.get_parameter("motion_mode").value),
            str(self.get_parameter("tts_mode").value),
            bool(self.get_parameter("motion_enabled").value),
            str(self.get_parameter("confirmation").value),
        )
        self._motion_mode = modes.motion_mode
        self._tts_mode = modes.tts_mode
        self._motion_enabled = modes.motion_enabled
        registry = self._build_motion_registry()
        registry.register(
            "introduce", IntroducePlaceholderSkill(self.get_logger())
        )
        speech_feedback = self._build_speech_feedback()
        if self._motion_mode == "simulation":
            self.get_logger().warning(
                "Agent motion_mode=simulation: AimDK preset motions "
                "will not be created or called."
            )
        if self._tts_mode == "real":
            self.get_logger().warning("Agent tts_mode=real: X2 TTS is enabled.")

        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self._executor = SkillExecutor(
            registry,
            self._publish_status,
            speech_feedback.execute,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("plan_topic").value),
            self._on_plan,
            10,
        )
        self.get_logger().info(
            f"Skill executor ready: motion_mode={self._motion_mode}, "
            f"tts_mode={self._tts_mode}, "
            f"skills={','.join(registry.names)}"
        )

    def _build_motion_registry(self) -> SkillRegistry:
        if self._motion_mode == "real":
            return self._build_real_motion_registry()
        registry = SkillRegistry()
        registry.register(
            "approach",
            SimulationApproachSkill(self.get_logger()),
        )
        for spec in MOTION_SKILLS:
            registry.register(
                spec.name,
                SimulationMotionSkill(self.get_logger(), spec),
            )
        return registry

    def _build_real_motion_registry(self) -> SkillRegistry:
        # Lazy imports guarantee simulation mode never initializes SDK clients.
        from aimdk_msgs.msg import McAction

        from emo_robot_motion import McActionStateClient, PresetMotionClient

        motion_client = None
        state_client = None
        if self._motion_enabled:
            motion_client = PresetMotionClient(self)
            state_client = McActionStateClient(self)
            self.get_logger().warning(
                "REAL PRESET MOTION IS ARMED. Every skill requires "
                "STAND_DEFAULT/RUNNING before AimDK is called."
            )

        registry = SkillRegistry()
        registry.register(
            "approach",
            ApproachPlaceholderSkill(self.get_logger()),
        )
        for spec in MOTION_SKILLS:
            registry.register(
                spec.name,
                PresetMotionSkill(
                    node=self,
                    logger=self.get_logger(),
                    spec=spec,
                    motion_client=motion_client,
                    state_client=state_client,
                    stand_action=McAction.STAND_DEFAULT,
                    wait_s=float(
                        self.get_parameter(
                            f"motion_wait_{spec.name}_s"
                        ).value
                    ),
                ),
            )
        return registry

    def _build_speech_feedback(self):
        if self._tts_mode == "simulation":
            return SimulationSpeechFeedback(self.get_logger())
        # TTS is independent from motion mode and never arms robot movement.
        from emo_robot_speech import SpeechClient

        return RealSpeechFeedback(
            SpeechClient(self),
            domain=str(self.get_parameter("tts_domain").value),
        )

    def _status_data(self, skill: str, state: str, detail: str) -> str:
        return json.dumps(
            {
                "motion_mode": self._motion_mode,
                "tts_mode": self._tts_mode,
                "skill": skill,
                "state": state,
                "detail": detail,
            },
            ensure_ascii=False,
        )

    def _publish_status(self, skill: str, state: str, detail: str) -> None:
        message = String()
        message.data = self._status_data(skill, state, detail)
        self._status_publisher.publish(message)
        self.get_logger().info(message.data)

    def _on_plan(self, message: String) -> None:
        if self._executor.busy:
            detail = "executor is busy"
            self.get_logger().warning(f"Ignoring plan: {detail}.")
            status = String()
            status.data = self._status_data("", "plan_failed", detail)
            self._status_publisher.publish(status)
            return
        try:
            plan = SkillPlan.from_json(message.data)
        except Exception as exc:
            self.get_logger().error(f"Invalid skill plan: {exc}")
            return
        self.get_logger().info(f"Executing plan: {plan.to_json()}")
        self._executor.execute(plan, self._on_plan_done)

    def _on_plan_done(self, success: bool, detail: str) -> None:
        message = String()
        message.data = self._status_data(
            "",
            "plan_succeeded" if success else "plan_failed",
            detail,
        )
        self._status_publisher.publish(message)
        log = self.get_logger().info if success else self.get_logger().error
        log(message.data)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkillExecutorNode()
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

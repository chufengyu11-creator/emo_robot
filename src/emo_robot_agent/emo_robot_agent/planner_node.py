"""ROS 2 text-to-plan node for the minimal language-agent demo."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from emo_robot_agent.planner import (
    DemoPlanner,
    TcpJsonPromptClient,
    TcpLlmPlanner,
)


class PlannerNode(Node):
    """Turn text commands into validated JSON skill sequences."""

    def __init__(self) -> None:
        super().__init__("language_planner")
        self.declare_parameter("input_topic", "/emo_robot/agent/text_input")
        self.declare_parameter("plan_topic", "/emo_robot/agent/skill_plan")
        self.declare_parameter("backend", "tcp_llm")
        self.declare_parameter("llm_host", "127.0.0.1")
        self.declare_parameter("llm_port", 8766)
        self.declare_parameter("llm_timeout_s", 15.0)
        self.declare_parameter("offline_fallback_enabled", True)

        backend = str(self.get_parameter("backend").value)
        self._fallback_planner = DemoPlanner()
        self._fallback_enabled = bool(
            self.get_parameter("offline_fallback_enabled").value
        )
        if backend == "tcp_llm":
            self._planner = TcpLlmPlanner(
                TcpJsonPromptClient(
                    host=str(self.get_parameter("llm_host").value),
                    port=int(self.get_parameter("llm_port").value),
                    timeout_s=float(
                        self.get_parameter("llm_timeout_s").value
                    ),
                )
            )
        elif backend == "demo":
            self._planner = self._fallback_planner
            self.get_logger().warning("Explicit offline planner mode enabled.")
        else:
            raise ValueError(f"unsupported planner backend: {backend!r}")

        self._publisher = self.create_publisher(
            String,
            str(self.get_parameter("plan_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("input_topic").value),
            self._on_text,
            10,
        )
        self.get_logger().info(f"Language planner ready (backend={backend}).")

    def _on_text(self, message: String) -> None:
        try:
            plan = self._planner.plan(message.data)
        except Exception as exc:
            fallback_unavailable = (
                not self._fallback_enabled
                or self._planner is self._fallback_planner
            )
            if fallback_unavailable:
                self.get_logger().error(f"Planning failed: {exc}")
                return
            self.get_logger().warning(
                f"LLM planning failed ({exc}); using offline fallback."
            )
            try:
                plan = self._fallback_planner.plan(message.data)
            except Exception as fallback_exc:
                self.get_logger().error(
                    f"Offline fallback also failed: {fallback_exc}"
                )
                return
        output = String()
        output.data = plan.to_json()
        self._publisher.publish(output)
        self.get_logger().info(f"Plan: {output.data}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
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

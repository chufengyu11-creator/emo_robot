"""Launch the YAML-configured language planner and skill executor."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nodes(context, config_path, asr_config_path):
    """Load YAML defaults and apply only explicit non-empty overrides."""
    planner_overrides = {}
    executor_overrides = {}

    planner_backend = LaunchConfiguration("planner_backend").perform(context)
    if planner_backend.strip():
        planner_overrides["backend"] = planner_backend.strip()

    llm_host = LaunchConfiguration("llm_host").perform(context)
    if llm_host.strip():
        planner_overrides["llm_host"] = llm_host.strip()

    llm_port = LaunchConfiguration("llm_port").perform(context)
    if llm_port.strip():
        planner_overrides["llm_port"] = int(llm_port.strip())

    motion_mode = LaunchConfiguration("motion_mode").perform(context)
    if motion_mode.strip():
        executor_overrides["motion_mode"] = motion_mode.strip()

    tts_mode = LaunchConfiguration("tts_mode").perform(context)
    if tts_mode.strip():
        executor_overrides["tts_mode"] = tts_mode.strip()

    motion_enabled = LaunchConfiguration("motion_enabled").perform(context)
    if motion_enabled.strip():
        normalized = motion_enabled.strip().lower()
        values = {"true": True, "false": False, "1": True, "0": False}
        if normalized not in values:
            raise ValueError(
                "motion_enabled must be true, false, 1, or 0"
            )
        executor_overrides["motion_enabled"] = values[normalized]

    confirmation = LaunchConfiguration("confirmation").perform(context)
    if confirmation.strip():
        executor_overrides["confirmation"] = confirmation.strip()

    planner_parameters = [config_path]
    if planner_overrides:
        planner_parameters.append(planner_overrides)

    executor_parameters = [config_path]
    if executor_overrides:
        executor_parameters.append(executor_overrides)

    nodes = [
        Node(
            package="emo_robot_agent",
            executable="language_planner",
            name="language_planner",
            output="screen",
            parameters=planner_parameters,
        ),
        Node(
            package="emo_robot_agent",
            executable="skill_executor",
            name="agent",
            output="screen",
            parameters=executor_parameters,
        ),
    ]
    enable_asr = LaunchConfiguration("enable_asr").perform(context)
    normalized = enable_asr.strip().lower()
    values = {"true": True, "false": False, "1": True, "0": False}
    if normalized not in values:
        raise ValueError("enable_asr must be true, false, 1, or 0")
    if values[normalized]:
        nodes.insert(
            0,
            Node(
                package="emo_robot_asr",
                executable="vad_asr",
                name="vad_asr",
                output="screen",
                parameters=[asr_config_path],
            ),
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    config_path = os.path.join(
        get_package_share_directory("emo_robot_agent"),
        "config",
        "agent.yaml",
    )
    asr_config_path = os.path.join(
        get_package_share_directory("emo_robot_asr"),
        "config",
        "asr.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_asr",
                default_value="true",
                description="Start AimDK VAD to remote ASR bridge.",
            ),
            DeclareLaunchArgument(
                "planner_backend",
                default_value="",
                description="Optional YAML override: demo or tcp_llm.",
            ),
            DeclareLaunchArgument(
                "motion_mode",
                default_value="",
                description="Optional preset-motion mode: simulation or real.",
            ),
            DeclareLaunchArgument(
                "tts_mode",
                default_value="",
                description="Optional TTS mode: simulation or real.",
            ),
            DeclareLaunchArgument(
                "motion_enabled",
                default_value="",
                description="Optional YAML override for AimDK preset motion.",
            ),
            DeclareLaunchArgument(
                "confirmation",
                default_value="",
                description=(
                    "Required acknowledgement for real preset motion: "
                    "I_UNDERSTAND."
                ),
            ),
            DeclareLaunchArgument(
                "llm_host",
                default_value="",
                description="Optional YAML override for TCP LLM host.",
            ),
            DeclareLaunchArgument(
                "llm_port",
                default_value="",
                description="Optional YAML override for TCP LLM port.",
            ),
            OpaqueFunction(
                function=_launch_nodes,
                args=[config_path, asr_config_path],
            ),
        ]
    )

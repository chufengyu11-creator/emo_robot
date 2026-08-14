from glob import glob
import os

from setuptools import setup


package_name = "emo_robot_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, f"{package_name}.skills"],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="agi",
    maintainer_email="agi@todo.todo",
    description=(
        "Language planner and allow-listed skill executor for emo_robot."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "language_planner = emo_robot_agent.planner_node:main",
            "skill_executor = emo_robot_agent.skill_executor_node:main",
        ],
    },
)

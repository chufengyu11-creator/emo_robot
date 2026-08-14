from glob import glob
import os

from setuptools import setup


package_name = "emo_robot_asr"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools", "requests"],
    zip_safe=True,
    maintainer="agi",
    maintainer_email="agi@todo.todo",
    description="AimDK VAD microphone to remote ASR bridge.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vad_asr = emo_robot_asr.vad_asr_node:main",
        ],
    },
)

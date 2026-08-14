import os
from glob import glob

from setuptools import find_packages, setup

package_name = "emo_robot_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
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
    maintainer_email="agi@example.com",
    description="Launch and configuration files for emo_robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "approach_rosout_logger = "
            "emo_robot_bringup.approach_rosout_logger:main",
        ],
    },
)

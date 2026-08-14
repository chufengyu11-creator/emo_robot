from glob import glob

from setuptools import find_packages, setup

package_name = "emo_robot_perception"

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
            "share/" + package_name + "/models",
            sorted(glob("models/*.pt") + glob("models/*.onnx")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="agi",
    maintainer_email="agi@example.com",
    description="Gesture detection and person tracking nodes for emo_robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "usb_camera = "
            "emo_robot_perception.usb_camera_node:main",
            "lidar_subscriber = "
            "emo_robot_perception.lidar_subscriber_node:main",
            "lidar_target_detector = "
            "emo_robot_perception.lidar_target_detector_node:main",
            "gesture_detector = "
            "emo_robot_perception.gesture_detector_node:main",
            "person_tracker = "
            "emo_robot_perception.person_tracker_node:main",
        ],
    },
)

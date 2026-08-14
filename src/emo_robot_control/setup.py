from setuptools import find_packages, setup

package_name = "emo_robot_control"

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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="agi",
    maintainer_email="agi@example.com",
    description="Safe approach state machine and X2 motion adapter.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "approach_controller = "
            "emo_robot_control.approach_controller_node:main",
            "interaction_responder = "
            "emo_robot_control.interaction_responder_node:main",
        ],
    },
)

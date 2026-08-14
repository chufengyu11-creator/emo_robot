from setuptools import find_packages, setup

package_name = "emo_robot_speech"

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
    description="Reusable TTS and audio playback client for emo_robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [],
    },
)

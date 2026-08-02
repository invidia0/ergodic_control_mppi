from glob import glob
from setuptools import find_packages, setup


package_name = "ergodic_control_mppi_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.rviz")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    tests_require=["pytest"],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mattia Mantovani",
    maintainer_email="mattia@example.com",
    description="ROS 2 launch and target-density visualization for ergodic_control_mppi.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "density_visualizer = ergodic_control_mppi_ros.density_visualizer:main",
            "ergodic_controller = ergodic_control_mppi_ros.ergodic_controller:main",
            "map_adapter = ergodic_control_mppi_ros.map_adapter:main",
            "recorder = ergodic_control_mppi_ros.recorder:main",
            "safety_shield = ergodic_control_mppi_ros.safety_shield:main",
        ],
    },
)

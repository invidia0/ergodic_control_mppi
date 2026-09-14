from setuptools import setup


package_name = "ergodic_mission"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mattia Mantovani",
    maintainer_email="mattia@example.com",
    description="Ergodic MPPI mission framework for the MULLET drone stack.",
    license="MIT",
    entry_points={"console_scripts": ["mission_node = ergodic_mission.mission_node:main"]},
)

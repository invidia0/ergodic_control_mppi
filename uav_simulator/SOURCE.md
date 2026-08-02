# Vendor source

This directory was copied from `src/uav_simulator/` in
[ZJU-FAST-Lab/ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)
at commit `23a8d5a191711dd65633df689bd00f55d4dea8f9`.

The vendored code is licensed under GPL-3.0; see `LICENSE`.

Local integration changes:

- `so3_quadrotor_simulator/launch/simulator_example.launch.py` can suppress
  its RViz process, defaults to a lower flight altitude, and uses the scene's
  black drone color.
- `so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp` publishes odometry
  in the canonical `world` frame.
- `mockamap` publishes its generated static map once with transient-local QoS
  so late subscribers receive it without periodic redraws; the combined
  scene uses a four-metre Perlin ceiling.
- `so3_control/src/control_example.cpp` uses a `0.75 m` demo target altitude.

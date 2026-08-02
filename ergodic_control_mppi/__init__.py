"""Flow-matching MPPI for single-robot ergodic coverage."""

import os

# XLA:GPU autotunes GEMM kernel choice by timing candidates at *compile* time, so a machine
# under load picks different kernels, sums in a different order, and returns different
# float32 results. Runs stay bit-identical within one process, which is what makes this so
# easy to miss: it only shows up when the same config is compared across two processes.
# The closed loop amplifies the difference until whole modes are visited or not -- one
# measured pair differed by 16 m of travel and by whether all three modes were reached.
# https://openxla.org/xla/determinism. Costs ~6% runtime, which is worth paying to make a
# repeated experiment mean something. Appended, so an explicit XLA_FLAGS still wins.
_DETERMINISM = "--xla_gpu_autotune_level=0"
if "xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_DETERMINISM}".strip()

from ergodic_control_mppi.config import AppConfig, load_config  # noqa: E402
from ergodic_control_mppi.simulation import SimulationResult, run_simulation  # noqa: E402

__all__ = ["AppConfig", "SimulationResult", "load_config", "run_simulation"]

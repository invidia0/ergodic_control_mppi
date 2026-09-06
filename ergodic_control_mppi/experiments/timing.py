"""Per-stage timing of one MPPI step, and cost scaling in K, T, P, Q.

The closed loop is a single ``jax.jit`` around a single ``lax.scan``
(simulation.py:91), so per-stage cost cannot be recovered by wrapping Python
calls. Two independent measurements, reported side by side:

1. Micro-benchmarks -- each stage jitted on its own and timed with an explicit
   ``block_until_ready()``. Their sum is compared against the jitted whole step;
   the difference is reported as ``residual``, not absorbed.
2. End-to-end differencing -- ``memory_length=2`` makes the occupancy KDE
   negligible *without* removing the code path (``memory_gain`` is a runtime
   value, so XLA cannot eliminate it), isolating the shared rollout cost.

A reporting script, never a test gate.

    python -m ergodic_control_mppi.experiments.timing --device gpu
    python -m ergodic_control_mppi.experiments.timing --scaling --device gpu
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.core import (
    _rollouts,
    mppi_step,
    sample_epsilon,
)
from ergodic_control_mppi.mppi.field import (
    kde_repulsion,
    memory_flow,
    score_pdf,
)

DEFAULT_CONFIG = "configs/mppi_params.yaml"


def _time(function: Callable[[], Any], repeats: int = 200, warmup: int = 5) -> dict[str, float]:
    """Median and IQR of the wall time of ``function``, in milliseconds."""
    for _ in range(warmup):
        jax.block_until_ready(function())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(function())
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "ms_median": statistics.median(samples),
        "ms_iqr": samples[int(0.75 * len(samples))] - samples[int(0.25 * len(samples))],
        "ms_min": samples[0],
        "repeats": repeats,
    }


def _setup(config, device: str):
    """Device-resident params and one representative step input."""
    from ergodic_control_mppi.simulation import random_state, select_device

    selected = select_device(device)
    params = jax.device_put(config.controller, selected)
    key = jax.random.PRNGKey(config.run.seed)
    simulation_key, state_key = jax.random.split(key)
    state = random_state(state_key, params)
    controls = jax.device_put(
        jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32), selected
    )
    # A memory buffer that looks like one mid-run: a noisy arc, not P copies of
    # a point (identical points would make the KDE degenerate and unrepresentative).
    length = params.mppi.memory_length
    angles = jnp.linspace(0.0, 6.0, length)
    memory = jax.device_put(
        jnp.stack((3.0 * jnp.cos(angles), 3.0 * jnp.sin(angles)), axis=-1).astype(jnp.float32),
        selected,
    )
    temperature = jnp.asarray(params.mppi.temperature, dtype=jnp.float32)
    return params, state, controls, simulation_key, temperature, memory, selected


def measure_stages(
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "auto",
    repeats: int = 200,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Time each stage of one MPPI step plus the whole step."""
    config = _loaded(config_path, overrides)
    params, state, controls, key, temperature, memory, _ = _setup(config, device)
    field = params.field
    workspace = params.workspace

    epsilon, _ = sample_epsilon(key, params)
    epsilon = jax.block_until_ready(epsilon)

    # Reproduce core.py's per-step derived quantities so the attraction and memory
    # stages are timed on the inputs they actually see.
    rollouts_fn = jax.jit(lambda p, s, c, e, t: _rollouts(p, s, c, e, t))
    _, _, sampled_positions = jax.block_until_ready(
        rollouts_fn(params, state, controls, epsilon, temperature)
    )
    initial = jnp.broadcast_to(state[:2], (params.mppi.samples, 1, 2))
    evaluation = jnp.concatenate((initial, sampled_positions[:, :-1]), axis=1)
    source = jnp.median(evaluation, axis=0)
    ages = jnp.arange(memory.shape[0])[::-1]
    recency = field.memory_decay ** ages
    ones = jnp.ones((source.shape[0],), dtype=source.dtype)
    density_floor = 1.0 / (
        (workspace.x_limits[1] - workspace.x_limits[0])
        * (workspace.y_limits[1] - workspace.y_limits[0])
    )

    sample_fn = jax.jit(lambda k, p: sample_epsilon(k, p)[0])
    # The attraction is pointwise now, so it is O(T) rather than the Stein path's O(T^2).
    # The quadratic block moved to the plan self-repulsion, which is timed separately.
    attraction_fn = jax.jit(score_pdf)
    plan_fn = jax.jit(kde_repulsion)
    memory_fn = jax.jit(memory_flow)
    step_fn = jax.jit(mppi_step)

    stages = {
        "sample_epsilon": _time(lambda: sample_fn(key, params), repeats),
        "rollouts_KT": _time(
            lambda: rollouts_fn(params, state, controls, epsilon, temperature), repeats
        ),
        "attraction_T": _time(lambda: attraction_fn(source, params.gmm), repeats),
        "plan_T2": _time(
            lambda: plan_fn(source, source, ones, field.fine_bandwidth), repeats
        ),
        "memory_P2": _time(
            lambda: memory_fn(source, memory, recency, params.gmm, field, density_floor),
            repeats,
        ),
        "mppi_step_total": _time(
            lambda: step_fn(params, controls, state, key, temperature, memory), repeats
        ),
    }

    accounted = sum(
        stages[name]["ms_median"]
        for name in ("sample_epsilon", "rollouts_KT", "attraction_T", "plan_T2", "memory_P2")
    )
    total = stages["mppi_step_total"]["ms_median"]
    return {
        "shape": {
            "K": int(params.mppi.samples),
            "T": int(params.mppi.horizon),
            "P": int(params.mppi.memory_length),
        },
        "device": str(jax.devices()[0]) if device != "cpu" else "cpu",
        "stages": stages,
        "accounted_ms": accounted,
        "total_ms": total,
        # Positive residual = fusion/overhead the isolated stages miss; negative
        # = XLA fuses stages more cheaply together than apart. Reported, not hidden.
        "residual_ms": total - accounted,
        "residual_pct": 100.0 * (total - accounted) / total if total > 0 else float("nan"),
    }


def _loaded(config_path: str | Path, overrides: dict[str, Any] | None):
    """Load a config, optionally patching dotted keys first."""
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if overrides:
        from ergodic_control_mppi.experiments.ablation import _set_dotted

        data = copy.deepcopy(data)
        for dotted, value in overrides.items():
            _set_dotted(data, dotted, value)
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory) / "timing_config.yaml"
        scratch.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return load_config(scratch)


def measure_scaling(
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "auto",
    repeats: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Per-step cost against each of K, T and P, one axis at a time.

    P is swept through ``mppi.memory_length`` directly rather than through
    ``memory_time``, so gamma is held fixed and only the truncation moves.
    """
    sweeps = {
        "K": ("mppi.K", [125, 250, 500, 1000, 2000, 4000]),
        "T": ("mppi.T", [50, 100, 150, 200, 350, 500, 700]),
        "P": ("mppi.memory_length", [500, 1000, 2000, 3000, 4500, 6000]),
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for axis, (dotted, levels) in sweeps.items():
        rows = []
        for level in levels:
            report = measure_stages(config_path, device, repeats, {dotted: level})
            rows.append(
                {
                    "level": level,
                    "total_ms": report["total_ms"],
                    "memory_ms": report["stages"]["memory_P2"]["ms_median"],
                    "rollouts_ms": report["stages"]["rollouts_KT"]["ms_median"],
                    "attraction_ms": report["stages"]["attraction_T"]["ms_median"],
                    "plan_ms": report["stages"]["plan_T2"]["ms_median"],
                    "shape": report["shape"],
                }
            )
            print(
                f"  {axis}={level:<5} total {report['total_ms']:7.3f} ms  "
                f"memory {rows[-1]['memory_ms']:7.3f}  rollouts {rows[-1]['rollouts_ms']:7.3f}",
                flush=True,
            )
        out[axis] = rows
    return out


def measure_endtoend(config_path: str | Path = DEFAULT_CONFIG, device: str = "auto",
                     steps: int = 2000, repeats: int = 200) -> dict[str, Any]:
    """Measure warmed synchronous controller calls, including transfer of the applied control.

    Args:
        config_path: Frozen configuration for this session.
        device: Requested execution device.
        steps: Untimed warmup steps on each measured configuration.
        repeats: Number of synchronized measured calls, excluding compilation.
    """
    from ergodic_control_mppi.mppi.single import initialize_single, single_step

    out = {"warmup_steps": steps, "repeats": repeats,
           "scope": "single_step plus host transfer of applied control; compilation excluded"}
    for label, overrides in (("with_memory", {}), ("memory_length_2", {"mppi.memory_length": 2})):
        config = _loaded(config_path, overrides)
        params, state, controls, key, _, _, selected = _setup(config, device)
        carry = initialize_single(params, state, controls, key)
        advance = jax.jit(single_step)

        def synchronous_step():
            nonlocal carry
            carry, result = advance(params, carry)
            return np.asarray(result.control)

        measured = _time(synchronous_step, repeats=repeats, warmup=steps)
        out[label] = {**measured, "ms_per_step": measured["ms_median"],
                      "P": params.mppi.memory_length, "device": str(selected)}
    return out


def power_state() -> dict:
    """Record available mains, GPU, driver and power information without changing settings."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.draw,power.limit,pstate,temperature.gpu,clocks.sm",
                             "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return {"gpu": result.stdout.strip(), "gpu_query_status": result.returncode,
            "mains": {str(p): p.read_text().strip()
                      for p in Path("/sys/class/power_supply").glob("*/online")}}


def _timing_output(output: Path, record: dict, overwrite: bool) -> tuple[Path, str]:
    """Prepare a staged overwrite so a failed measurement preserves the last report."""
    from ergodic_control_mppi.experiments.common import ensure_bundle

    staged = output.with_name(f".{output.stem}.pending{output.suffix}") if overwrite else output
    return staged, ensure_bundle(staged, record, overwrite)


def _commit_timing_output(staged: Path, output: Path) -> None:
    """Atomically publish a completed staged report and its manifest."""
    if staged != output:
        staged.replace(output)
        staged.with_suffix(".manifest.json").replace(output.with_suffix(".manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--scaling", action="store_true", help="also sweep K, T, P, Q")
    parser.add_argument("--endtoend", action="store_true", help="also run the loop cross-check")
    parser.add_argument("--steps", type=int, default=20,
                        help="untimed warmup calls before synchronized end-to-end timing")
    parser.add_argument("--output", type=Path, default=Path("results/campaign/timing/timing.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.steps) < 1:
        parser.error("repeats and warmup steps must be positive")
    from ergodic_control_mppi.experiments.common import execution_record, numerical_record
    from ergodic_control_mppi.simulation import select_device

    staged, digest = _timing_output(args.output, {
        "controller": numerical_record(load_config(args.config).controller),
        "execution": execution_record("ergodic_control_mppi/experiments/timing.py", str(select_device(args.device))),
        "repeats": args.repeats, "warmup_steps": args.steps,
        "scaling": args.scaling, "endtoend": args.endtoend,
    }, args.overwrite)
    if staged.exists():
        print(f"already measured: {args.output}; use --overwrite for a new session")
        return
    power_before = power_state()

    report: dict[str, Any] = {
        "bundle_hash": digest, "power_before": power_before,
        "stages": measure_stages(args.config, args.device, args.repeats)
    }
    breakdown = report["stages"]
    print(f"\nshape {breakdown['shape']}  device {breakdown['device']}")
    for name, values in breakdown["stages"].items():
        share = 100.0 * values["ms_median"] / breakdown["total_ms"]
        print(f"  {name:<18} {values['ms_median']:8.3f} ms  (IQR {values['ms_iqr']:.3f})  {share:5.1f}%")
    print(f"  {'residual':<18} {breakdown['residual_ms']:8.3f} ms  "
          f"({breakdown['residual_pct']:.1f}% of total)")

    if args.scaling:
        print("\nscaling:")
        report["scaling"] = measure_scaling(args.config, args.device, args.repeats)
    if args.endtoend:
        print("\nend-to-end cross-check:")
        report["endtoend"] = measure_endtoend(args.config, args.device, args.steps, args.repeats)
        print(json.dumps(report["endtoend"], indent=2))

    report["power_after"] = power_state()
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _commit_timing_output(staged, args.output)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

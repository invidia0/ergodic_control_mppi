"""Shared scenario, CSV, and summary utilities."""

import csv
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import jax
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.field import pdf
from ergodic_control_mppi.parameters import ControllerParams, RunConfig


@dataclass(frozen=True)
class Scenario:
    """Controller inputs and normalized grid representation of one scenario."""

    name: str
    params: ControllerParams
    run_config: RunConfig
    target_density_grid: np.ndarray
    map_x_limits: tuple[float, float]
    map_y_limits: tuple[float, float]
    obstacle_map: np.ndarray
    safety_radius: float


def build_target_grid(params: ControllerParams, grid_shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    """Evaluate and normalize the configured GMM on a regular grid."""
    ny, nx = grid_shape
    x = jnp.linspace(params.workspace.x_limits[0], params.workspace.x_limits[1], nx)
    y = jnp.linspace(params.workspace.y_limits[0], params.workspace.y_limits[1], ny)
    grid_x, grid_y = jnp.meshgrid(x, y)
    values = np.asarray(pdf(jnp.stack((grid_x, grid_y), axis=-1), params.gmm), dtype=np.float64)
    total = values.sum()
    if total <= 0:
        raise ValueError("target grid mass must be positive")
    return values / total


def load_scenario(
    config_path: str = "configs/mppi_params.yaml",
    scenario_name: str = "yaml_default",
    grid_shape: tuple[int, int] = (80, 80),
    safety_radius: float | None = None,
) -> Scenario:
    """Load one YAML scenario through the package configuration entrypoint."""
    config = load_config(config_path)
    params = config.controller
    workspace = params.workspace
    return Scenario(
        scenario_name,
        params,
        config.run,
        build_target_grid(params, grid_shape),
        tuple(map(float, workspace.x_limits)),
        tuple(map(float, workspace.y_limits)),
        np.asarray(workspace.obstacles),
        float(workspace.safe_distance if safety_radius is None else safety_radius),
    )


def append_csv(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one row, creating its parent and header when needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    new = not output.exists()
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow(row)


def prepare_outputs(paths: Iterable[str | Path], overwrite: bool) -> None:
    """Protect result files from replacement unless explicitly authorized."""
    outputs = [Path(path) for path in paths]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"result already exists: {existing[0]}; pass --overwrite to replace it")
    for path in existing:
        path.unlink()


def summarize(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, float]:
    """Return mean and sample standard deviation columns for scalar metrics."""
    result: dict[str, float] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows])
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    return result


def numerical_record(value):
    """Convert resolved inputs to canonical JSON, hashing arrays including dtype and shape."""
    if is_dataclass(value):
        return {f.name: numerical_record(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict) or isinstance(value, np.lib.npyio.NpzFile):
        return {str(k): numerical_record(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [numerical_record(v) for v in value]
    if hasattr(value, "dtype"):
        array = np.ascontiguousarray(np.asarray(value))
        return {"shape": list(array.shape), "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return value


def fingerprint(value) -> str:
    """Hash resolved numerical inputs without depending on names or file locations."""
    payload = json.dumps(numerical_record(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def execution_record(driver: str, device: str) -> dict:
    """Record numerical source contents, source revision, and execution environment."""
    root = Path(__file__).resolve().parents[2]
    sources = [root / driver]
    sources += sorted((root / "ergodic_control_mppi").rglob("*.py"))
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False).stdout.strip()
    environment = {"python": platform.python_version(), "jax": jax.__version__,
                   "numpy": np.__version__, "machine": platform.machine(),
                   "host": platform.node(), "device": device,
                   "flags": {k: os.environ.get(k, "") for k in (
                       "XLA_FLAGS", "JAX_PLATFORMS", "JAX_ENABLE_X64",
                       "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE")}}
    return {"revision": revision, "sources": source, "source_digest": fingerprint(source),
            "environment": environment, "environment_digest": fingerprint(environment)}


def ensure_bundle(output: Path, record: dict, overwrite: bool = False) -> str:
    """Create or verify an adjacent manifest before any simulation or output mutation.

    Unknown legacy provenance remains readable but cannot satisfy a new run. A fresh
    path or explicit overwrite is required for incompatible results.
    """
    manifest = output.with_suffix(".manifest.json")
    record = numerical_record(record)
    bundle_hash = fingerprint(record)
    expected = {"bundle_hash": bundle_hash, "inputs": record}
    if overwrite:
        prepare_outputs([output, manifest, output.with_name(output.stem + "_paths.npz")], True)
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f"{output}: incompatible bundle; use a fresh path or --overwrite")
    else:
        if output.exists() or output.with_name(output.stem + "_paths.npz").exists():
            raise ValueError(f"{output}: unknown legacy provenance; use a fresh path or --overwrite")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
    return bundle_hash


def verified_rows(path: Path, identity_fields: tuple[str, ...],
                  *, legacy: bool = False) -> list[dict]:
    """Read one bundle and reject mixed provenance and duplicate cell identities.

    Args:
        path: CSV and adjacent manifest location.
        identity_fields: Columns defining one unique observation.
        legacy: Allow a wholly legacy CSV for historical analysis, never for resume.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    hashes = {r.get("bundle_hash", "") for r in rows}
    if hashes == {""} and legacy:
        pass
    elif rows:
        manifest = path.with_suffix(".manifest.json")
        if "" in hashes or len(hashes) != 1 or not manifest.exists():
            raise ValueError(f"{path}: mixed or unknown bundle provenance")
        record = json.loads(manifest.read_text())
        if hashes != {record["bundle_hash"]} or fingerprint(record["inputs"]) != record["bundle_hash"]:
            raise ValueError(f"{path}: bundle manifest does not match rows")
    if rows and hashes != {""}:
        configurations = record["inputs"].get("configurations")
        if configurations is not None:
            for row in rows:
                if "config_hash" in row and row["config_hash"] not in configurations:
                    raise ValueError(f"{path}: row configuration absent from bundle manifest")
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in identity_fields)
        if key in seen:
            raise ValueError(f"{path}: duplicate identity {key}")
        seen.add(key)
    return rows


def artifact_digests(paths: Iterable[Path]) -> dict[str, str]:
    """Hash completed files by streaming, including large trajectory captures."""
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result

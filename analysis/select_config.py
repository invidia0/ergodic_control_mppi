from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from analysis.pareto import compute_pareto_front


def filter_feasible(
    rows: Sequence[dict[str, float | int | str]],
    max_safety_metric: float | None = None,
    max_runtime_ms: float | None = None,
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for row in rows:
        if max_safety_metric is not None and float(row["safety_metric"]) > max_safety_metric:
            continue
        if max_runtime_ms is not None and float(row["runtime_ms"]) > max_runtime_ms:
            continue
        out.append(dict(row))
    return out


def aggregate_across_seeds(
    rows: Sequence[dict[str, float | int | str]],
    metric_cols: Sequence[str],
    group_cols: Sequence[str],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        grouped[key].append(row)

    out: list[dict[str, float | int | str]] = []
    for key, group in grouped.items():
        agg: dict[str, float | int | str] = {group_cols[i]: key[i] for i in range(len(group_cols))}
        agg["num_seeds"] = len(group)
        for col in metric_cols:
            agg[col] = float(np.mean([float(r[col]) for r in group]))
        out.append(agg)
    return out


def _normalize_points(points: np.ndarray) -> np.ndarray:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.where((hi - lo) < 1e-12, 1.0, (hi - lo))
    return (points - lo) / span


def select_knee_point(
    rows: Sequence[dict[str, float | int | str]],
    metric_cols: Sequence[str],
) -> dict[str, float | int | str]:
    """
    Selects a balanced point by minimum distance to normalized utopia point.
    """
    if len(rows) == 0:
        raise ValueError("cannot select knee point from empty rows")
    pareto_rows = compute_pareto_front(rows, metric_cols)
    points = np.asarray(
        [[float(row[col]) for col in metric_cols] for row in pareto_rows],
        dtype=np.float64,
    )
    norm = _normalize_points(points)
    # Minimization objectives => utopia at origin.
    dists = np.linalg.norm(norm, axis=1)
    idx = int(np.argmin(dists))
    return dict(pareto_rows[idx])


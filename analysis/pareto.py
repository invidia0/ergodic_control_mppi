from __future__ import annotations

from typing import Sequence

import numpy as np


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """
    Pareto-efficient mask for minimization objectives.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must have shape (n_points, n_metrics)")
    n = pts.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated_by_i = np.all(pts[i] <= pts, axis=1) & np.any(pts[i] < pts, axis=1)
        efficient[dominated_by_i] = False
        efficient[i] = True
    return efficient


def compute_pareto_front(
    rows: Sequence[dict[str, float | int | str]],
    metric_cols: Sequence[str],
) -> list[dict[str, float | int | str]]:
    if len(rows) == 0:
        return []
    points = np.asarray(
        [[float(row[col]) for col in metric_cols] for row in rows],
        dtype=np.float64,
    )
    mask = is_pareto_efficient(points)
    return [dict(rows[i]) for i in range(len(rows)) if mask[i]]

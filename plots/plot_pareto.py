from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_pareto_scatter(
    csv_path: str = "results/sweeps/open_multimodal.csv",
    x_metric: str = "pairwise_overlap",
    y_metric: str = "team_ergodic_error",
    color_metric: str = "safety_metric",
) -> None:
    rows = _load_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("no rows found for plotting")

    x = np.asarray([float(r[x_metric]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r[y_metric]) for r in rows], dtype=np.float64)
    c = np.asarray([float(r[color_metric]) for r in rows], dtype=np.float64)

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.85)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title("Pareto Scatter")
    cb = plt.colorbar(sc)
    cb.set_label(color_metric)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_pareto_scatter()


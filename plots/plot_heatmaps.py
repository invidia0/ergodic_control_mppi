from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_alpha_ellx_heatmap(
    csv_path: str = "results/sweeps/open_multimodal.csv",
    metric: str = "team_ergodic_error",
) -> None:
    rows = _load_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("no rows found for plotting")

    alpha_vals = sorted({float(r["alpha_cross"]) for r in rows})
    ellx_vals = sorted({float(r["ell_x"]) for r in rows})
    grid = np.full((len(alpha_vals), len(ellx_vals)), np.nan, dtype=np.float64)

    for i, alpha in enumerate(alpha_vals):
        for j, ellx in enumerate(ellx_vals):
            vals = [
                float(r[metric])
                for r in rows
                if float(r["alpha_cross"]) == alpha and float(r["ell_x"]) == ellx
            ]
            if vals:
                grid[i, j] = float(np.mean(vals))

    plt.figure(figsize=(7, 5))
    im = plt.imshow(grid, aspect="auto", origin="lower", cmap="magma")
    plt.xticks(range(len(ellx_vals)), [f"{v:.2f}" for v in ellx_vals])
    plt.yticks(range(len(alpha_vals)), [f"{v:.2f}" for v in alpha_vals])
    plt.xlabel("ell_x")
    plt.ylabel("alpha_cross")
    plt.title(f"Heatmap: {metric}")
    cb = plt.colorbar(im)
    cb.set_label(metric)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_alpha_ellx_heatmap()


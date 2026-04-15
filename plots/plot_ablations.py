from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_runtime_distribution(
    csv_path: str = "results/dars2026/sweeps/open_multimodal.csv",
) -> None:
    rows = _load_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("no rows found for plotting")

    runtime = np.asarray([float(r["runtime_ms"]) for r in rows], dtype=np.float64)
    plt.figure(figsize=(7, 4))
    plt.hist(runtime, bins=20, color="tab:blue", alpha=0.85)
    plt.xlabel("runtime_ms")
    plt.ylabel("count")
    plt.title("Runtime Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_runtime_distribution()


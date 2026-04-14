from __future__ import annotations

import csv
from pathlib import Path

from experiments.config import load_sweep_config
from experiments.run_single_trial import run_single_trial
from experiments.scenarios import load_yaml_scenario


def _append_row(path: Path, row: dict[str, float | int | str], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_sweep(
    sweep_config_path: str = "configs/sweeps/open_multimodal.yaml",
    scenario_config_path: str = "configs/mppi_params.yaml",
    output_csv_path: str = "results/dars2026/sweeps/open_multimodal.csv",
) -> list[dict[str, float | int | str]]:
    """
    Run grid sweep and append results incrementally to CSV.
    """
    sweep_cfg = load_sweep_config(sweep_config_path)
    scenario = load_yaml_scenario(
        config_path=scenario_config_path,
        scenario_name=sweep_cfg.scenario_name,
    )
    seeds = sweep_cfg.iter_seeds()

    all_rows: list[dict[str, float | int | str]] = []
    out_path = Path(output_csv_path)
    fieldnames: list[str] | None = None

    for i, params in enumerate(sweep_cfg.parameter_grid()):
        for seed in seeds:
            print(f"[{i}/{len(list(sweep_cfg.parameter_grid()))}] Running trial with params={params} and seed={seed}...")
            row = run_single_trial(
                scenario=scenario,
                controller_config=params,
                seed=seed,
                team_size=sweep_cfg.team_size,
                steps=scenario.run_config.steps,
            )
            all_rows.append(row)
            if fieldnames is None:
                fieldnames = list(row.keys())
            _append_row(out_path, row, fieldnames=fieldnames)
            print(f"Success.")

    return all_rows


if __name__ == "__main__":
    run_sweep()


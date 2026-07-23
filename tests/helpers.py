"""Small configuration helpers shared by unittest modules."""

from pathlib import Path

import yaml


def write_small_config(directory: Path, *, robots: int = 1, steps: int = 2) -> Path:
    data = yaml.safe_load(Path("configs/mppi_params.yaml").read_text(encoding="utf-8"))
    data["steps"] = steps
    data["robots"]["num_robots"] = robots
    data["mppi"]["K"] = 8
    data["mppi"]["T"] = 4
    data["mppi"]["history_len"] = 3
    output = directory / "small.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output

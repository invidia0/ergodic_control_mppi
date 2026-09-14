"""Small configuration helpers shared by unittest modules."""

from pathlib import Path

import yaml


def write_small_config(directory: Path, *, steps: int = 2) -> Path:
    data = yaml.safe_load(Path("configs/uav_profile.yaml").read_text(encoding="utf-8"))
    data["steps"] = steps
    data["mppi"]["K"] = 8
    data["mppi"]["T"] = 4
    # The deployment profile takes its obstacles from the runtime grid; keep a few circles
    # so the obstacle-cost path stays exercised.
    data["map"]["obstacles"]["num_obstacles"] = 3
    output = directory / "small.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output

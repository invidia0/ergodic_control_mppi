"""Command-line entrypoint for single-robot simulations."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.simulation import run_simulation


def main() -> None:
    """Load configuration, run the selected controller, and optionally plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mppi_params.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Running single-robot simulation on {args.device}...")
    result = run_simulation(config, args.device, progress=args.progress)
    print(f"Done on {result.device}; paths shape={result.paths.shape}.")
    if not args.no_plot:
        from ergodic_control_mppi.plotting.simulation import plot_simulation

        plot_simulation(config, result)


if __name__ == "__main__":
    main()

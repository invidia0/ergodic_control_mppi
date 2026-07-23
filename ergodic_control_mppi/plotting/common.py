"""Small shared helpers for plotting modules."""

from pathlib import Path


def prepare_output(path: str | Path) -> Path:
    """Create an output directory and return its normalized path."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output

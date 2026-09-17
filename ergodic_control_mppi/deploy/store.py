"""
The drone's mission library: every spec the framework accepted, kept as a JSON file.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

# ponytail: no pruning; listing is capped instead. Add retention when the drone's disk fills.
LIST_LIMIT = 50


def save_spec(directory: Path, spec: dict, now: datetime) -> str:
    """
    Store an accepted spec atomically, so a crash never leaves a half-written file.

    Args:
            directory: Library folder; created when missing.
            spec: The spec as loaded; its ``mission_id`` names the file.
            now: UTC time of acceptance, which orders the library.
    Returns:
            The stored name, ``<UTC stamp>_<mission_id>``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(spec["mission_id"]))
    name = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}_{safe_id}"
    temporary = directory / f".{name}.tmp"
    temporary.write_text(json.dumps(spec, indent=2))
    os.replace(temporary, directory / f"{name}.json")
    return name


def stored_specs(directory: Path, limit: int = LIST_LIMIT) -> list[tuple[str, str]]:
    """
    List stored specs, newest first.

    Args:
            directory: Library folder; a missing one is an empty library.
            limit: Most specs to return.
    Returns:
            ``(name, spec_json)`` pairs.
    """
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("*.json"), reverse=True)[:limit]
    return [(path.stem, path.read_text()) for path in files]

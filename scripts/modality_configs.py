"""Emit the modality-sweep profiles and report the log-odds margins they carry.

uv run python scripts/modality_configs.py --check   # margins only, writes nothing
uv run python scripts/modality_configs.py           # write profiles, update freeze.json"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.field import responsibility_gaps

BUNDLE = Path("results/uav/T150")
#: The deployed sigma*. Only used to report where a single scalar kappa would have put each
#: mode instead; the controller reads its own value from the profile.
DEPLOYED_RELEASE = 2.24

#: The one place these geometries are defined. `configs/experiments/*.yaml` carry same-named
#: scenarios on a +-6 workspace; these are on the UAV workspace ([-20,20] x [-10,10]) and are
#: not interchangeable with them.
MODALITY_DENSITIES: dict[str, dict] = {
    # No rival, so Delta = +inf and the demotion is inert by construction (field.py:325-328).
    # This is the arm that exercises the `inf * 0` guard on the real controller.
    "unimodal": {
        "weights": [1.0],
        "means": [[0.0, 0.0]],
        "covariances": [[[14.0, 0.0], [0.0, 5.0]]],
    },
    # The two-mode shuttle the destination bias exists to break (Sec. 3.5). Symmetric, so
    # both margins are equal and per-mode kappa_j degenerates to a single scalar -- the case
    # where the two calibrations should agree exactly.
    "bimodal": {
        "weights": [0.5, 0.5],
        "means": [[-12.0, 0.0], [12.0, 0.0]],
        "covariances": [[[8.0, 0.0], [0.0, 3.0]]] * 2,
    },
    "four_mode": {
        "weights": [0.25] * 4,
        "means": [[-12.0, 5.0], [12.0, 5.0], [-12.0, -5.0], [12.0, -5.0]],
        "covariances": [[[8.0, 0.0], [0.0, 3.0]]] * 4,
    },
    # The point of the sweep: one broad component, two tight ones, and a mid-scale fourth,
    # so the margins span 7.4x (15.45 to 114.85 nats) against the deployed target's 1.7x.
    # Under a single scalar kappa tuned to hold the deepest at 2.24x, the shallowest would
    # release at 1.17x fair share; per-mode kappa_j is what collapses that spread, and this
    # is the only target here where the two calibrations differ at all (`bimodal` and
    # `four_mode` are symmetric, so they agree by construction).
    #
    # The modes are *separated*, not nested. An earlier version placed the tight components
    # inside the broad one's 2-sigma basin: the margins were right but every mode-membership
    # outcome saturated (in_mode_fraction 1.0, one visit per 400 s run) because the broad
    # basin covered 27% of the workspace against the deployed target's 7.6%, and mode centres
    # sat 1.45 Mahalanobis apart -- inside each other. This geometry holds the deployed
    # target's measurable profile (widest basin 9.8% of the workspace, centres 5.35 apart)
    # while widening the margin spread, which is the only thing the sweep needs varied.
    #
    # Tight covariances are 1.0: a 2-sigma basin of 2.00 m, about three track spacings at the
    # deployed h = 0.94, the smallest basin the memory fills rather than merely enters.
    "multiscale": {
        "weights": [0.4, 0.2, 0.2, 0.2],
        "means": [[-13.0, 0.0], [1.0, 6.0], [1.0, -6.0], [14.0, 0.0]],
        "covariances": [
            [[10.0, 0.0], [0.0, 4.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ],
    },
}


def profile(base: dict, density: dict) -> dict:
    """The frozen profile with its density block replaced."""
    return {**base, "density": density}


def margins(spec: dict) -> np.ndarray:
    """Return ``Delta_j`` for one profile, via the loader the campaign itself uses."""
    handle, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(handle, "w") as stream:
            yaml.safe_dump(spec, stream, sort_keys=False)
        return np.asarray(responsibility_gaps(load_config(path).controller.gmm))
    finally:
        os.unlink(path)


def single_kappa_release(gaps: np.ndarray, deployed: float = DEPLOYED_RELEASE) -> np.ndarray:
    """Where each mode would release if ``kappa`` were one scalar tuned for the deepest."""
    return 1.0 + gaps / (np.max(gaps) / (deployed - 1.0))


def _row(values: np.ndarray, places: int = 2) -> str:
    """Format an array without float32 noise widening every printed number."""
    return "[" + ", ".join(f"{v:.{places}f}" for v in values) + "]"


def report(name: str, gaps: np.ndarray) -> None:
    """Print one target's margins and its single-scalar counterfactual."""
    print(f"{name:11s} J={gaps.size}  Delta_j={_row(gaps)}")
    if not np.isfinite(gaps).all():
        # A single infinite margin makes the counterfactual meaningless, not just noisy:
        # there is no deepest mode to tune the scalar against.
        print(f"{'':11s}   no rival: demotion inert, nowhere to be released to")
        return
    released = single_kappa_release(gaps)
    verdict = ("identical -- symmetric target" if np.ptp(released) < 5e-3
               else f"spread {released.min():.2f}x-{released.max():.2f}x")
    print(f"{'':11s}   single-kappa release={_row(released)}  ({verdict}; "
          f"per-mode kappa_j releases all at {DEPLOYED_RELEASE})")


def write(base: dict, output: Path, freeze: Path) -> list[Path]:
    """Emit every modality profile and register its digest in the bundle's freeze list."""
    output.mkdir(parents=True, exist_ok=True)
    digests = json.loads(freeze.read_text(encoding="utf-8"))
    written = []
    for name, density in MODALITY_DENSITIES.items():
        path = output / f"config_{name}.yaml"
        path.write_text(
            yaml.safe_dump(profile(base, density), sort_keys=False), encoding="utf-8"
        )
        digests[str(path.relative_to(freeze.parent))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        written.append(path)
    freeze.write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")
    return written


def self_check(base: dict) -> None:
    """Tie this script to three numbers the manuscript states literally."""
    trimodal = np.asarray(responsibility_gaps(load_config(BUNDLE / "config.yaml").controller.gmm))
    # main.tex:306 -- "the margins are (18.71, 31.09, 31.09) nats".
    assert np.allclose(np.round(trimodal, 2), [18.71, 31.09, 31.09]), trimodal
    # main.tex:809 -- "the shallow mode would go at 1.75x fair share while the deep ones are
    # held to 2.24x". Reproducing it is what says `single_kappa_release` is the same
    # counterfactual the paper argues against.
    assert np.allclose(np.round(single_kappa_release(trimodal), 2), [1.75, 2.24, 2.24])
    # main.tex:813 -- "A component with no rival has Delta_j = infinity". The controller's
    # `inf * 0` guard (field.py:325-328) exists only because this is true.
    assert not np.isfinite(margins(profile(base, MODALITY_DENSITIES["unimodal"]))).any()
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--check", action="store_true",
                        help="report margins and verify, without writing anything")
    args = parser.parse_args()
    base = yaml.safe_load((args.bundle / "config.yaml").read_text(encoding="utf-8"))

    self_check(base)
    report("trimodal", np.asarray(
        responsibility_gaps(load_config(args.bundle / "config.yaml").controller.gmm)))
    for name, density in MODALITY_DENSITIES.items():
        report(name, margins(profile(base, density)))
    if args.check:
        return
    for path in write(base, args.bundle / "modality", args.bundle / "freeze.json"):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

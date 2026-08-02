"""Target-mode visit, dwell, and cycling metrics.

The metrics are experiment independent: mode geometry comes from the mixture itself, so
the same code applies to any configured density. Membership uses Mahalanobis distance
with hysteresis, which keeps a path that lingers on a mode boundary from generating a
burst of spurious visits, and a minimum dwell, which keeps a fast transit through a mode
from counting as coverage of it.
"""

import numpy as np

_UNASSIGNED = -1


def _mahalanobis(positions: np.ndarray, means: np.ndarray, inverses: np.ndarray) -> np.ndarray:
    """Return per-sample Mahalanobis distances with shape ``(N, M)``."""
    offsets = positions[:, None, :] - means[None, :, :]
    squared = np.einsum("nmi,mij,nmj->nm", offsets, inverses, offsets)
    return np.sqrt(np.maximum(squared, 0.0))


def _assign(distances: np.ndarray, enter_sigma: float, exit_sigma: float) -> np.ndarray:
    """Assign each sample to a mode, holding the previous mode until it is released.

    A new mode is entered when its distance drops to ``enter_sigma``; the current mode is
    kept until its distance exceeds ``exit_sigma``. Between the two thresholds the
    assignment is sticky, which is what removes boundary chatter.
    """
    labels = np.full(distances.shape[0], _UNASSIGNED, dtype=np.int64)
    nearest = np.argmin(distances, axis=1)
    current = _UNASSIGNED
    for index in range(distances.shape[0]):
        if current != _UNASSIGNED and distances[index, current] <= exit_sigma:
            labels[index] = current
            continue
        candidate = int(nearest[index])
        current = candidate if distances[index, candidate] <= enter_sigma else _UNASSIGNED
        labels[index] = current
    return labels


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(label, start, length)`` for each maximal constant stretch."""
    if labels.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], boundaries))
    lengths = np.diff(np.concatenate((starts, [labels.size])))
    return [(int(labels[s]), int(s), int(n)) for s, n in zip(starts, lengths)]


def compute_mode_metrics(
    positions: np.ndarray,
    means: np.ndarray,
    covariance_inverses: np.ndarray,
    delta_t: float,
    enter_sigma: float = 2.0,
    exit_sigma: float = 2.5,
    min_dwell: float = 1.0,
) -> dict[str, float]:
    """Summarize how a path visits, dwells in, and cycles through the target modes.

    Args:
        positions: Executed positions with shape ``(N, 2)``.
        means: Mode centers with shape ``(M, 2)``.
        covariance_inverses: Inverse mode covariances with shape ``(M, 2, 2)``.
        delta_t: Control timestep in seconds.
        enter_sigma: Mahalanobis distance at which a mode is entered.
        exit_sigma: Mahalanobis distance at which the current mode is released.
        min_dwell: Seconds a stretch must last before it counts as a visit.

    Returns:
        Mapping with ``mode_visits``, ``mode_switches``, ``mode_revisits``,
        ``mode_dwell_median_s``, ``mode_dwell_total_s``, ``mode_transitions``,
        ``mode_cycles``, ``first_all_modes_s`` (NaN if never reached), and
        ``in_mode_fraction``.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    means = np.asarray(means, dtype=np.float64).reshape(-1, 2)
    inverses = np.asarray(covariance_inverses, dtype=np.float64).reshape(-1, 2, 2)
    mode_count = means.shape[0]
    minimum_samples = max(1, int(np.ceil(min_dwell / delta_t)))

    labels = _assign(_mahalanobis(positions, means, inverses), enter_sigma, exit_sigma)
    # A visit is a qualified stretch: inside one mode, uninterrupted, long enough.
    visits = [
        (label, start, length)
        for label, start, length in _runs(labels)
        if label != _UNASSIGNED and length >= minimum_samples
    ]

    order = [label for label, _, _ in visits]
    dwells = np.array([length * delta_t for _, _, length in visits], dtype=np.float64)
    seen: set[int] = set()
    revisits = 0
    for label in order:
        if label in seen:
            revisits += 1
        seen.add(label)

    # First time every mode has been visited at least once, and how many further full
    # sweeps of all modes complete after that.
    first_all_modes = float("nan")
    pending = set(range(mode_count))
    cycles = 0
    for label, start, length in visits:
        pending.discard(label)
        if pending:
            continue
        if np.isnan(first_all_modes):
            first_all_modes = (start + length) * delta_t
        else:
            cycles += 1
        pending = set(range(mode_count))

    return {
        "mode_visits": float(len(visits)),
        "mode_switches": float(max(0, len(order) - 1)),
        "mode_revisits": float(revisits),
        "mode_dwell_median_s": float(np.median(dwells)) if dwells.size else 0.0,
        "mode_dwell_total_s": float(dwells.sum()),
        "mode_transitions": float(sum(1 for a, b in zip(order, order[1:]) if a != b)),
        "mode_cycles": float(cycles),
        "first_all_modes_s": first_all_modes,
        "in_mode_fraction": (
            float(sum(length for _, _, length in visits) / labels.size) if labels.size else 0.0
        ),
    }

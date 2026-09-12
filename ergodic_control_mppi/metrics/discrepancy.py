"""Squared-MMD occupation error, its RKHS witness, and the descent gap."""

from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray

#: Query rows per kernel block. Keeps the ``(block, support)`` matrix to tens of MB on the
#: ~5,800-cell reachable grids the campaign uses.
_BLOCK = 512


def grid_target(
    density: ArrayLike, mask: ArrayLike, limits_x: tuple[float, float],
    limits_y: tuple[float, float],
) -> tuple[ArrayLike, ArrayLike]:
    """
    Return the support points and weights of the paper's discrete target.
    
    Args:
            density: Target density on the metric grid, ``(rows, columns)``, any positive scale.
            mask: Boolean ``(rows, columns)``, True on the reachable component.
            limits_x: Workspace ``x`` bounds.
            limits_y: Workspace ``y`` bounds.
    Returns:
            ``(support, weights)`` with shapes ``(S, 2)`` and ``(S,)``, the weights summing to
            one over the ``S`` reachable cells.
    """
    density = np.asarray(density, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if density.shape != mask.shape:
        raise ValueError(f"density {density.shape} and mask {mask.shape} must agree")
    rows, columns = density.shape
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(*limits_x, columns), np.linspace(*limits_y, rows)
    )
    keep = mask & (density > 0.0)
    total = float(density[keep].sum())
    if total <= 0.0:
        raise ValueError("the reachable component carries no target mass")
    return (np.column_stack([mesh_x[keep], mesh_y[keep]]),
            density[keep] / total)


def mean_embedding(
    query: ArrayLike, support: ArrayLike, weights: ArrayLike, bandwidth: float
) -> ArrayLike:
    """
    Evaluate ``m_pi(z) = sum_q p_q k(z, c_q)`` at ``(Q, 2)`` queries, as a finite sum.
    
    Args:
            query: Positions with shape ``(Q, 2)``.
            support: Target support points, ``(S, 2)``.
            weights: Target weights, ``(S,)``; must sum to one.
            bandwidth: The kernel's ``h``, positive.
    Returns:
            ``m_pi`` with shape ``(Q,)``.
    """
    query = np.asarray(query, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    support_squared = np.sum(support ** 2, axis=1)
    out = np.empty(query.shape[0])
    for start in range(0, query.shape[0], _BLOCK):
        chunk = query[start:start + _BLOCK]
        squared = (np.sum(chunk ** 2, axis=1)[:, None] - 2.0 * chunk @ support.T
                   + support_squared)
        out[start:start + _BLOCK] = np.exp(-np.maximum(squared, 0.0) / bandwidth) @ weights
    return out


def self_inner(support: ArrayLike, weights: ArrayLike, bandwidth: float) -> float:
    """Return ``||m_pi||^2 = p^T K p``, the same finite sum evaluated on the support."""
    return float(np.asarray(weights) @ mean_embedding(support, support, weights, bandwidth))


def radius_squared(support: ArrayLike, weights: ArrayLike, bandwidth: float) -> float:
    """
    Return the ``R^2 = sup_z ||k(z, .) - m_pi||^2`` the theorem's noise term uses.
    """
    return 1.0 + self_inner(support, weights, bandwidth)


def trivial_bound(support: ArrayLike, weights: ArrayLike, bandwidth: float) -> float:
    """The bound any pair of probability measures satisfies, as the comparator."""
    return radius_squared(support, weights, bandwidth)


def walk(
    positions: ArrayLike, support: ArrayLike, weights: ArrayLike, bandwidth: float,
    step_radius: float | None = None,
) -> dict[str, ArrayLike]:
    """
    Reconstruct ``E_n``, ``delta_n`` and the theorem's bound along one trajectory.
    
    Args:
            positions: Audited path with shape ``(N, 2)``.
            support: Target support points, ``(S, 2)``, from :func:`grid_target`.
            weights: Target weights, ``(S,)``, summing to one.
            bandwidth: The kernel's ``h``.
            step_radius: Optional one-step reach, in workspace units.
    Returns:
            ``n``, ``error``, ``gap``, ``gap_reach`` as ``(N,)`` arrays (the gaps are NaN in the
            final slot, which has no successor), the running theorem ``bound``, and the scalars
            ``radius_squared``, ``trivial``, ``weighted_gap``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = positions.shape[0]
    if total < 2:
        raise ValueError("a descent gap needs at least two audited samples")

    path_embedding = mean_embedding(positions, support, weights, bandwidth)
    support_embedding = mean_embedding(support, support, weights, bandwidth)
    inner = float(weights @ support_embedding)

    error = np.empty(total)
    gaps = np.full((2, total), np.nan)
    kernel_sum = 0.0        # sum_{i,j<=n} k(z_i, z_j)
    embedding_sum = 0.0     # sum_{i<=n} m_pi(z_i)
    kernel_support = np.zeros(support.shape[0])

    for index in range(total):
        point = positions[index]
        previous = positions[:index]
        kernel_sum += 2.0 * np.exp(
            -np.sum((previous - point) ** 2, axis=1) / bandwidth
        ).sum() + 1.0
        embedding_sum += path_embedding[index]
        kernel_support += np.exp(-np.sum((support - point) ** 2, axis=1) / bandwidth)
        count = index + 1
        error[index] = kernel_sum / count ** 2 - 2.0 * embedding_sum / count + inner
        if index + 1 == total:
            break
        successor = positions[index + 1]
        witness_next = np.exp(
            -np.sum((positions[: index + 1] - successor) ** 2, axis=1) / bandwidth
        ).sum() / count - path_embedding[index + 1]
        witness_support = kernel_support / count - support_embedding
        reachable = (np.ones(support.shape[0], dtype=bool) if step_radius is None
                     else np.sum((support - point) ** 2, axis=1) <= step_radius ** 2)
        for slot, subset in enumerate((slice(None), reachable)):
            selected = witness_support[subset]
            gaps[slot, index] = witness_next - selected.min() if selected.size else np.nan

    count = np.arange(1, total + 1, dtype=np.float64)
    squared_radius = 1.0 + inner
    # bound[N-1] is the theorem evaluated at N samples: E_1 plus N-1 noise terms plus the
    # weighted gaps of the N-1 completed steps, all over N^2. Slot 0 is E_1 itself.
    accumulated = np.concatenate(([0.0], np.cumsum(count[:-1] * gaps[0, :-1])))
    bound = (error[0] + (count - 1.0) * squared_radius + 2.0 * accumulated) / count ** 2
    return {
        "n": count, "error": error, "gap": gaps[0], "gap_reach": gaps[1], "bound": bound,
        "radius_squared": squared_radius, "trivial": squared_radius,
        "weighted_gap": 2.0 * accumulated[-1] / total ** 2,
    }

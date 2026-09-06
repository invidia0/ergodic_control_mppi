"""Squared-MMD occupation error, its RKHS witness, and the descent gap.

Post-processing only: everything here reads an already-executed trajectory, so it is
NumPy like the rest of :mod:`ergodic_control_mppi.metrics` and never imports JAX.

The kernel is the controller's own, ``k(x, y) = exp(-||x - y||^2 / h)``
(:func:`ergodic_control_mppi.mppi.field.kernel`), so the quantities below are stated in
the same bandwidth the reference field is tuned in.

**The target is discrete, and it is the paper's.** ``pi`` is the target every other metric
in this package is evaluated against: the configured GMM sampled on the metric grid
(:func:`ergodic_control_mppi.experiments.common.build_target_grid`), restricted to the
reachable connected component (``reachable_mask``, the flood fill of
:func:`ergodic_control_mppi.deploy.grid.metric_reachable_mask`) and renormalized -- exactly
what :func:`ergodic_control_mppi.metrics.ergodicity._restrict_to_mask` does to both sides of
every coverage metric. Both grids are sampled on the same ``linspace`` nodes, so the mask
and the density are evaluated at the same points and :func:`grid_target` only has to pair
them. Writing those points ``c_q`` and their normalized masses ``p_q``,

    pi = sum_q p_q delta_{c_q},  m_pi(z) = sum_q p_q k(z, c_q),  ||m_pi||^2 = p^T K p,

both finite sums, exact for that target up to floating-point arithmetic. That is the whole
of the certificate's target claim: it certifies the *declared discrete measure*. It says
nothing about a continuous truncated GMM, which would need a proven discretization-error
bound this module does not attempt.

Restricting ``pi`` is not a refinement, it is the correction that makes the certificate
answer the paper's question. Against the raw whole-plane mixture, the mass sitting inside
obstacles (up to 8% of it on the shipped clutter maps) is unreachable by construction, so
``E_n`` has a nonzero floor, ``delta_n`` cannot reach zero, and the obstacle geometry is
silently charged to the controller as descent gap.

Writing ``rho_n`` for the empirical occupation measure of the first ``n`` executed samples
-- the continuous positions, not binned -- ``g_n = m_rho_n - m_pi`` and
``E_n = ||g_n||^2 = MMD_k^2(rho_n, pi)``, the witness is
``w_n(z) = <g_n, k(z, .)> = (1/n) sum_i k(z, z_i) - m_pi(z)``.

**The recurrence.** ``m_rho_{n+1} = (n m_rho_n + k(z_{n+1}, .)) / (n + 1)`` gives

    (n+1)^2 E_{n+1} = n^2 E_n + 2 n [w_n(z_{n+1}) - c_n] + ||k(z_{n+1}, .) - m_pi||^2,

with ``c_n = <g_n, m_pi>`` free of ``z``. Since ``int w_n dpi = <g_n, m_pi> = c_n``, the
minimum of ``w_n`` over any set carrying all of ``pi`` is at most ``c_n``, so with

    delta_n = w_n(z_{n+1}) - min_z w_n(z),   R^2 = sup_z ||k(z, .) - m_pi||^2,

the bracket is at most ``delta_n`` and telescoping yields

    E_N <= [E_1 + (N-1) R^2 + 2 sum_{n<N} n delta_n] / N^2.

This holds for *any* sequence of points -- it is an identity plus one inequality, not an
assumption about the controller. What it does not do is promise ``delta_n`` is small; that
is the measurement :func:`walk` exists to make.

Two consequences are worth naming separately, because they are what the audit tests:
``delta_n = 0`` gives ``E_N = O(1/N)``, and a merely *bounded* gap gives
``limsup E_N <= 2 limsup N^-2 sum n delta_n`` -- the weighted-average form, which is what
a speed-limited vehicle can hope to satisfy. The uniform form ``limsup E_N <= sup delta_n``
follows from it and is strictly weaker.

**The minimizing set is the support, and that is exact.** ``min_z`` must range over a set
carrying all of ``pi``; ``pi`` is carried by its finitely many support cells, so minimizing
over them is sufficient and is evaluated by enumeration rather than estimated. No Lipschitz
correction and no query-grid convergence study enter the certified number: the grid does not
approximate the minimization, it *defines* the target. ``delta_n`` may come out negative
where the vehicle sits off-support in a hole the witness dips into -- the bound is unharmed,
that is the controller beating the on-support oracle at that step.

``gap_reach`` restricts further, to support cells within one observed step of the vehicle.
It is a **diagnostic and not part of the bound**: a Euclidean disc of the largest observed
displacement is only a proxy for the true double-integrator reachable set, which depends on
velocity, acceleration limits, clamping and collision geometry. Shrinking the choice set
raises the minimum and so lowers the gap, giving the split

    delta_n = [w_n(z_{n+1}) - min_reach w_n] + [min_reach w_n - min_supp w_n],

whose first term is how well the controller chose among the moves it plausibly had and
whose second is a proxy price of being a vehicle rather than an oracle. Read it as
direction, not as a measured decomposition of the physics.
"""

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
    """Return the support points and weights of the paper's discrete target.

    ``density`` and ``mask`` are the ``target_grid`` and ``reachable_mask`` of a run's
    ``arrays.npz``; both are sampled on ``linspace`` nodes spanning the workspace
    (:func:`ergodic_control_mppi.experiments.common.build_target_grid` and
    :func:`ergodic_control_mppi.deploy.grid.metric_reachable_mask`), which is the node
    convention reproduced here. Masking and renormalizing matches
    :func:`ergodic_control_mppi.metrics.ergodicity._restrict_to_mask`, so this and the
    coverage metrics define ``p*`` identically.

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
    """Evaluate ``m_pi(z) = sum_q p_q k(z, c_q)`` at ``(Q, 2)`` queries, as a finite sum.

    Exact for the discrete target, which is what makes the audit a certificate rather than
    a quadrature estimate. Blocked over the query axis so the kernel matrix never
    materializes in full.

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
    """Return the ``R^2 = sup_z ||k(z, .) - m_pi||^2`` the theorem's noise term uses.

    ``||k(z, .) - m_pi||^2 = k(z, z) - 2 m_pi(z) + ||m_pi||^2`` and ``k(z, z) = 1``, so the
    supremum is attained where ``m_pi`` vanishes. Bounding ``m_pi >= 0`` rather than
    minimizing it over a grid keeps this rigorous and grid-free; it is loose by at most
    ``2 min_z m_pi``, which is Gaussian-tail small.
    """
    return 1.0 + self_inner(support, weights, bandwidth)


def trivial_bound(support: ArrayLike, weights: ArrayLike, bandwidth: float) -> float:
    """The bound any pair of probability measures satisfies, as the comparator.

    ``E_n = ||m_rho||^2 - 2<m_rho, m_pi> + ||m_pi||^2`` with ``||m_rho||^2 <= 1`` (an
    average of ``k <= 1``) and ``<m_rho, m_pi> >= 0`` (an average of ``k >= 0``), so
    ``E_n <= 1 + ||m_pi||^2``. That is strictly tighter than the ``(1 + ||m_pi||)^2`` a
    Cauchy--Schwarz reading gives, and it is numerically the same expression as
    :func:`radius_squared`: the point mass attaining ``R^2`` is itself a probability
    measure, and both are bounded by dropping the same cross term. A claimed bound that
    does not beat this says nothing at all, which is why the audit reports the ratio.
    """
    return radius_squared(support, weights, bandwidth)


def walk(
    positions: ArrayLike, support: ArrayLike, weights: ArrayLike, bandwidth: float,
    step_radius: float | None = None,
) -> dict[str, ArrayLike]:
    """Reconstruct ``E_n``, ``delta_n`` and the theorem's bound along one trajectory.

    ``positions`` is the sequence the argument is applied to, so pass the already strided
    subsequence: ``rho_n`` is then the empirical measure of the audited samples and
    ``delta_n`` is the gap at an audited step, which is what the bound is stated over.

    ``gap`` minimizes the witness over the target's support and is the certified quantity
    ``bound`` uses -- exact, by enumeration. ``gap_reach`` additionally restricts to support
    cells within ``step_radius``; it is a diagnostic proxy for the vehicle's choice set and
    enters nothing. See the module docstring for why.

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

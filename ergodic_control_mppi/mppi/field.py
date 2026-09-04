"""The reference field: analytic scores, KDE repulsion, and the service gate.

Every term here is the gradient of an explicit scalar potential in the query ``z``,
and :func:`potential` assembles that potential. The controller tracks
``Gamma_v(grad Phi)`` -- see :func:`ergodic_control_mppi.mppi.core.reference_flow`,
whose pre-gauge output this module supplies.
"""

from dataclasses import replace
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ergodic_control_mppi.parameters import GMMParams, FieldParams

# Design constant: dimensionless floor gating the over-coverage field off as the
# total relative excess vanishes. Small against the O(1) excesses of normal
# operation, so it only acts in the (near-)fully-under-covered regime.
ACTIVITY_FLOOR = 1e-3


def component_logpdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Return component log densities with shape ``(..., M)``."""
    delta = position[..., None, :2] - params.means
    quadratic = jnp.einsum("...mi,mij,...mj->...m", delta, params.covariance_inverse, delta)
    return params.log_normalizers - 0.5 * quadratic


def logpdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the log Gaussian-mixture density at ``(..., 2)`` positions."""
    return logsumexp(params.log_weights + component_logpdf(position, params), axis=-1)


def pdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the Gaussian-mixture density at ``(..., 2)`` positions."""
    return jnp.exp(logpdf(position, params))


def score_pdf(position: jax.Array, params: GMMParams) -> jax.Array:
    """Evaluate the analytic score ``grad(log p)`` with shape ``(..., 2)``."""
    delta = position[..., None, :2] - params.means
    component_scores = -jnp.einsum("mij,...mj->...mi", params.covariance_inverse, delta)
    logits = params.log_weights + component_logpdf(position, params)
    responsibilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("...m,...mi->...i", responsibilities, component_scores)


def responsibility_gaps(params: GMMParams) -> jax.Array:
    """Log-odds margin each component holds at its own centre, shape ``(J,)``.

    ``Delta_i = [log w_i + log N(mu_i; mu_i, S_i)] - max_{j != i} [log w_j + log N(mu_i; mu_j, S_j)]``

    This is the quantity a pointwise score field has to overturn to leave component ``i``:
    below it the responsibility of the component you are standing in stays ~1 and the field
    keeps pointing at its own centre. It depends only on the target, so it is constant for a
    given mission and costs one ``J x J`` evaluation at trace time.

    A single-component target has no rival, so its margin is ``+inf``: there is nowhere the
    field could point instead. That is the honest value, and :func:`per_mode_weighted`
    handles it rather than this masking it into a finite number.
    """
    logits = params.log_weights + component_logpdf(params.means, params)
    own = jnp.diagonal(logits)
    rival = jnp.max(jnp.where(jnp.eye(logits.shape[0], dtype=bool), -jnp.inf, logits), axis=-1)
    return own - rival


def kernel(x: jax.Array, y: jax.Array, bandwidth: jax.Array) -> jax.Array:
    """Evaluate ``exp(-||x-y||^2 / bandwidth)``."""
    delta = x - y
    return jnp.exp(-jnp.sum(delta * delta, axis=-1) / bandwidth)


def kernel_gradient(x: jax.Array, y: jax.Array, bandwidth: jax.Array) -> jax.Array:
    """Return the analytic RBF gradient with respect to ``x``."""
    return (-2.0 / bandwidth) * (x - y) * kernel(x, y, bandwidth)[..., None]


def kde_repulsion(
    positions: jax.Array,
    particles: jax.Array,
    weights: jax.Array,
    bandwidth: jax.Array,
) -> jax.Array:
    """Weighted-mean RBF repulsion pushing each query away from ``particles``.

    Exactly descent on a weighted kernel density estimate. ``kernel_gradient`` is taken
    with respect to the *particle*, and ``grad_m kappa = -grad_z kappa``, so with weights
    that do not depend on ``z``

        kde_repulsion(z) = -grad_z [ sum_i w_i kappa_h(m_i, z) / sum_i w_i ],

    which is the identity :func:`potential` is built on. Used for both the fading memory
    of executed positions and the plan's repulsion of itself.

    Args:
        positions: Query positions with shape ``(Q, 2)``.
        particles: Source positions with shape ``(P, 2)``.
        weights: Non-negative per-particle weights with shape ``(P,)``.
        bandwidth: Positive RBF bandwidth.

    Returns:
        Weighted-mean repulsion with shape ``(Q, 2)``.
    """
    repulsion = kernel_gradient(particles[None, :, :], positions[:, None, :], bandwidth)
    return jnp.sum(repulsion * weights[None, :, None], axis=1) / jnp.maximum(
        jnp.sum(weights), 1e-12
    )


def kde_potential(
    positions: jax.Array,
    particles: jax.Array,
    weights: jax.Array,
    bandwidth: jax.Array,
) -> jax.Array:
    """The scalar whose negative gradient is :func:`kde_repulsion`, shape ``(Q,)``."""
    values = kernel(particles[None, :, :], positions[:, None, :], bandwidth)
    return jnp.sum(values * weights[None, :], axis=1) / jnp.maximum(jnp.sum(weights), 1e-12)


def smoothed(gmm: GMMParams, bandwidth: jax.Array) -> GMMParams:
    """Convolve the target with the normalized kernel ``kappa_h / (pi h)``.

    That kernel is ``N(0, (h/2) I)``, so the convolution is again a Gaussian
    mixture with every covariance inflated by ``(h/2) I`` -- closed form, no
    quadrature. Comparing occupancy at scale ``h`` against ``smoothed(gmm, h)``
    instead of the raw target removes the bandwidth-dependent bias of comparing
    a smoothed density to an unsmoothed one.
    """
    covariance = gmm.covariance + 0.5 * bandwidth * jnp.eye(2)
    return GMMParams(
        means=gmm.means,
        covariance=covariance,
        covariance_inverse=jnp.linalg.inv(covariance),
        log_weights=gmm.log_weights,
        log_normalizers=-0.5
        * (2 * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(covariance)[1]),
    )


def memory_weights(
    memory: jax.Array,
    recency: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    density_floor: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the ``(trail, excess, gate)`` weightings of the memory term.

    All three are functions of ``(memory, recency)`` alone -- nothing here depends on the
    query position, which is what makes the memory term a gradient. Split out of
    :func:`memory_flow` so the potential and the field read the same weights rather than
    two copies that can drift apart.

    The trail weighting is ``omega_i`` and the excess weighting is ``omega_i e_i`` with
    ``e_i = [o_h(m_i) - p_h(m_i)]_+ / (p_h(m_i) + density_floor)``. ``gate`` is returned
    *separately* rather than folded into the excess weights, and must be applied to the
    resulting field: :func:`kde_repulsion` and :func:`kde_potential` both divide by the
    weight sum, so a constant factor inside the weights cancels exactly and the gate would
    silently do nothing.
    """
    bandwidth = field.fine_bandwidth
    occupancy = (kernel(memory[:, None, :], memory[None, :, :], bandwidth) @ recency) / (
        jnp.sum(recency) * jnp.pi * bandwidth
    )
    target = pdf(memory, smoothed(gmm, bandwidth))
    excess = jnp.maximum(occupancy - target, 0.0) / (target + density_floor)
    # Activity gate. The normalized excess field is scale-invariant in the excess
    # (multiplying every e_i by c leaves it unchanged), so it does NOT tend to zero
    # as over-coverage disappears -- it would jump to zero only at the weight-sum
    # floor. Multiplying by S/(S+eps_S), with S the recency-mean excess, makes the
    # excess component genuinely continuous at S=0 while acting as the identity
    # whenever S >> eps_S, i.e. in normal operation.
    activity = jnp.sum(recency * excess) / jnp.sum(recency)
    return recency, recency * excess, activity / (activity + ACTIVITY_FLOOR)


def memory_flow(
    positions: jax.Array,
    memory: jax.Array,
    recency: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    density_floor: jax.Array,
) -> jax.Array:
    """Normalized over-coverage repulsion from the memory buffer.

    Blends two *separately normalized fields* by ``memory_balance`` in ``[0, 1]``: the
    fading trail (0) and the relative over-coverage (1). Because :func:`kde_repulsion`
    divides by the weight sum, blending the two fields makes ``memory_balance`` a genuine
    interpolation coefficient -- unlike blending the weights themselves, whose effective
    mixing coefficient drifts with time, bandwidth and the instantaneous total surplus.

    The trail term is a true probability-weighted average; the excess term is
    deliberately a *sub*-probability one, its deficit being the activity gate.

    Divided by ``max_r |grad kappa_h| = sqrt(2/h) e^{-1/2}`` so ``memory_gain`` carries no
    hidden bandwidth dependence and is commensurate with ``plan_gain``.

    Args:
        positions: Query positions with shape ``(Q, 2)``.
        memory: Executed-position buffer with shape ``(P, 2)``.
        recency: Non-negative fading weights with shape ``(P,)``.
        gmm: Target density terms.
        field: Bandwidth and balance.
        density_floor: Positive density scale stabilizing the relative excess.

    Returns:
        Gauge-normalized repulsion with shape ``(Q, 2)``.
    """
    bandwidth = field.fine_bandwidth
    trail, excess, gate = memory_weights(memory, recency, gmm, field, density_floor)
    blended = (1.0 - field.memory_balance) * kde_repulsion(
        positions, memory, trail, bandwidth
    ) + field.memory_balance * gate * kde_repulsion(positions, memory, excess, bandwidth)
    return jnp.sqrt(0.5 * jnp.e * bandwidth) * blended


def responsibilities(position: jax.Array, gmm: GMMParams) -> jax.Array:
    """GMM responsibilities at one position, shape ``(J,)``."""
    return jax.nn.softmax(component_logpdf(position, gmm))


def service_ratio(memory: jax.Array, recency: jax.Array, gmm: GMMParams) -> jax.Array:
    """Return how well-served the mode the vehicle currently sits in has been.

    ``sigma = sum_j r_j(m_P) * share_j / w_j`` where ``r_j`` is the GMM responsibility,
    ``share_j`` is the recency-weighted share of buffer mass assigned to component ``j``,
    and ``w_j`` is that component's target weight. ``sigma = 1`` means the current mode has
    received exactly the share of recent path the target asked for; below 1 it is
    under-served, above 1 over-served.

    Unlike the relative excess it does **not** carry ``p*`` in a denominator, so it is not
    suppressed inside a mode -- measured offline it rises monotonically through 4/4 visits,
    while the excess falls through 0/4. It is a deterministic readout of ``(memory, recency)``
    and adds no state, so the Markov triple ``(x, mu, M)`` is unchanged.

    Args:
        memory: Executed-position buffer with shape ``(P, 2)``; oldest first.
        recency: Non-negative fading weights with shape ``(P,)``.
        gmm: Target density terms.

    Returns:
        Scalar service ratio at the newest buffer entry.
    """
    weights = jax.nn.softmax(
        jax.vmap(component_logpdf, in_axes=(0, None))(memory, gmm), axis=-1
    )                                                    # (P, J)
    mass = recency @ weights                             # (J,)
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    target = jnp.exp(gmm.log_weights)
    return jnp.sum(weights[-1] * share / jnp.maximum(target, 1e-12))


def service_ratio_from_mass(mass: jax.Array, position: jax.Array,
                            gmm: GMMParams) -> jax.Array:
    """Service ratio from an exponentially-weighted mass accumulator.

    ``mass`` is the same recency-weighted per-component buffer mass that
    :func:`service_ratio` sums out of the trail, carried instead as a running
    ``m <- gamma m + r(x)``. The two agree in the limit, but the accumulator's
    timescale is set by ``gamma`` alone, so the window over which "has this mode had
    its share?" is asked is free of the repulsion's ``memory_time`` -- which matters
    because the repulsion wants a trail of metres and this wants a history of visits.
    It also costs ``O(J)`` a step rather than the trail's ``O(P^2)``.
    """
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    weights = jnp.exp(gmm.log_weights)
    return jnp.sum(responsibilities(position, gmm) * share / jnp.maximum(weights, 1e-12))


def deficit_weighted(mass: jax.Array, gmm: GMMParams, floor: jax.Array) -> GMMParams:
    """Re-weight the mixture toward components the path has under-served.

    The release gate (:func:`service_ratio_from_mass`) answers "should I leave?" but is
    evaluated at the current position, so it carries no preference between destinations.
    Measured, that is exactly where the remaining gap sits: with the gate on, the vehicle
    dwells deeply and reaches every mode, but settles into a two-mode shuttle and completes
    no full cycle, because the attraction always prefers the nearer partner.

    Per component, ``sigma_j = (mass_j / sum mass) / w_j`` is the same service ratio the gate
    uses, read per-mode rather than at a point; ``[1 - sigma_j]_+`` is 0 for an over-served
    component and approaches 1 for an untouched one. Weights become
    ``w_j (floor + [1 - sigma_j]_+)``, renormalized, so ``floor`` bounds how far the target
    may be bent and ``floor -> inf`` recovers the true mixture.

    Only the *attraction* is re-weighted. The memory term keeps the true ``p*``: the excess
    field is what defines coverage, and bending it would change the objective rather than
    the route to it.
    """
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    weights = jnp.exp(gmm.log_weights)
    deficit = jnp.maximum(1.0 - share / jnp.maximum(weights, 1e-12), 0.0)
    bent = weights * (floor + deficit)
    return replace(gmm, log_weights=jnp.log(bent / jnp.maximum(jnp.sum(bent), 1e-12)))


def per_mode_weighted(mass: jax.Array, gmm: GMMParams, floor: jax.Array,
                      release_ratio: float) -> GMMParams:
    """`deficit_weighted` plus a demotion that releases every mode at the same over-service.

    The multiplicative bend above shifts a log-weight by at most ``log((c+1)/c)`` -- 3.04
    nats at the deployed ceiling of 0.05 -- which cannot overturn the ``Delta_j`` margins of
    :func:`responsibility_gaps` (18.7 and 31.1 nats on the deployed target). Promotion alone
    therefore never empties a basin; the demotion term does, by penalising the *over*-served
    component directly in log space, where the margin lives.

    A single scalar penalty releases component ``j`` at ``sigma*_j = 1 + Delta_j / kappa``,
    so the component with the smallest log-odds gap leaves earliest and is under-served by
    construction. Setting ``kappa_j = Delta_j / (sigma* - 1)`` equalizes the release point
    across modes. ``Delta_j`` comes from the target itself, so this *removes* a tuned
    nats-scale constant rather than adding one: the knob becomes "leave at ``sigma*`` times
    fair share". Smooth in ``mass``, so the continuity the closed-loop argument needs
    survives; a hard mask would not.
    """
    share = mass / jnp.maximum(jnp.sum(mass), 1e-12)
    weights = jnp.exp(gmm.log_weights)
    deficit = jnp.maximum(1.0 - share / jnp.maximum(weights, 1e-12), 0.0)
    bent = weights * (floor + deficit)
    ratio = share / jnp.maximum(weights, 1e-12)
    # release_ratio -> 1 demands an unbounded penalty (release exactly at fair share); the
    # config floor keeps the divisor away from zero.
    penalty = responsibility_gaps(gmm) / max(release_ratio - 1.0, 1e-6)
    # A component with no rival has an infinite margin, and no penalty can release it --
    # there is nowhere to go. Demoting it is meaningless, and `inf * 0` at exactly fair
    # share is NaN, which would take the whole field down on a unimodal target.
    penalty = jnp.where(jnp.isfinite(penalty), penalty, 0.0)
    log_bent = jnp.log(jnp.maximum(bent, 1e-30)) - penalty * jnp.maximum(ratio - 1.0, 0.0)
    return replace(gmm, log_weights=log_bent - logsumexp(log_bent))


def attraction_target(gmm: GMMParams, field: FieldParams,
                      service_mass: jax.Array | None) -> GMMParams:
    """The mixture the score attraction is taken of.

    No service mass, or ``deficit_ceiling <= 0``, leaves the true target untouched. The
    ceiling is a *traced* leaf -- so the ``c`` arms can share one batched call, which is
    what makes them comparable -- and therefore selects with ``jnp.where`` rather than a
    Python branch. ``release_ratio`` is static, so it selects which bend is traced at all.
    """
    if service_mass is None:
        return gmm
    ceiling = jnp.maximum(field.deficit_ceiling, 1e-12)
    bent = (per_mode_weighted(service_mass, gmm, ceiling, field.release_ratio)
            if field.release_ratio > 0
            else deficit_weighted(service_mass, gmm, ceiling))
    return jax.tree.map(
        lambda b, true: jnp.where(field.deficit_ceiling > 0, b, true), bent, gmm
    )


def potential(
    positions: jax.Array,
    memory: jax.Array,
    recency: jax.Array,
    plan: jax.Array,
    gmm: GMMParams,
    field: FieldParams,
    density_floor: jax.Array,
    service_mass: jax.Array | None = None,
) -> jax.Array:
    """The scalar ``Phi`` whose gradient is the pre-gauge reference field, shape ``(Q,)``.

    ``Phi(z; M, s, Z) = log p_hat*(z; s)
                        - lambda sqrt(he/2) [ (1-a) rho_omega(z) + a A_hat rho_excess(z) ]
                        - g sqrt(he/2) rho_plan(z)``

    with each ``rho`` the weighted KDE of :func:`kde_potential`. Every weight -- recency,
    excess, the activity gate, ``memory_balance``, the bent mixture -- is constant in ``z``,
    which is exactly why the field is a gradient rather than merely a direction field. It is
    only a gradient because there is no rotation: ``R(theta) grad Phi`` is not the gradient
    of anything unless ``R = I``.

    Not called on the control path -- :func:`ergodic_control_mppi.mppi.core.reference_flow`
    evaluates the three gradients in closed form. This exists so the claim is falsifiable
    (``tests/test_field.py`` finite-differences it against the field) and so Sec. III can
    draw the controller as a landscape.
    """
    bandwidth = field.fine_bandwidth
    gauge = jnp.sqrt(0.5 * jnp.e * bandwidth)
    trail, excess, gate = memory_weights(memory, recency, gmm, field, density_floor)
    phi = logpdf(positions, attraction_target(gmm, field, service_mass))
    phi -= field.memory_gain * gauge * (
        (1.0 - field.memory_balance) * kde_potential(positions, memory, trail, bandwidth)
        + field.memory_balance * gate * kde_potential(positions, memory, excess, bandwidth)
    )
    phi -= field.plan_gain * gauge * kde_potential(
        positions, plan, jnp.ones((plan.shape[0],), dtype=positions.dtype), bandwidth
    )
    return phi

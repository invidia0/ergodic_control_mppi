def _legacy_memory_flow(
    params: ControllerParams,
    source_particles: jax.Array,
    memory: jax.Array,
    recency: jax.Array,
) -> jax.Array:
    """Two-scale fading-memory feedback (``memory_mode: legacy``).

    Kept verbatim as the A/B baseline for the multiscale reformulation; slated
    for removal once the comparison settles.
    """
    # Fading-memory coverage feedback: repel the plan from the executed-position
    # buffer. Two per-particle weightings, blended by deficit_gate in [0, 1]:
    #  - fading (gate=0): recency decay gamma**age; the recent orbit drives
    #    spiral-fill while old positions fade so modes can be revisited.
    #  - coverage-deficit (gate=1): weight w_i = relu(o_i - p_i) comparing the
    #    recency-weighted occupancy *density* o (a proper KDE, integrates to 1) to
    #    the target density p at memory point i. Zero pressure while a region is
    #    under-filled (dwell), positive once occupancy exceeds its target share
    #    (eject) -> distributes dwell-time proportionally and frees under-covered
    #    modes (o~0 there -> no repulsion, so attraction pulls the robot in).
    #    Both are true densities (no visited-only calibration), so a mode filled
    #    past its share ejects globally. Bounded, deterministic -> Markov.
    target_density = pdf(memory, params.gmm)

    def occupancy_density(bandwidth):
        """Recency-weighted KDE of visitation at the memory points (integrates to ~1)."""
        kde = kernel(memory[:, None, :], memory[None, :, :], bandwidth) @ recency
        return kde / (jnp.sum(recency) * jnp.pi * bandwidth)

    fine_bw = params.stein.spiral_bandwidth
    o_fine = occupancy_density(fine_bw)

    # Coarse coverage-deficit: leave a filled mode and reach a far one (distribution).
    coarse_bw = params.stein.repulsion_bandwidth
    deficit = jnp.maximum(occupancy_density(coarse_bw) - target_density, 0.0)
    # Fill-gated eject: with eject_fill_gated on, the eject force acts only from cells that
    # are already fine-filled (o_fine >= p), so under-filled cells don't push the robot out
    # -> it dwells until the mode is densely covered, THEN leaves (fill-driven, not on a
    # timer). eject_fill_gated=0 leaves the coarse deficit untouched.
    fill_gate = jnp.where(params.stein.eject_fill_gated > 0, (o_fine >= target_density) * 1.0, 1.0)
    deficit = deficit * fill_gate
    gate = params.stein.deficit_gate
    memory_weights = (1.0 - gate) * recency + gate * deficit
    flow = params.stein.repulsion_weight * stein_repulsion(
        source_particles, memory, memory_weights, params.stein, coarse_bw
    )
    # Fine second term: push each orbit onto the next unfilled strip -> dense spiral-fill
    # within a mode. spiral_deficit blends the fine weighting: 1 => fine coverage-deficit
    # relu(o_fine - p) (two-scale deficit), 0 => fading recency trail. spiral_weight=0 off.
    deficit_fine = jnp.maximum(o_fine - target_density, 0.0)
    spiral = params.stein.spiral_deficit
    fine_weights = (1.0 - spiral) * recency + spiral * deficit_fine
    flow += params.stein.spiral_weight * stein_repulsion(
        source_particles, memory, fine_weights, params.stein, fine_bw
    )
    return flow

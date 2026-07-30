# Sec. III-C ↔ code map

**Scope.** The fading-memory coverage feedback is now written into the paper
(`main_updated.tex`, §III-C *Fading-Memory Coverage Feedback*). This note is no longer a parallel
derivation — it is the map from those equations to the lines that implement them, plus the few
implementation facts the paper deliberately leaves out.

Earlier revisions of this file derived a six-parameter two-scale version
$(\lambda_d,\lambda_s,h_c,h_f,r_w,s_w)$ with a binary fill gate. That formulation is retired; see
[`legacy/memory_v1/README.md`](../legacy/memory_v1/README.md) for what it was and why it went, and
[`results/multiscale_memory_horizon/summary.txt`](../results/multiscale_memory_horizon/summary.txt)
for the measurements that replaced it.

---

## Equation → line

| paper | object | code |
|---|---|---|
| (memory_shift) | buffer shift $\mathcal M_{t+1}$ | [`single.py:73`](../ergodic_control_mppi/mppi/single.py#L73) |
| (recency_weights) | $\omega_i$, and $P$ from $\tau_{\mathcal M}$ | [`core.py:184–185`](../ergodic_control_mppi/mppi/core.py#L184-L185), [`config.py:267–269`](../ergodic_control_mppi/config.py#L267-L269) |
| (occupancy_density) | $o^h_t$ at the memory points | [`stein.py:164–166`](../ergodic_control_mppi/mppi/stein.py#L164-L166) |
| (smoothed_target) | $p^\star_h$ | [`stein.py:104`](../ergodic_control_mppi/mppi/stein.py#L104) (`smoothed`), used at [`:167`](../ergodic_control_mppi/mppi/stein.py#L167) |
| (relative_excess) | $e^h_{t,i}$, floor $\varepsilon=1/\lvert\Omega\rvert$ | [`stein.py:168`](../ergodic_control_mppi/mppi/stein.py#L168), floor at [`core.py:187–190`](../ergodic_control_mppi/mppi/core.py#L187-L190) |
| (memory_repulsion) | $\boldsymbol\rho^h_t(\mathbf z;\mathbf w)$ | [`stein.py:74`](../ergodic_control_mppi/mppi/stein.py#L74) (`stein_repulsion`) |
| (memory_distributions), (excess_field), (scale_blend) | $q^{\rm rec}$, activity gate $S/(S+\varepsilon_S)$, balance $a$ | [`stein.py:176–182`](../ergodic_control_mppi/mppi/stein.py#L176-L182) |
| (scale_bank), (scale_gauge), (multiscale_repulsion) | log-spaced bank, $\sqrt{he/2}$ gauge, average | [`stein.py:158–177`](../ergodic_control_mppi/mppi/stein.py#L158-L177) |
| (augmented_reference_flow) | $\widetilde{\mathbf h}^{\mathcal M}_t$ | [`core.py:191–193`](../ergodic_control_mppi/mppi/core.py#L191-L193) |
| (speed_gauge) | $\Pi_v$, $\varepsilon_v=10^{-3}$ | [`core.py:205–209`](../ergodic_control_mppi/mppi/core.py#L205-L209) |
| (coarse_scale) | $h_c$; $h_f=2\delta_{\rm res}^2$ | [`config.py:103`](../ergodic_control_mppi/config.py#L103) (`_mode_scale`), [`:275–283`](../ergodic_control_mppi/config.py#L275-L283) |

Symbol ↔ config key: $\tau_{\mathcal M}$ = `stein.memory_time`, $a$ = `stein.memory_balance`,
$k_{\mathcal M}$ = `stein.memory_gain`, $Q$ = `stein.memory_scales`,
$\delta_{\rm res}$ = `stein.fill_resolution`, $v$ = `stein.reference_speed`.

Design constants, not config keys: $Q$'s default 3, the $3\tau_{\mathcal M}$ truncation
([`config.py:269`](../ergodic_control_mppi/config.py#L269)), $c_d=1/4$
([`config.py:103`](../ergodic_control_mppi/config.py#L103)), $\varepsilon_p=1/|\Omega|$
([`core.py:187–190`](../ergodic_control_mppi/mppi/core.py#L187-L190)), and
$\varepsilon_S=10^{-3}$ (`ACTIVITY_FLOOR`, [`stein.py`](../ergodic_control_mppi/mppi/stein.py)).

## What the paper leaves out

- **Attraction source.** §III-B writes the attraction as an average over $\hat\pi_t$, the $NT$ rollout
  states. The code uses the **median rollout trajectory**
  ([`core.py:171`](../ergodic_control_mppi/mppi/core.py#L171)), $\approx0.995$ rank-correlated with
  the faithful per-rollout field in the warm-started regime. Only a genuine per-step ensemble split
  would need per-cluster representatives.
- **Self-kernel in $o^h$.** The occupancy KDE is evaluated *at* the memory points, so each point
  contributes its own kernel, $\omega_i/((\sum_j\omega_j)\pi h)$, to its own occupancy. This is a
  uniform positive offset across points at a given scale, absorbed by the relative normalization in
  (relative_excess); it is not corrected for.
- **Boundary margin.** The soft inward margin at
  [`core.py:61–70`](../ergodic_control_mppi/mppi/core.py#L61-L70) is a **task cost** inside $S^{(b)}$,
  not part of the reference field — it stops the curl from parking the robot in a corner. It belongs
  to the "constraints via cost" story in §V, not §III.
- **Cost.** The occupancy KDE is $O(Q P^2)$ per step and dominates the memory term; the rollouts are
  $O(NT)$. Measured: 5.96 ms/step total at $P=3000$, $Q=3$ vs 5.03 ms/step with the retired two-scale
  term at $P=2000$, on an RTX PRO 500.

## Markov structure (load-bearing for §IV)

$\mathcal M_t$ is a deterministic finite shift register of past executed states. Augmenting the state
to $\mathbf Y_t=(\mathbf x_t,\boldsymbol\mu_t,\mathcal M_t)\in\mathcal X\times\mathcal U^T\times\Omega^P$
keeps the closed loop a time-homogeneous Markov chain on a compact space, because $\mathcal M_t$ adds
no independent stochastic dimensions — it is a deterministic readout of the state history, exactly
analogous to the nominal-sequence shift the paper's reduced state already handles. The controller is
therefore **not memoryless**; it carries a bounded, fixed-size fading memory. The manuscript states
this as "constant-memory" throughout (abstract, §I, §III-C), which is the defensible claim.

The reference field is also **continuous** in $\mathbf Y_t$: $[\cdot]_+$ is continuous, the
$\varepsilon_p$- and $\varepsilon_S$-floored denominators are bounded away from zero, and no
indicator appears. This discharges the Feller half of Assumption 6 — the retired fill gate was an
indicator and was in tension with it.

The activity gate is what makes that true, and it is a real fix rather than presentational: without
it the excess field is scale-invariant in the excess ($e\mapsto ce$ leaves
$\sum\omega_ie_i\mathbf g_i/\sum\omega_je_j$ unchanged), so it would *not* fade as over-coverage
disappears and would instead jump to zero when `stein_repulsion`'s `1e-12` weight-sum floor engaged.

Assumption 6's **$C^1$ clause** is only satisfied locally. The positive part is not differentiable on
the coincidence set $\{o^{h_q}_t(\mathbf m_{t,i})=p^\star_{h_q}(\mathbf m_{t,i})\}$; since that set
has empty interior, diagonal witnesses can be chosen off it, which is all Sec. IV needs. A softplus
would buy global smoothness at the cost of one more constant; not currently used.

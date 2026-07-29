# Added theory: from the paper's in-horizon Stein flow to the implemented coverage controller

**Scope.** This note documents the reference-flow machinery that the *implementation* uses to
achieve dwell→fill→eject→revisit coverage (ergodic error ≈ 1.0e-7 @ 20k on the reachable-masked
occupancy MSE), and how it differs from the reference flow written in the paper
(§III-B *Curl-Augmented Stein Reference Flow*). Everything downstream of the reference field —
the flow-matching cost (§III-C, eq. 25–28) and the closed-loop analysis (§IV) — is **structurally
unchanged**: the implementation only changes the definition of the field $\widetilde{\mathbf h}_t(\mathbf z)$
that gets plugged into eq. 25. The paper's §III-B currently defines a field that, on its own, does
**not** produce long-run ergodic coverage; the code replaces it with a richer field. This document
is written to be dropped into the paper: it uses the paper's notation and ends with a concrete
section-refactoring proposal (§8).

The single source of truth for the math below is
[`mppi/core.py:160–242`](../ergodic_control_mppi/mppi/core.py#L160-L242) and
[`mppi/stein.py`](../ergodic_control_mppi/mppi/stein.py). Every equation here is annotated with the
line it implements.

---

## 1. Notation (paper + additions)

Paper symbols kept verbatim: workspace $\Omega\subset\mathbb R^{d_s}$; target density $p^\star$;
score $\nabla\log p^\star$; RBF kernel $\kappa(\mathbf z',\mathbf z)=\exp(-\lVert\mathbf z'-\mathbf z\rVert^2/h)$
(bandwidth $h$ enters as the denominator, matching [`stein.py:36–39`](../ergodic_control_mppi/mppi/stein.py#L36-L39));
Stein operator $\mathcal A_{p^\star}$ (eq. 16); preconditioned operator $\mathcal A^{\mathbf C}_{p^\star}$
(eq. 21) with $\mathbf C=\mathbf D+\beta\mathbf Q$; rollout spatial states $\mathbf z^{(b)}_{t,k}=\Pi(\mathbf x^{(b)}_{t,k})$;
rollout occupancy measure $\hat\pi_t$ (eq. 15); reference field $\widetilde{\mathbf h}_t$ (eq. 22);
replanning time $t$, horizon index $k$.

New symbols introduced by the implementation:

| symbol | meaning | code |
|---|---|---|
| $\mathcal M_t=(\mathbf m_{t,1},\dots,\mathbf m_{t,P})$ | **fading memory buffer**: the last $P$ *executed* spatial positions, oldest-first | [`single.py:73`](../ergodic_control_mppi/mppi/single.py#L73) |
| $a_i=P-i$ | age of buffer entry $i$ ($a=0$ newest) | [`core.py:192`](../ergodic_control_mppi/mppi/core.py#L192) |
| $\omega_i=\gamma^{a_i}$ | recency weight, $\gamma\in(0,1)$ = `memory_decay` | [`core.py:193`](../ergodic_control_mppi/mppi/core.py#L193) |
| $o^{h}_t(\mathbf z)$ | recency-weighted KDE **occupancy density** at bandwidth $h$ | [`core.py:196–199`](../ergodic_control_mppi/mppi/core.py#L196-L199) |
| $h_c,\,h_f$ | coarse / fine repulsion bandwidths (`repulsion_bandwidth`, `spiral_bandwidth`) | [`core.py:201,205`](../ergodic_control_mppi/mppi/core.py#L201-L205) |
| $r_w,\,s_w$ | coarse / fine repulsion gains (`repulsion_weight`, `spiral_weight`) | [`core.py:215,224`](../ergodic_control_mppi/mppi/core.py#L215-L224) |
| $v$ | reference speed (`reference_speed`, $0$ = off) | [`core.py:238`](../ergodic_control_mppi/mppi/core.py#L238) |
| $\lambda_d,\lambda_s\in[0,1]$ | design toggles `deficit_gate`, `spiral_deficit` | [`core.py:213,222`](../ergodic_control_mppi/mppi/core.py#L213-L222) |

**Central discrepancy in one sentence.** The paper builds the *entire* reference field (attraction
**and** repulsion) from the current rollout ensemble $\hat\pi_t$. The implementation keeps the
attraction/curl from (a surrogate of) $\hat\pi_t$ but builds the *coverage-driving repulsion* from
the **persistent fading memory** $\mathcal M_t$ of executed positions, and weights that repulsion by a
**local occupancy-vs-target deficit** at **two spatial scales**. The memory is what turns an
in-horizon particle sampler into a long-run ergodic controller.

---

## 2. Why the paper's §III-B field is insufficient (the motivation)

The paper's $\widetilde{\mathbf h}_t$ (eq. 22–23) averages $\mathcal A^{\mathbf C}_{p^\star}\kappa$ over
$\hat\pi_t$, the $NT$ *current* rollout states. When the field is later evaluated at those same states
(§III-B "source and evaluation sets coincide"), this is a Stein variational particle system: within
one planning horizon the $NT$ predicted states spread out under kernel repulsion and climb the score.

The problem is temporal. $\hat\pi_t$ contains only the **current horizon's** predicted states — a cloud
of spatial extent $\approx v\,T\,\Delta t$ around the robot. It has **no representation of where the
robot has already spent time**. The score term $\nabla\log p^\star$ pulls monotonically toward the
nearest mode; the in-horizon repulsion spreads the *sampling cloud* but cannot stop the *executed
trajectory* from settling into one mode and orbiting it forever. There is no force that says "this mode
already has its share of visitation time — leave." Consequently the §III-B field yields, in closed
loop, a stationary law $\rho^\star$ concentrated on the nearest reachable mode, **not** $p^\star$
(large $\mathrm{TV}(\rho^\star,p^\star)$, so the eq. 64 bound is loose and coverage fails). This is
exactly the mode-collapse the implementation had to fix.

The fix is to make the repulsion source a **fading memory of executed visitation** and weight it by
the **occupancy deficit**. Sections 3–6 build this field; §7 assembles it.

---

## 3. Added mechanism I — fading memory occupancy measure

Replace the transient $\hat\pi_t$ (as the *repulsion source*) with the bounded, recency-weighted
empirical measure of executed positions
$$
\hat\pi^{\mathcal M}_t(\mathbf z)\;=\;\frac{1}{\sum_{i=1}^{P}\omega_i}\sum_{i=1}^{P}\omega_i\,
\delta\!\big(\mathbf z-\mathbf m_{t,i}\big),
\qquad \omega_i=\gamma^{a_i},\;\; a_i=P-i .
\tag{3.1}
$$
$\mathcal M_t$ is a length-$P$ ring buffer updated by the shift
$\mathcal M_{t+1}=(\mathbf m_{t,2},\dots,\mathbf m_{t,P},\Pi(\mathbf x_{t+1}))$
([`single.py:73`](../ergodic_control_mppi/mppi/single.py#L73)). The geometric weights $\gamma^{a_i}$
give an effective memory horizon $\tau\approx 1/(1-\gamma)$ steps: the recent orbit dominates (drives
within-mode fill), old positions fade (modes become revisitable). This is a genuine **fading memory**,
strictly cleaner than an oldest-window split.

From (3.1) define the **fading occupancy density** — a proper KDE that integrates to $\approx 1$:
$$
o^{h}_t(\mathbf z)\;=\;\frac{\displaystyle\sum_{i=1}^{P}\omega_i\,\kappa_h(\mathbf m_{t,i},\mathbf z)}
{\big(\textstyle\sum_i\omega_i\big)\,\pi h},
\qquad \kappa_h(\mathbf x,\mathbf y)=e^{-\lVert\mathbf x-\mathbf y\rVert^2/h},
\tag{3.2}
$$
(the $\pi h$ normalizer is $\int_{\mathbb R^2}e^{-\lVert\mathbf u\rVert^2/h}\,d\mathbf u$;
[`core.py:196–199`](../ergodic_control_mppi/mppi/core.py#L196-L199)). $o^h_t$ is the online estimate
of the closed-loop visitation law $\rho^\star$ that the paper's §IV reasons about **asymptotically** —
here it is available at runtime and used *inside* the controller.

**Markov / "memoryless" note (load-bearing for §IV).** $\mathcal M_t$ is a deterministic finite shift
register of past executed states. Adding it to the augmented state,
$\mathbf Y_t=(\mathbf x_t,\boldsymbol\mu_t,\mathcal M_t)\in\mathcal X\times\mathcal U^T\times\Omega^P$,
keeps the closed loop a **time-homogeneous Markov chain on a compact space**. The controller is
therefore *not* memoryless in the paper's current strict sense — it carries a **bounded, fixed-size
fading memory**. This is the one claim in the paper (title/abstract/§I "memoryless") that must be
reconciled; see §8. The Markov structure — hence the entire §IV apparatus — survives the augmentation,
because $\mathcal M_t$ adds no independent stochastic dimensions (it is a deterministic readout of the
state history, exactly analogous to the deterministic nominal-sequence shift already handled by the
paper's reduced state $\widetilde{\mathbf Y}_t$, eq. 33).

---

## 4. Added mechanism II — coverage-deficit weighting

Uniform repulsion from $\mathcal M_t$ would push the robot away from *all* visited regions equally,
including under-covered ones. Instead, weight each memory point by its **local occupancy deficit**
relative to the target:
$$
d^{c}_{t,i}\;=\;\big[\,o^{h_c}_t(\mathbf m_{t,i})-p^\star(\mathbf m_{t,i})\,\big]_{+},
\tag{4.1}
$$
([`core.py:206`](../ergodic_control_mppi/mppi/core.py#L206)). The interpretation is the crux of the
added theory:

- Where a region is **under-covered** ($o<p^\star$): $d^{c}=0$, **no repulsion** — the score
  attraction is free to pull the robot in and let it dwell.
- Where a region is **over-covered** ($o>p^\star$): $d^{c}>0$, repulsion **proportional to the excess
  visitation** — the robot is pushed out.

Both $o$ and $p^\star$ are *true densities* (no visited-only calibration), so a mode filled past its
$p^\star$-share ejects *globally*, freeing time for the other modes. This is precisely a **local,
online surrogate for the gradient of the ergodic metric** (eq. 2): eq. 2 penalizes
$\big[o(\cdot)-p^\star(\cdot)\big]^2$ at every scale, and $-\nabla$ of that pushes down the
positive-deficit regions. In spirit this is the same object HEDAC realizes with a PDE-solved
potential field, but here it is obtained in closed form from the fading KDE and injected as a
**per-particle repulsion weight** — no PDE solve, and it rides inside the existing MPPI cost.

The deficit weighting is blended with the plain recency weighting by the toggle $\lambda_d$
(`deficit_gate`, hardcoded to $1$ in the shipped config):
$$
w^{c}_{t,i}=(1-\lambda_d)\,\omega_i+\lambda_d\,d^{c}_{t,i}\,g_{t,i},
\tag{4.2}
$$
with the fill gate $g_{t,i}$ defined next.

---

## 5. Added mechanism III — multiscale repulsion and fill-gated ejection

The ergodic metric (eq. 2) is intrinsically **multiscale** (the integral over ball radius $r\in[0,R]$).
The implementation approximates this with **two** repulsion bandwidths that do opposite jobs:

- **Coarse** $h_c$ (`repulsion_bandwidth` ≈ 3 m): "is this *mode* over its share?" → drives
  **distribution across modes** (leave a filled mode, reach a far one).
- **Fine** $h_f$ (`spiral_bandwidth` ≈ 0.3 m): "is this *strip* already covered?" → drives
  **within-mode dense spiral-fill** (step each orbit onto the next unfilled strip).

A single bandwidth cannot do both: a coarse scale declares a mode "filled" after ≈2 orbits and ejects
before the mode is densely covered. The two terms use the same occupancy density (3.2) evaluated at the
two scales, $o^{h_c}_t$ and $o^{h_f}_t$.

**Fill-gated ejection** decouples eject *strength* from eject *timing*. Gate the coarse deficit so it
fires only from cells that are **already fine-filled**:
$$
g_{t,i}\;=\;\mathbb 1\!\big[\,o^{h_f}_t(\mathbf m_{t,i})\;\ge\;p^\star(\mathbf m_{t,i})\,\big]
\quad(\text{when }\texttt{eject\_fill\_gated}=1),\qquad g_{t,i}\equiv1\ \text{otherwise},
\tag{5.1}
$$
([`core.py:211`](../ergodic_control_mppi/mppi/core.py#L211)). Under-filled cells contribute no eject
force, so the robot **dwells until the mode is densely covered at the fine scale, then leaves** — a
fill-driven departure rather than a timer-driven one. This is what let dense fill and mode-distribution
coexist (Phase 4: a *strong* eject, gated off until fill completes).

The fine term carries its own weighting, blended by $\lambda_s$ (`spiral_deficit`, hardcoded $0$):
$$
w^{f}_{t,i}=(1-\lambda_s)\,\omega_i+\lambda_s\,\big[o^{h_f}_t(\mathbf m_{t,i})-p^\star(\mathbf m_{t,i})\big]_{+}.
\tag{5.2}
$$
With $\lambda_s=0$ (shipped) the fine weight is the pure fading trail $\omega_i$: the recent orbit
repels the next at the fine scale, producing concentric spirals.

---

## 6. Added mechanism IV — the speed-normalization gauge

The paper's eq. 27 surrogate is minimized at $\Delta\mathbf z^{(b)}_{t,k}\approx\Delta t\,
\widetilde{\mathbf h}_t(\mathbf z^{(b)}_{t,k})$, i.e. the *commanded speed equals the field magnitude*
$\lVert\widetilde{\mathbf h}_t\rVert$. The implementation optionally **normalizes the field to a fixed
reference speed** before it enters the cost:
$$
\Pi_v[\mathbf u]\;=\;
\begin{cases}
\dfrac{v\,\mathbf u}{\max(\lVert\mathbf u\rVert,\varepsilon)}, & v>0,\\[1.2ex]
\mathbf u, & v=0,
\end{cases}
\qquad \varepsilon=10^{-3},
\tag{6.1}
$$
([`core.py:238–242`](../ergodic_control_mppi/mppi/core.py#L238-L242)). This tracks the field
**direction** at constant speed $v$ (matching the LQR flow-matching reference's constant-speed dots),
and shifts dwelling from "slow down at a peak" to **path-length-in-region** — the actual fill
mechanism. Two structural consequences that shrink the tuning space (documented in
[`core.py:231–237`](../ergodic_control_mppi/mppi/core.py#L231-L237)):

1. **Magnitude gauge.** After (6.1) only the *direction* of the pre-normalization field survives, so the
   overall scale is a free gauge. The attraction has implicit weight 1; therefore $r_w$ and $s_w$ act as
   **ratios to the attraction**, not independent magnitudes. The search space is **two relative weights,
   not three magnitudes**.
2. **Horizon reach.** The geometric span of one plan is $L=v\,T\,\Delta t$. Hence $v$ and the MPPI
   horizon $T$ are **partially redundant** through $L$; tune one against the reported horizon.

**Analysis impact:** none structural. $\Pi_v[\cdot]$ is still a fixed field, evaluated once from the
current ensemble/memory and held constant during the MPPI update, so it enters eq. 25 exactly as
$\widetilde{\mathbf h}_t$ did. Proposition 4 and Assumption 6 are stated for a generic held-fixed
reference field and are unaffected — only the *definition* of that field changes.

---

## 7. The assembled reference field

Two more implementation facts before assembling:

- **Curl $\mathbf C$ is a single-angle rotation.** The code uses $\mathbf C=\mathbf R(\theta)=
  \cos\theta\,\mathbf I+\sin\theta\,\mathbf J$ (with $\mathbf J=\big[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\big]$;
  [`config.py:225–227`](../ergodic_control_mppi/config.py#L225-L227)). This is exactly the paper's
  $\mathbf C=\mathbf D+\beta\mathbf Q$ specialized to the isotropic, one-parameter family
  $\mathbf D=\cos\theta\,\mathbf I$, $\beta\mathbf Q=\sin\theta\,\mathbf J$. So $\theta$ is a single knob
  spanning pure attraction ($\theta=0$) to strong circulation, and it preserves the zero set of the
  field (Remark 1) since $\mathbf R(\theta)$ is non-singular.
- **Attraction source.** The score/curl term is built from a surrogate of $\hat\pi_t$: the **median
  rollout trajectory** $\bar{\mathbf z}_t=\mathrm{median}_b\,\mathbf z^{(b)}_{t,\cdot}$
  ([`core.py:171`](../ergodic_control_mppi/mppi/core.py#L171)), which is $\approx0.995$ rank-correlated
  with the faithful per-rollout field in the warm-started regime. Write $\bar\pi_t$ for the empirical
  measure on $\bar{\mathbf z}_t$; it plays the role of $\hat\pi_t$ in eq. 22 for the attraction only.

**Full implemented reference field.** For query $\mathbf z$,
$$
\boxed{\;
\widetilde{\mathbf h}_t(\mathbf z)\;=\;\Pi_v\!\Big[\;
\underbrace{\mathbb E_{\mathbf z'\sim\bar\pi_t}\big[\mathcal A^{\mathbf R(\theta)}_{p^\star}\kappa_b(\mathbf z',\mathbf z)\big]}_{\text{attraction + curl (paper eq. 22, source }\bar\pi_t)}
\;+\; r_w\,\boldsymbol\rho^{h_c}_t(\mathbf z;w^c)
\;+\; s_w\,\boldsymbol\rho^{h_f}_t(\mathbf z;w^f)\;\Big]\;}
\tag{7.1}
$$
where the **weighted memory-repulsion operator** at scale $h$ is
$$
\boldsymbol\rho^{h}_t(\mathbf z;w)\;=\;
\frac{\displaystyle\sum_{i=1}^{P} w_{i}\,\mathbf R(\theta)\,\nabla_{\mathbf x}\kappa_h(\mathbf x,\mathbf z)\big|_{\mathbf x=\mathbf m_{t,i}}}
{\displaystyle\sum_{i=1}^{P} w_{i}},
\qquad
\nabla_{\mathbf x}\kappa_h(\mathbf x,\mathbf z)=-\tfrac{2}{h}(\mathbf x-\mathbf z)\,\kappa_h(\mathbf x,\mathbf z),
\tag{7.2}
$$
([`stein.py:99–101`](../ergodic_control_mppi/mppi/stein.py#L99-L101)), the weights $w^c,w^f$ from
(4.2)/(5.2), and the adaptive attraction bandwidth
$b=\max\big(\mathrm{med}_{j,j'}\lVert\bar{\mathbf z}_{t,j}-\bar{\mathbf z}_{t,j'}\rVert^2,\ \ell_{\rm self}^2\big)$
([`core.py:173–176`](../ergodic_control_mppi/mppi/core.py#L173-L176)). Eq. (7.1) is what enters the
eq. 25 flow-residual; everything after eq. 25 is the paper as written.

Note $\boldsymbol\rho^{h}_t$ has the **same functional form** as the kernel-gradient half of
$\mathcal A^{\mathbf C}_{p^\star}$ (eq. 21) — it *is* Stein repulsion, but (i) sourced from $\mathcal M_t$
rather than $\hat\pi_t$, (ii) carrying non-uniform occupancy-deficit weights, and (iii) without the score
term (the score attraction lives only in the first bracket, evaluated once). So (7.1) is not a departure
from the Stein-flow framework — it is the same operator, decomposed across an **attraction source
$\bar\pi_t$** and a **coverage source $\mathcal M_t$**, at multiple scales, with occupancy-aware weights.

*(The soft inward boundary margin at [`core.py:61–70`](../ergodic_control_mppi/mppi/core.py#L61-L70) is a
**task cost** term inside $S^{(b)}$, not part of the flow field — it belongs to the paper's "constraints
via cost" story, penalizing $\big[\text{margin}-\mathrm{dist}_{\partial\Omega}\big]_+^2$ to stop the curl
from parking the robot in a corner. Mention it in §V experiments, not §III.)*

---

## 8. Proposed paper refactoring

Your intuition is correct: §III-B is the subsection that must change, and new material is needed **before**
§III-C. Concretely:

**§III-A (Rollout ensemble).** Keep. Add one sentence distinguishing the two roles of empirical measures
that the current text conflates: $\hat\pi_t$ (in-horizon, for the attraction source) vs. the fading
memory $\hat\pi^{\mathcal M}_t$ (cross-time, for the coverage repulsion) — forward-referencing the new
subsection.

**§III-B (Curl-Augmented Stein Reference Flow).** Trim to what it actually earns: the preconditioned
operator $\mathcal A^{\mathbf C}_{p^\star}$, Remark 1, and Fig. 2. State that $\mathbf C=\mathbf R(\theta)$
is the isotropic one-parameter instance actually used (§7 above). **Remove the implication that this
field alone drives coverage** — add the §2 motivation paragraph: score attraction + in-horizon repulsion
gives within-horizon spread but not long-run ergodicity.

**New §III-C *Fading-Memory Occupancy and Coverage-Deficit Flow*** (insert before the flow-matching cost).
This is the home for the added theory:
1. Fading memory measure $\hat\pi^{\mathcal M}_t$ and occupancy density $o^h_t$ — eqs. (3.1)–(3.2).
2. Coverage-deficit weighting (4.1)–(4.2), with the "online ergodic-metric-gradient / HEDAC-without-a-PDE"
   interpretation (§4) — this is the paper's strongest new conceptual claim.
3. Multiscale (coarse/fine) split and fill-gated ejection (5.1)–(5.2), tied back to the multiscale
   structure of the metric eq. 2.
4. The assembled field (7.1)–(7.2), plus the speed-gauge (6.1) and its two reparameterization remarks (§6).

**§III-D (was §III-C, Flow Matching as an MPPI Cost).** Essentially unchanged — just have eq. 25 ingest the
new $\widetilde{\mathbf h}_t$ of (7.1). One added sentence: with $v>0$ the tracked field is
speed-normalized, so eq. 27 tracks a fixed-speed direction field (a projected flow).

**§IV (Analysis).** Two surgical edits, no restructuring:
- Augment the state to $\mathbf Y_t=(\mathbf x_t,\boldsymbol\mu_t,\mathcal M_t)$ and add one paragraph
  (the §3 Markov note) arguing the finite deterministic buffer preserves time-homogeneity and compactness,
  so Prop. 1–2 / Thm. 1 carry over — the buffer is handled exactly like the deterministic nominal shift
  already is (reduced state, eq. 33).
- Reconcile **"memoryless."** Recommended: replace "memoryless" with **"bounded / constant-size fading
  memory"** throughout (title clause, abstract, §I contributions). The honest and defensible claim is *no
  growing trajectory-history state and $O(P)$ memory independent of run length* — unlike SMC's
  running spectral history — **not** zero memory. This is the one place the current paper overclaims
  relative to the code.

**§V (Experiments).** The ablations already map onto the new levers: curl/attraction (Fig. 5) test
$\theta$ and the score term; add the fading-memory / deficit / two-scale / fill-gate / speed-normalization
ablations here (these are the "most influential" axes — the sensitivity ranking in
[`results/ktsigma_sweep/sensitivity_ranking.txt`](../results/ktsigma_sweep/sensitivity_ranking.txt) puts
`reference_speed`, `ell_self`, `weight_stein`, `memory_decay` on top). Mention the boundary-margin task cost.

**Net effect.** One trimmed subsection (§III-B), one new subsection (§III-C), two paragraph-level edits in
§IV, and the memoryless→fading-memory wording change. The field equation (7.1) is the anchor; everything
else in the pipeline is already written.

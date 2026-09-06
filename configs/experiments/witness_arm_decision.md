# The witness-descent arm: measured, rejected, and why

**Decision: stop. Keep the reference-field controller. Do not build the witness arm.**
Recorded 2026-09-06 against commit `b3df81e`, on the frozen T150 profile.
`results/` is gitignored, so this file is the durable record; the run that produced it is
`uv run --extra cuda13 python scripts/witness_prototype.py`, and its data is
`results/uav/witness_prototype.json`.

## What was being decided

The Phase 0 certificate (`metrics/discrepancy.py`) is *a posteriori* and
controller-agnostic: it reads an executed path and reports `E_n`, the witness gap
`delta_n`, and the bound they satisfy. Nothing in the controller descends any of it. The
open question was whether it should -- whether replacing the field-tracking cost with a
witness cost, which is what kernel herding actually prescribes, buys anything.

That is a large piece of work (a second rollout objective, a config surface, an O(1)-state
running mean-embedding accumulator to reconcile herding's uniform average with the
controller's bounded state). It should not be started on a hunch, so it was gated on a
disposable prototype: `scripts/witness_prototype.py` monkeypatches `mppi_step` for one
process, scores rollout `k` by `sum_t w_n(z_{k,t})` instead of by the field-tracking cost,
and changes nothing else -- same sampling, clamping, control cross term, adaptive
temperature and smoothing.

Thresholds and the witness weight were committed before the comparison ran (`3f72cf9`),
which is the only thing that makes a directional gate evidence rather than a story fitted
to the output. The weight is a calibration, not a tuning: the witness term's spread across
rollouts was matched to the field arm's tracking term at the same state (787.6 against
34.21, so 23), giving both arms' coverage term the same authority against the shared
obstacle, boundary and control costs.

## What was measured

Six paired cells -- maps `(10, 517)`, `(10, 538)`, `(15, 513)` x seeds `0, 1` -- 20,000
steps each, identical starts, identical lane count, one GPU. Medians over cells:

| | field | witness | change | cells better |
|---|---|---|---|---|
| `error_final` (`E_N`) | 0.00338 | 0.00581 | **+71.9%** | 0/6 |
| `weighted_gap` | 0.01950 | 0.01895 | −2.8% | 5/6 |
| `looseness` | 5.76 | 3.44 | −40.2% | 6/6 |
| `occupancy_mse` | 8.05e−08 | 1.03e−07 | +27.7% | 0/6 |
| `fourier_ergodic` | 0.00579 | 0.02119 | +265.9% | 1/6 |
| `ball_ergodic` | 1.533 | 3.688 | +140.6% | 0/6 |
| `tv` | 0.441 | 0.538 | +22.1% | 0/6 |
| `jerk_rms` | 142.9 | 169.6 | +18.7% | 0/6 |
| `max_speed` [m/s] | 6.87 | 9.16 | +33.3% | 0/6 |
| step [ms, 1 lane] | 1.82 | 2.04 | +12.4% | — |

Feasibility held on both arms: no workspace exits, and the witness arm's share of samples
in masked cells was no worse (0.0045 against 0.0050). The Phase 0 prefix inequality held on
every lane of both arms, which is the certificate behaving as advertised -- it is a
property of any path, so it cannot distinguish the arms and was never expected to.

Against the predeclared rule: three checks pass (no new obstacle contact, no new workspace
exit, prefix inequality) and four fail (median `E_N` improvement, sign agreement, coverage
regression, step-time regression). The one-lane step stayed inside the 50 Hz budget.

## Why it lost, as far as this prototype can say

**The witness arm descends its own witness slightly better and covers substantially
worse.** `weighted_gap` fell on 5/6 cells and `looseness` on 6/6 while `E_N` rose on 6/6.
That is the signature of optimizing the wrong measure, not of optimizing badly, and the
prototype's own shortcut is the obvious candidate: its empirical measure is the
controller's *fading, finite* 825-sample ring buffer (16.5 s of trail at 50 Hz), while
`E_N` is scored against the uniform average over the whole 400 s path. A controller that
only remembers the last few seconds will happily re-cover; the term it is descending is
not the one it is graded on.

**The swap also removed more than it replaced.** `_flow_tracking_cost` carries a velocity
gauge (`0.5||displacement||^2`, plus alignment with a field whose magnitude is set by the
speed schedule); the witness cost carries none, which is visible in the 33% higher peak
speed and 19% higher jerk. And the field arm keeps machinery the witness arm does not
have at all: plan self-repulsion, the 45 s service-gate accumulator, and the destination
bias. So this is not a clean test of "witness versus field" -- it is a test of "witness
alone versus the whole shipped architecture", and the shipped architecture wins.

**The step-time regression is intrinsic, not a shortcut.** The field arm compresses its
query set to the median surrogate path and evaluates `T * P` kernels; a witness cost
cannot be compressed that way, because a term identical across rollouts cannot rank them,
so it pays `K * T * P` = 31M kernel evaluations a step. Measured, that is +12.4% at one
lane and +80% batched. A grid accumulator would remove the `P` factor and is the one
change that could fix both this and the fading-buffer problem at once.

## What this does and does not establish

It establishes that **the cheap version of the idea loses, decisively and on every
coverage metric**, so the production arm is not justified by anything measured so far.
Phases 1 and 2 of the plan (build the arm, run the paired head-to-head) are skipped.

It does not establish that witness descent cannot work. The two candidate causes above are
both fixable in principle, and a future attempt that wanted to reopen this should build the
running mean-embedding accumulator *first* -- it is the object the theory actually names,
it makes the descended witness the one the certificate scores, and it removes the `P`
factor from the cost. Reopening without it would repeat this run.

Six cells is a small, directional sample. It is not the reason for the verdict: the
margins are large (+72% `E_N`, +141% `ball_ergodic`) and unanimous (0/6 on five of the
seven quality metrics).

## Consequences for the paper

The proposed method stays the constrained reference-field MPPI controller, and the
terminology migration in the plan's Phase 4 stays unperformed -- renaming a controller
after a mechanism it does not implement would be cosmetic and false. The Phase 0 result
stands on its own and is what the audit section should claim: an a posteriori,
controller-agnostic finite-trajectory certificate on the paper's own target, holding on
36/36 lanes with a median looseness of 5.48. It is not a herding-controller rate, and this
run is the evidence that the distinction is load-bearing rather than pedantic.

# The witness-descent arm: two attempts, and where they left it

**Status: the predeclared gate fails on both attempts, so the reference-field controller
stays the proposed method and the discrepancy machinery stays a certificate.** The second
attempt fails much more narrowly than the first, and on two criteria whose evidence is thin
at six cells -- that is recorded below rather than argued away, because the rule was frozen
before either run and the whole point of freezing it was to not relitigate it afterwards.

Recorded 2026-09-07. `results/` is gitignored, so this file is the durable record; the run
is `uv run --extra cuda13 python scripts/witness_prototype.py` and its data is
`results/uav/witness_prototype.json` (currently holding attempt 2).

## What was being decided

The Phase 0 certificate (`metrics/discrepancy.py`) is *a posteriori* and
controller-agnostic: it reads an executed path and reports `E_n`, the witness gap
`delta_n`, and the bound they satisfy. Nothing in the controller descends any of it. The
open question was whether it should -- whether replacing the field-tracking cost with a
witness cost, which is what kernel herding actually prescribes, buys anything.

That is a large piece of work, so it was gated on a disposable prototype:
`scripts/witness_prototype.py` swaps the closed loop's entry points for one process, scores
rollout `k` by `sum_t w_n(z_{k,t})` instead of by the field-tracking cost, and changes
nothing else -- same sampling, clamping, control cross term, adaptive temperature and
smoothing. Thresholds were committed before each comparison ran and were **identical for
both attempts**, so the two are judged by the same rule and against the same field arm.

Six paired cells both times -- maps `(10, 517)`, `(10, 538)`, `(15, 513)` x seeds `0, 1` --
20,000 steps, identical starts, identical lane count, one GPU.

## Attempt 1: the fading buffer (commit `b3df81e`, rejected)

The empirical measure was the controller's fading 825-sample ring buffer, because that is
the machinery the package already had. Medians over cells:

| | field | witness | change | cells better |
|---|---|---|---|---|
| `error_final` (`E_N`) | 0.00338 | 0.00581 | **+71.9%** | 0/6 |
| `weighted_gap` | 0.01950 | 0.01895 | −2.8% | 5/6 |
| `ball_ergodic` | 1.533 | 3.688 | +140.6% | 0/6 |
| `fourier_ergodic` | 0.00579 | 0.02119 | +265.9% | 1/6 |
| `tv` | 0.441 | 0.538 | +22.1% | 0/6 |
| step [ms, 1 lane] | 1.82 | 2.04 | +12.4% | — |

**It descended its own witness better and covered worse** -- the signature of optimizing
the wrong measure. Its witness was scoped to the last 16.5 s of trail while `E_N` scores
against the whole 400 s path, so it could forget a region and re-cover it at no cost.

## Attempt 2: the running mean-embedding accumulator (commit `d9d371b`)

The one change the first attempt's diagnosis called for. `A_n(c) = (1/n) sum_i k(c, z_i)`
is kept on a fixed 0.2 m grid and updated in place,
`A_{n+1} = (n A_n + k(., z_{n+1})) / (n + 1)` -- the exact uniform average over the whole
executed history in **bounded** state, one grid and a counter, independent of `n`. That is
what reconciles herding with main.tex:103's explicit position against methods needing the
entire history. Querying it is a bilinear lookup, so the witness costs `O(G + K T)` a step
rather than the buffer's `O(K T P)` = 31M kernel evaluations.

| | field | witness | change | cells better |
|---|---|---|---|---|
| `error_final` (`E_N`) | 0.00338 | 0.00174 | **−48.5%** | **6/6** |
| `weighted_gap` | 0.01950 | 0.01410 | **−27.7%** | **6/6** |
| `ball_ergodic` | 1.533 | 1.007 | **−34.3%** | **6/6** |
| `occupancy_mse` | 8.05e−08 | 7.97e−08 | −1.0% | 3/6 |
| `tv` | 0.441 | 0.444 | +0.8% | 2/6 |
| `fourier_ergodic` | 0.00579 | 0.00661 | +14.2% | 3/6 |
| `obstacle_fraction` | 0.0050 | 0.0100 | +100% | 1/6 |
| `jerk_rms` | 142.9 | 159.9 | +11.9% | 0/6 |
| step [ms, 1 lane] | 1.82 | **1.78** | **−2.0%** | — |
| `looseness` | 5.76 | 9.04 | +57.0% | 0/6 |

Against the predeclared rule: **six checks pass** (median `E_N` improvement, sign
agreement, prefix inequality, no workspace exit, step budget, step-time regression) and
**two fail** (no new obstacle contact, no coverage metric worse by more than 5%).

The accumulator did what it was supposed to do on every axis it was built for. It also made
the arm *faster* than the field arm, which the first attempt could not be: a witness cost
cannot use the surrogate compression the field arm's tracking cost does, because a term
identical across rollouts cannot rank them -- but the accumulator removes the `P` factor
outright, so the comparison never needed that compression.

### Read `looseness` as a property of the bound, not of the controller

It rose 57% while everything else improved. That is arithmetic, not a regression: the
bound carries a controller-independent noise term `(N-1) R^2 / N^2`, so as `E_N` falls
faster than the accumulated gap, the ratio must rise. **A better controller has a looser
certificate.** Any later head-to-head must not report `looseness` as a quality metric.

### The two failures, and how much they weigh

**Obstacle contact doubled (0.0050 -> 0.0100), and this is the hard criterion.** It is
systematic in direction -- 5 of 6 lanes worse -- so it is not a fluke. Two things bound how
much it establishes. The field arm's own seed-to-seed spread on this metric is 0.0015 to
0.0094, a factor of 6.2, so the excess (~0.008) is the same size as the noise the baseline
already carries at this sample size. And the excess tracks clutter (mean +0.0082 on
`10/538`, +0.0078 on `15/513`, −0.0006 on the most open map `10/517`) while the coverage
gain tracks it the same way (`10/538` improves `E_N` 3.1x, `10/517` only 2.2x) -- the arm
spends more time in the gaps between pillars because that is where it had not been. The two
effects are coupled, not independent.

The obvious mechanistic candidate -- that the controller's `m_pi` is the *continuous*
mixture, so unreachable target mass inside pillars pulls the vehicle in -- **is not
supported by the data**: `10/517` carries the most unreachable mass (0.0212, against 0.0152
on `538` and 0.0701 on `513`) and is the one map with no excess. Closing that shortcut is
still the first thing a third attempt should do, but it should not be sold as the known
cause.

**The coverage regression is `fourier_ergodic` alone, and it is not a stable estimate.**
Per cell it runs −80.0%, −18.7%, −12.1%, +44.4%, +56.6%, +162.3%: a factor of twenty,
3 cells better and 3 worse. The +14.2% median is a coin flip at six cells. The other three
coverage metrics are a wash (`occupancy_mse` −1.0%, `tv` +0.8%, both split 3/3) and the
multiscale ball metric -- the one the paper leans on -- improves on 6/6.

## What this establishes

The accumulator was the right diagnosis of attempt 1: the same objective, given the measure
the theory actually names, reverses a 72% loss into a 48% gain on `E_N` and a 34% gain on
the ball metric, unanimously across cells, at no cost in step time. **Witness descent
works.**

What is not established is that it is deployable. It fails a hard feasibility criterion, and
six cells cannot separate that failure from the baseline's own seed noise. Under the frozen
rule that is a stop, and the default holds: the field controller stays the proposed method,
the terminology migration stays unperformed, and the discrepancy machinery stays what Phase
0 made it -- an a posteriori, controller-agnostic finite-trajectory certificate on the
paper's own target, holding on 36/36 lanes with a median looseness of 5.48.

## What a third attempt would need

Not more design -- more evidence, and one shortcut closed:

1. **A larger paired sample.** The gate turns on two quantities that six cells cannot
   resolve: the obstacle excess (within baseline seed spread) and `fourier_ergodic`
   (3/3 split, twenty-fold range). Six maps x six seeds under the campaign's own promotion
   rule would settle both, and that is the Phase 2 harness rather than new code.
2. **Restrict the controller's `m_pi` to the reachable component**, the last shortcut the
   prototype still carries. It is the same defect class Phase 0 fixed in the audit, it is
   cheap now that `grid_target` exists, and it can only reduce attraction toward pillars --
   even though the map-by-map evidence above says it is not the cause of the excess.
3. **A velocity gauge.** The witness cost carries none, which shows in +11.9% jerk and a
   higher peak speed. The field arm's tracking cost is `0.5||d - dt*flow||^2`; the witness
   analogue would keep the quadratic and replace only the direction.

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

## Attempt 2 at 36 cells: both contested criteria settled

Six cells could not say whether the obstacle excess was real or whether the Fourier
regression meant anything, so the same arm was rerun on the full clutter tier -- 6 maps x 6
seeds, 20,000 steps, paired -- with a Wilcoxon signed-rank test per metric and Holm across
the family. Medians and cells-better counts say which way a difference points; only the
test says whether it is there.

| | field | witness | change | cells better | p (Holm) | |
|---|---|---|---|---|---|---|
| `error_final` (`E_N`) | 0.00387 | 0.00194 | **−50.0%** | **35/36** | 2.9e−10 | significant |
| `weighted_gap` | 0.02026 | 0.01384 | −31.7% | 32/36 | 3.2e−09 | significant |
| `ball_ergodic` | 1.702 | 1.217 | **−28.5%** | 27/36 | 0.0115 | significant |
| `occupancy_mse` | 8.42e−08 | 8.53e−08 | +1.2% | 13/36 | 0.451 | not significant |
| `tv` | 0.4449 | 0.4487 | +0.9% | 15/36 | 0.166 | not significant |
| `fourier_ergodic` | 0.00751 | 0.00973 | +29.6% | 17/36 | 0.115 | **not significant** |
| `obstacle_fraction` | 0.0072 | 0.0158 | **+119%** | 5/36 | 9.4e−08 | **significant** |
| `jerk_rms` | 143.5 | 161.6 | +12.7% | **0/36** | 2.9e−11 | significant |
| step [ms, 1 lane] | 1.82 | 1.79 | −1.5% | — | — | — |

**The obstacle regression is real.** The six-cell reading -- that the excess was the size of
the field arm's own seed spread and might be noise -- does not survive: at 36 cells it is
`p = 9.4e-08`, the median more than doubles, and only 5 of 36 cells go the other way. That
was the hard feasibility criterion and it genuinely fails. The jerk regression is real too,
and unanimous: 0 of 36 cells improve.

**The coverage regression is not real.** `fourier_ergodic` fails the frozen 5% threshold on
its median and then fails to reject at `p = 0.115`, with 17 of 36 cells going each way. The
other two shipped coverage metrics are flat and non-significant. So one of the two gate
failures was an artifact of putting a median threshold on a noisy statistic, and the other
was not.

**The coverage win is real and large.** `E_N` halves on 35 of 36 cells at `p = 2.9e-10`, the
accumulated gap falls 32%, and the multiscale ball metric -- the one the paper leans on --
improves on 27 of 36 at `p = 0.0115` after correction. The arm is also *faster* than the
one it replaces (−1.5% at one lane, −11.4% batched).

`results/uav/witness_figures/witness_controller.png` shows why on three maps: the witness
arm's occupancy at scale `h` reproduces the target's three lobes visibly more evenly, while
the field arm's is patchy and over-concentrated, and the witness field it built ends nearly
flat -- the objective driven toward zero. The same figure shows the cost: its path crowds
the pillars where the field arm's sweeps around them.

## What this establishes

Two things, and they point opposite ways.

**Witness descent works as a coverage objective.** Given the measure the theory actually
names -- the uniform running mean, not a fading window -- the same swap that lost 72% of
`E_N` in attempt 1 gains 50% of it in attempt 2, significantly, on the full clutter tier,
at no cost in step time. That is no longer a hint; it is a measured result on 36 paired
cells.

**The arm is not deployable as built.** It doubles the share of samples in cells the
reachable mask excludes and it is jerkier on every single cell. Both are significant, and
obstacle contact is the criterion that was written down as an immediate stop precisely
because a coverage gain bought with clearance is not a gain.

Under the frozen rule this is a stop, and the default holds: the field controller stays the
proposed method, the terminology migration stays unperformed, and the discrepancy machinery
stays what Phase 0 made it -- an a posteriori, controller-agnostic finite-trajectory
certificate on the paper's own target, holding on 36/36 lanes with a median looseness
of 5.48.

## What a deployable version would need

Not more evidence now -- the evidence is in. Two things the current cost simply does not
contain, both of which are redesign rather than tuning:

1. **A velocity gauge.** The witness cost has none. `_flow_tracking_cost` is
   `0.5||d - dt*flow||^2`; dropping it dropped the quadratic along with the direction, which
   is what the +12.7% jerk and the higher peak speed are. The witness analogue keeps the
   quadratic and replaces only the direction with the witness descent direction.
2. **An obstacle-aware target in the controller.** The prototype's `m_pi` is the continuous
   mixture, so mass inside pillars still pulls. Restricting it to the reachable component --
   the same defect class Phase 0 fixed in the audit, and cheap now that `grid_target`
   exists -- is the first thing to try against the clearance regression, though the
   six-cell map ordering gave it no support and it should not be assumed sufficient.

Both change what is being descended, so either would need its own frozen gate and its own
36-cell run. That is a project, not a follow-up, and nothing in the paper depends on it.

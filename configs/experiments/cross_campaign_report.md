# Cross-campaign audit — authoritative interpretation

Written 2026-08-02 from the local per-seed campaign and UAV CSVs. This document supersedes
the interpretations in `campaign_findings.md` and `uav_findings.md`; those remain as the
historical tuning log. Negative and unresolved results are retained here rather than folded
into a preferred narrative.

## 1. Evidence and statistical unit

| source | rows | audit status |
|---|---:|---|
| campaign screening | 225 | re-analyzed from per-seed CSV |
| campaign interactions | 879 + 3 infeasible | re-analyzed from per-seed CSV |
| campaign core | 560 | re-analyzed from per-seed CSV |
| campaign structure | 130 | re-analyzed from per-seed CSV |
| campaign components | 60 | re-analyzed from per-seed CSV |
| campaign generalization | 560 | re-analyzed from per-seed CSV |
| UAV fixed-temperature ablation | 108 | re-analyzed from per-seed CSV |
| UAV low-temperature / temperature arms | 28 / 18 | re-analyzed from per-seed CSV |
| flown UAV summary | 21 | re-analyzed from per-run CSV |

The campaign CSVs are raw scalar rows, but the local `runs/` and `configs/` stage directories
contain no archived files. Occupancy and Fourier metrics therefore cannot be recomputed from
paths. Factorial campaign claims use each density × obstacle-count environment as the unit:
the five controller seeds within an environment are repeated trials, not 80 independent
environments. UAV arms are paired by controller seed.

The campaign ran one Python process per stage without deterministic GPU autotuning disabled.
Dynamic arms sharing an XLA executable are less exposed than comparisons that require new
static compilations (`T`, `K`, memory length, scale count), but every historical GPU result
still carries a process/compile reproducibility caveat. New campaigns set
`--xla_gpu_autotune_level=0` before JAX imports.

## 2. Conclusions that survive the audit

### Memory is load-bearing

- Campaign `memory_off`: median within-environment paired MSE ratio **14.33**, wins **0/16**
  environments. The raw 80-path ratio is 13.58 with 0/80 wins.
- UAV `memory_off`: **12.2×** worse MSE, 0/10 wins (two-sided exact sign p = 0.002), and
  0/10 versus 8/10 paths reaching all modes.

This is the only large effect independently established in both campaigns. Keep
`memory_gain > 0`.

### Weak flow and removing the speed gauge are consistently bad

In the 10-seed campaign components stage, `weak_flow` is 2.37× worse and
`no_speed_gauge` 1.63× worse; every paired seed loses for both (exact sign p = 0.00195).
Keep a dominant flow weight and a nonzero reference speed. These results establish broken
regions, not precise optima at 3000/6000 or 1.8/4.0.

### One well-placed scale is a simplification, not a proven tie

Against Q=3 in the 16-environment core stage:

| arm | median environment MSE ratio | environment wins |
|---|---:|---:|
| Q=1 at fine endpoint | 2.617 | 0/16 |
| Q=1 at campaign midpoint | 1.069 | 5/16 |
| Q=2 | 1.011 | 7/16 |

The midpoint Q=1 arm is close but directionally worse, so “within 7%” is supported and
“equal” is not. The deployment claim that Q=3 dilutes visitation is based on a 1/6 screen
whose rows are not in the archived UAV CSV. Treat it as a mechanism hypothesis. Q=1 remains
a reasonable simplicity choice, conditional on placing its one active bandwidth correctly.

### Memory balance is low leverage, not proven inert

Campaign `excess_only` is 1.024× with 6/16 environment wins; `trail_only` is 1.058× with
4/16 wins. The structure stage likewise cannot rank the nearby balance arms at ten seeds.
Do not spend a tuning campaign on `memory_balance`, but do not claim mathematical or
empirical equivalence.

## 3. Corrections to previous numeric claims

### `K=250` is a real-time trade, not free

Across the 18 `T >= 100` interaction rows per level, paired to K=4000:

| K | median paired MSE ratio | wins | median time ratio |
|---:|---:|---:|---:|
| 125 | 1.148 | 5/18 | 0.764 |
| **250** | **1.088** | **5/18** | **0.783** |
| 500 | 1.056 | 7/18 | 0.822 |
| 1000 | 1.079 | 7/18 | 0.849 |
| 3000 | 1.002 | 9/18 | 0.904 |

K=250 buys about 22% campaign compute and clears the deployed 16 ms deadline. Retain it as
the preregistered engineering choice, but do not state “under 5%” or “no accuracy cost.”

### The horizon campaign is confounded by the temperature cap

`horizon_150` reports a 0.787 median environment MSE ratio and 13/16 wins against T=350.
However, every campaign cell used `lam_max=20`: replay diagnostics show T=350 wanted
lambda ≈158 while T=150 wanted ≈22. The cap therefore penalized the longer horizon by
forcing it into a much sharper argmin regime. The result cannot tune the UAV and T=350
remains preregistered.

### Raising `lam_max` fixes ESS; outcome deltas remain unresolved

The deployed cap of 20 pinned the controller at approximately 0.6% ESS against a 30% target.
With authority up to 1000, lambda self-limits near 151.8 and settled ESS reaches 33.9%.
That mechanism measurement justifies the fix.

Paired across 18 seeds, the reported outcome changes do not separate:

| outcome | `lam_max=1000` / `20` | paired wins | paired test |
|---|---:|---:|---:|
| occupancy MSE | 0.916 | 9/18 | p = 0.702 |
| Fourier | 1.143 | 8/18 | p = 1.000 |
| all modes | 18/18 vs 14/18 | 4 discordant | p = 0.125 |

Do not justify the cap using “8.4% better MSE” or warn that it causes “14% worse Fourier.”

## 4. Bandwidth and curl: directional evidence only

Fixed-temperature UAV ablation, paired on seeds 43–60:

| arm | MSE ratio | MSE wins | Wilcoxon p | all modes |
|---|---:|---:|---:|---:|
| h=0.94 | 1.165 | 5/18 | 0.108 | 17/18 |
| h=6.6 | 1.226 | 4/18 | 0.043 | 18/18 |
| theta=0 | 0.977 | 8/18 | 0.832 | 13/18 |
| theta=15 | 1.108 | 9/18 | 0.671 | 18/18 |
| theta=45 | 1.573 | 0/18 | <0.0001 | 18/18 |

Holm correction over the ten MSE/Fourier arm tests leaves only theta=45 as distinguishably
worse. The theta=0 visitation pattern is five discordant pairs, all favoring theta=30, but
its exact McNemar p is 0.0625: directional and at the experiment's resolution floor.

Keep curl because both campaigns point in the same direction and removing it risks mode
visitation, not because theta=30 is proven optimal. Theta=15 has the best point estimates
for first tour, Fourier, and cycles, but is not separable at 18 seeds. Keep the preregistered
theta=30 configuration for the pillar evaluation.

## 5. Online deployment claims

The five paired Perlin flights are descriptive. UAV/ideal Fourier is worse in 3/5 pairs and
better in 2/5 (paired p = 0.625), contradicting the old statement that every flight moved in
the same direction. Occupancy shows the same variance-dominated pattern. Neither metric
supports equivalence or systematic degradation at n=5.

Offline reached all modes on 18/18 fixed-profile paths; online reached them on 3/5 and more
slowly. This is a real observed gap but its cause remains unresolved. The pillar campaign
measures whether it persists and does not introduce a speculative replay model.

## 6. Reproducibility limits retained in the record

- The 8000-step h sweep cited for h=6.0, 8.5, and 11.0 is not present in the shipped CSVs.
- The low-temperature headline columns for h=0.94/5.0/6.6 are not present in
  `ablation_lowtemp.csv`.
- Six-seed screens can filter gross failures but cannot rank close arms.
- Mode visitation needs at least 20,000 steps because the observed median first tour is
  longer than an 8000-step run.
- Historical campaign scalar metrics are available, but the path/config archives needed
  to rescore new metrics are empty locally.

The preregistered successor is `scripts/run_pillar_campaign.sh`. It selects maps from
geometry before controller execution and writes the resulting audit to
`results/archive/2026-08-05/pillar_45/pillar_report.md`.

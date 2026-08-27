# Ablation campaign — findings for the UAV deployment

> **Historical tuning log.** Its numeric rankings and transfer claims are superseded by
> `cross_campaign_report.md`, which re-analyzes the now-local per-seed CSVs. In particular,
> K=250 is not “free,” T=150 is temperature-confounded, and close surviving arms are not
> statistically ranked. Keep this file for provenance, not as the paper interpretation.

Handoff note for the ROS 2 / UAV work. Written 2026-08-01 against commit `61eee15`.
Source data: `results/campaign/*.csv` on the GPU box (gitignored locally).

**Read the "Do not copy" section before touching `configs/uav_profile.yaml`.** Three of the
headline numbers do not transfer to the UAV, and one of them (`mppi.T`) would put the
vehicle in a regime where 3 of 3 seeds diverged in our sweep.

## Status

| stage | cells | what it establishes | state |
|---|---|---|---|
| screening | 225 | 11 axes, 1-at-a-time, 5 seeds | done |
| interactions | 879/882 | 6 pairwise 7x7 grids, 3 seeds | done (3 infeasible cells) |
| core | 560 | 7 arms x 4 densities x 4 obstacle counts x 5 seeds | done |
| structure | 130 | balance x scale-bank cross, 10 seeds | done |
| components | 60 | auxiliary arms, 10 seeds | done |
| generalization | 560 | tuned package + curl/horizon verdicts, 16 envs | **running, ETA ~23:20 UTC 2026-08-01** |

Everything below stages 1-3 ran on the campaign's own `trimodal` density (means at ±6,
median tr(Sigma) = 5.0) — **not** the density in `mppi_params.yaml` / `uav_profile.yaml`
(means at ±12, median tr(Sigma) = 11.0). That factor-of-2 in spatial scale is the reason
several results need rescaling rather than copying.

## Settled — mechanism-level, should transfer

1. **Memory is the load-bearing component.** `memory_off` loses **80/80 seeds across all
   16 environments**, median 13.6x worse occupancy MSE (11.2x in the 10-seed structure
   stage). Keep `memory_gain > 0`. This is the only claim with no caveats attached.

2. **The scale bank is worth almost nothing against a well-chosen single scale.**
   - `fine_only` (Q=1 pinned at h_f) is 2.64x worse — but that is a *bad scale*, not
     evidence for a span.
   - `one_good_scale` (Q=1 at the geometric midpoint sqrt(h_f·h_c)) is only **1.07x**, and
     the 10-seed structure grid puts Q=1-at-midpoint within **3.3%** of Q=3 at every
     balance setting (a = 0 / 0.85 / 1 → +0.0% / -3.1% / +0.5%).
   - It does *not* help where it should: on the deliberately multi-scale density Q=1 ties
     (1.03) and Q=2 wins (0.98); on `unimodal` Q=1 **beats** Q=3 outright (0.89, 19/20 seeds).

   For the UAV: `memory_scales: 1` at the geometric midpoint is a defensible simplification.
   Be aware the compute saving is small (Q=1→3 measured 4.18→4.28 ms/step); the argument is
   simplicity at no accuracy cost, not speed.

3. **`memory_balance` is nearly inert.** Smallest span of all 11 screened axes (8.9%);
   a = 0 / 0.85 / 1 land within ~14% in the 10-seed grid. Do not spend screening budget here.

4. **The flow weight must dominate.** `weak_flow` (500) is **2.37x worse**. 6000 scored best,
   3000 (current UAV profile) is acceptable, anything near 500 is broken.

5. **Keep the speed gauge.** `no_speed_gauge` (reference_speed = 0, i.e. raw magnitude) is
   **1.63x worse**. The UAV profile's 1.8 m/s is fine as a value; just don't set it to 0.

6. **K is not a lever — cut it.** Over T >= 100, K = 125 and K = 4000 differ by under 5%
   (2.14e-07 vs 2.09e-07), across 21 runs per level. **`mppi.K: 250` is free**, and this is
   the cheapest real-time win available. Well supported.

7. **Buffer truncation at 3·tau is enough.** `long_buffer` (5·tau) gave no benefit (1.03x).
   Keep the derived `memory_length`.

## Rescale, do not copy

These were tuned at the campaign's spatial scale. Transfer the *ratio*, not the number.

| quantity | campaign value | how it is derived | UAV equivalent |
|---|---|---|---|
| `coarse_bandwidth` h_c | derived 5.0; **tuned 3.0 won** (0.93x, 6/10 wins, -41% ergodicity) | median tr(Sigma_j), capped at 0.25·min separation² | derived is **11.0**; the same 0.6 ratio gives **≈ 6.6**. Copying 3.0 lands at 0.27x derived — far off. |
| `fine_bandwidth` h_f | 0.08 | 2·`fill_resolution`², fill_resolution = sensor footprint / track spacing | re-derive from the camera ground footprint at flight altitude. Do not copy 0.2. |
| Q=1 midpoint | sqrt(0.08·5.0) = 0.632 | sqrt(h_f·h_c) | recompute with the UAV's h_f and h_c (≈ 0.94 if h_f stays 0.08) |
| `memory_time` tau | **2.5 s won** (-18%) at v = 4 | tau·v is the *trail length* it forgets over: 2.5·4 = 10 m | for the same 10 m at v = 1.8, **tau ≈ 5.5 s** — scale up, don't copy 2.5 |

Note tau is not a cost driver in our sim despite the comment in `mppi_params.yaml`: tau = 2.5
(P = 375) and tau = 20 (P = 3000) both ran ~4.3 ms/step. Rollout length dominates, not the
O(Q·P²) occupancy KDE.

## Do not copy: `mppi.T: 150`

The "T=150 beats 350 at 2.2x less compute" result **is a high-speed phenomenon** and the UAV
is speed-capped at 2.0 m/s by the safety shield (profile runs 1.8).

Per-seed occupancy MSE from the speed × horizon grid (3 seeds, reach L = v·T·dt):

```
v = 4.0 (campaign default)            v = 2.0 (nearest measured to the UAV's 1.8)
  T=50   L= 4m  1.20e-6 3.47e-6 4.08e-6   T=50   L= 2m  9.00e-8 1.54e-7 4.39e-6
  T=100  L= 8m  1.26e-7 2.00e-7 4.59e-7   T=100  L= 4m  1.46e-7 1.52e-7 2.56e-7
  T=150  L=12m  9.47e-8 1.24e-7 3.86e-7   T=150  L= 6m  1.32e-7 1.89e-7 4.99e-7
  T=350  L=28m  1.61e-7 2.06e-7 2.22e-7   T=350  L=14m  2.07e-7 2.08e-7 2.36e-7
  T=700  L=56m  2.27e-7 2.64e-7 3.01e-7   T=700  L=28m  2.80e-7 2.88e-7 3.06e-7
```

Three things to take from this:

- **T = 50 is unstable, and the instability grows with speed.** At v = 2 it caught 1 seed in
  3 (4.39e-6); at v = 4 it caught **3 of 3** (1.2–4.1e-6). Never screen below T = 100.
- **At the UAV's speed the curve is much flatter.** The v = 4 row shows T=150 beating T=350
  by 40%; the v = 2 row shows T=100 beating T=350 by only ~27%, and T=150 carries a fatter
  tail (one seed at 4.99e-7). Treat a short horizon on the UAV as a *compute* win, not an
  accuracy win.
- **Two dimensional arguments disagree, so measure it.** Reach-matching says the UAV needs
  T ≈ 350 to reproduce our best L = 12 m at v = 1.8; the directly-measured v = 2 row says
  T = 100–150. The v = 2 row is the more direct evidence but rests on 3 seeds and a narrower
  density than the UAV flies. **Screen T ∈ {100, 150, 250, 350, 500}** with
  `scripts/run_uav_screen.sh` rather than adopting either.

The UAV target is also ~2x wider than the campaign's (means ±12 vs ±6, tr(Sigma) 11 vs 5) and
the vehicle is slower with `max_accel_lin_abs: 8.0` and an SO3 attitude loop underneath.
Wider target plus slower vehicle means longer transits — the regime where long horizons pay.
That pushes the answer toward the high end of the screen range.

## RESOLVED — `generalization` reported 2026-08-01T23:46Z (560/560, 0 errors)

- **The curl stays. Your warning was right and my hypothesis was wrong.** theta = 0 loses at
  *both* horizons over 16 environments: 1.062x at T = 350 (wins 32/80 seeds, 6/16 envs) and
  1.108x at T = 150 (30/80, 5/16). The kill rule was "delete only if theta = 0 wins at both";
  it won at neither. Independently confirmed by your 18-seed mode-visitation table.
- **`best` failed** — 1.321x vs `full`, 4/16 envs. Do not adopt any part of it. It is 1.581x
  worse than the horizon change alone, so the five non-T ingredients cost 58% together. Your
  finding that k_M = 60 is actively harmful at a correct lengthscale (0/6 seeds) is the most
  likely explanation, and it means **k_M = 60 should not be copied anywhere.**
- **`horizon_150` (T = 150) measured 0.784x, 13/16 envs, 59/80 seeds at 2.2x less compute —
  and it is CONFOUNDED. See below. Do not adopt it.**

## Your `lam_max` warning applies to the whole campaign — confirmed

Replaying `run_single`'s scan at the campaign commit on `configs/mppi_params.yaml`
(`lam_max: 20.0`, `ess_target: 0.3`), 600 steps on the GPU box:

| config | temperature (last 200) | pinned at cap | achieved ESS |
|---|---|---|---|
| T = 350, lam_max = 20 (**the campaign**) | 20.00 | **100%** | **0.0014** |
| T = 350, lam_max = 1000 | 158.3 | 0% | 0.426 |
| T = 150, lam_max = 20 | 19.9 | 81.5% | 0.114 |
| T = 150, lam_max = 1000 | 21.9 | 0% | 0.305 |

**Every cell of all 2414 runs was ranked in the argmin regime**, at 0.14% ESS against a 30%
target — worse than the 0.6% you measured on the deployment.

This confounds the horizon result specifically, and it is not a small correction. The cost
scale grows with T, so the temperature the loop *wants* grows with it: ~158 at T = 350 but
only ~22 at T = 150. The fixed cap of 20 is therefore 8x too low at T = 350 and very nearly
correct at T = 150. **"T = 150 beats T = 350" is substantially "T = 150 happens to sit where
the cap is adequate"**, not evidence that a shorter horizon plans better.

The same objection applies in proportion to every axis that moves the cost scale — T, K,
`weight_stein`, `memory_gain`, obstacle count and density. Axes that change direction or
shape rather than scale (theta, `memory_balance`, `memory_scales`, the bandwidths) compare
two arms at the same cap and are the more trustworthy half of the campaign. That is why the
curl verdict is reported above with more confidence than the horizon one.

**Nothing here is safe to adopt as a tuned value until the campaign is re-run with
`lam_max` raised.** The mechanism-level findings (memory is load-bearing; a bank does not
beat one well-placed scale; balance is inert) are ordinal and large, and are corroborated by
your independent runs, so they survive. The numeric optima do not.

## Suggested starting point for the UAV profile

Only the well-supported, scale-corrected items — deliberately excluding everything still
pending:

```yaml
mppi:
  K: 250          # free: K is not a lever (finding 6)
  T: <screen>     # 100-500; do NOT copy 150
stein:
  memory_gain: 15.0      # >0 is what matters; 60 won on trimodal but is unconfirmed
  memory_time: 5.5       # tau*v trail-length match to the campaign's winning 2.5 s at v=4
  memory_balance: 0.85   # inert; leave it
  memory_scales: 1       # ties Q=3 when placed at the midpoint
  fine_bandwidth: 0.94   # sqrt(h_f * h_c) with h_c = 11; recompute if h_f changes
  coarse_bandwidth: 6.6  # 0.6 x derived 11.0, the ratio that won on trimodal
  weight_stein: 3000.0   # keep high; 6000 scored better but 500 is 2.37x worse
  reference_speed: 1.8   # keep non-zero (finding 5)
```

---

# Addendum from the UAV deployment — 2026-08-01

Written after screening the above on the deployment's own density (means at ±12,
tr(Sigma) = 11) on the seed-518 map, 18 seeds per arm, offline ideal controller. Raw
per-seed rows: `results/archive/2026-08-05/csv/ablation.csv` (20 000 steps) and
`results/uav/ablation_screen.csv` (8 000 steps), regenerated by
`python -m ergodic_control_mppi.experiments.uav_ablation`.

## Read this first: `fine_bandwidth: 0.94` above is wrong for a wide density

The midpoint rule `sqrt(h_f * h_c)` was fitted at modes ±6. Applied at ±12 it collapses the
single active scale to 0.94, and **that one number costs the deployment its mode
visitation**. The memory repulsion kernel peaks at `sqrt(h/2)`, and that peak has to land at
0.55–0.80 of a mode radius (2.35–2.65 m here). At h = 0.94 the peak is 0.69 m — track
spacing. Nothing in the cost then perceives that a *mode* is saturated, so the vehicle
covers one basin beautifully and never leaves it.

Seeds reaching all three modes, 20 000 steps, 18 seeds:

| h | peak sqrt(h/2) | fraction of mode radius | all 3 modes | median occupancy MSE |
|---|---|---|---|---|
| 0.94 (the rule above) | 0.69 m | 0.28 | 9/18 | 3.73e-07 |
| **5.0** | 1.58 m | 0.64 | **16/18** | 3.83e-07 |
| 6.0 | 1.73 m | 0.70 | 15/18 | 4.09e-07 |
| 6.6 | 1.82 m | 0.74 | 13/18 | 4.12e-07 |
| 8.5 | 2.06 m | 0.84 | 0/6 (8 000 steps) | — |
| 11.0 (derived h_c) | 2.35 m | 0.95 | 1/6 (8 000 steps) | — |

It fails at both ends for opposite reasons: too narrow and nothing senses a finished basin,
too broad and the repulsion evicts the vehicle before a dwell qualifies as a visit. The
derived `h_c = tr(Sigma)` sits at exactly one mode radius, i.e. in the failing region —
which is a mechanism for *why* your tuned h_c = 0.6 x derived won. That ratio was right.
The error was stacking the midpoint rule on top of it.

**Corrected transfer rule: at Q = 1, place the single scale at the tuned coarse bandwidth
(~0.6 x derived h_c), not at the geometric midpoint.** Equivalently `h ≈ r_mode²`.

Occupancy MSE is flat across this whole range (3.7–4.1e-07), which is exactly why the
campaign could not have caught it: **Q = 1 at the midpoint does tie Q = 3 on MSE, and is
still wrong.** Mode visitation is a separate axis and needs to be scored separately.

## Confirmations, at the corrected h

- **Finding 2 holds, and the mechanism is now clear.** `stein.py` averages the bank as
  `total / Q`, so extra scales split weight rather than add resolution. Q = 3 over
  [0.08, 0.73, 6.6] spends two thirds of the memory on track-spacing detail and reaches all
  three modes on 1 seed in 6 — as bad as having no coarse scale at all. A bank cannot beat
  one well-placed scale; it can only dilute it.
- **Finding 6 (`K: 250`) holds.** p99 = 10.1 ms in the flown loop against a 16 ms deadline.
- **`memory_time` and `memory_gain` are already right at 5.5 and 15.0** — but not for the
  reason given above, and the trail-length argument does not generalise. Once h is
  basin-sized the memory saturates a basin quickly, so a *long* tau makes the whole
  workspace read as covered, flattens the excess field, and removes the gradient that
  drives transits. At h = 6.6, tau = 8 reached all three modes on **0 of 6** seeds. Spatial
  and temporal scale trade against each other; do not tune them independently.
- **`memory_gain: 60` does not transfer.** At the corrected h it is actively harmful:
  gain 8 → 4/6, 15 → 5/6, 30 → 2/6, 60 → **0/6**. With the lengthscale right the flow needs
  less push, not more. Whatever `generalization` says about k_M = 60 on trimodal, it should
  not be copied to a wide density.
- **`memory_balance` inert, confirmed.** 0.5 → 2/6 and 1.0 → 1/6 against 0.85 → 5/6.

## Process warning that applies to the whole campaign

XLA:GPU autotunes GEMM kernel selection by timing candidates **at compile time**, so a
machine under load picks different kernels, accumulates in a different order, and returns
different float32 results. Runs are bit-identical *within* a process, which is what makes
this so easy to miss — it only appears when the same config is compared across two
processes. In this closed loop it amplifies until it changes which modes are visited: one
measured pair of identical-config runs differed by 16 m of travel and by whether all three
modes were reached. It produced a clean, entirely spurious "gain = 60 wins" result here.

`ergodic_control_mppi/__init__` now sets `--xla_gpu_autotune_level=0` for every entry point
(~6% runtime). **Campaign cells compared across separate processes without this flag carry
an unquantified error term.** See https://openxla.org/xla/determinism. Note also that the
CPU and GPU backends are each deterministic but disagree with each other, so `device` is
part of a run's identity, not an implementation detail.

## Curl (`stein.theta`) — do NOT delete it

Your "the curl may be deleted entirely" hypothesis **does not transfer**. Measured on the
deployment density, 18 seeds x 20 000 steps, with a working temperature loop:

| theta | all 3 modes | occupancy MSE | Fourier | cycles |
|---|---|---|---|---|
| **0** | **13/18** | **3.776e-07** | 0.0567 | 2 |
| 15 | 18/18 | 4.284e-07 | **0.0367** | **3** |
| 30 | 18/18 | 3.866e-07 | 0.0506 | 2 |
| 45 | 18/18 | 6.079e-07 | 0.0552 | 1 |

theta = 0 gives the **best occupancy MSE of any arm we tested** and still loses 5 of 18
seeds' mode visitation. That is exactly how the hypothesis was formed: on the reference
environment `no_curl` won on ergodicity (-29%) and error (0.98x), and both of those metrics
are blind to whether a mode is ever reached. Keep theta, and keep the curl in `stein.py`.

theta = 15 beats 30 on Fourier (-27%), tour time (143 s vs 169 s) and cycles (3 vs 2),
losing 11% of occupancy MSE. Worth a look on trimodal if `generalization` has budget.

## One more process warning: fix the temperature before ranking anything

The deployment ran with `mppi.lam_max: 20`, which **capped the ESS controller out of
action**: lambda pinned at 20 from step ~250 and achieved ESS was 0.6% against an
`ess_target` of 0.3, so the MPPI update was argmin over rollouts rather than a weighted
average. Raising the cap to 1000 lets the loop converge to lambda = 151.8 on its own and
hold ESS at 34%; caps of 1e3/1e4/1e5 all converge to the same value, so it self-limits.

This matters to the campaign because **arm rankings measured in the argmin regime do not
survive the fix**. Our `fine_bandwidth` result moved from 9/18 to 17/18 for the *bad* value
once the temperature worked -- the ranking held, the effect size collapsed. If
`mppi_params.yaml` also caps lambda well below where the loop wants to sit, every ranking
in the campaign is measured on a degenerate weighting. Check it with
`uav_diagnostics ess`; it is a 4-minute CPU run.

Note also that rescaling the penalty weights does **not** substitute for raising the cap:
with the cap at 20, scaling oom/obstacle/boundary by 0.1 or 0.01 still pins lambda and
still gives ESS < 1%.

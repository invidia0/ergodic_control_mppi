# Legacy memory feedback (v1) — disposable

The six-parameter two-scale fading-memory term, superseded by the multiscale
over-coverage feedback. Kept only until the paper revision lands; delete this
folder then.

Extracted from commit `f9502e3` (branch `single-robot`). Live at the time in
`ergodic_control_mppi/mppi/core.py`, `parameters.py`, `config.py` and
`configs/mppi_params.yaml`.

## What was removed

Six knobs: `deficit_gate` (λ_d), `spiral_deficit` (λ_s), `repulsion_bandwidth`
(h_c), `spiral_bandwidth` (h_f), `repulsion_weight` (r_w), `spiral_weight` (s_w),
plus the `eject_fill_gated` toggle. The bandwidths survive, renamed
`coarse_bandwidth` / `fine_bandwidth` and now derived rather than tuned.

## Why

- λ_d and λ_s blended the per-particle *weights*, but `stein_repulsion` divides
  by the weight sum, so the effective mixing coefficient was
  `α_t = λ Σb / ((1-λ) Σa + λ Σb)` — drifting with time, bandwidth and
  occupancy, not λ. That is why `(λ_d, λ_s) = (1, 0)` had to be frozen by hand.
- The occupancy `o^h` was smoothed at scale h but compared against the
  *unsmoothed* `p*`, a bandwidth-dependent bias.
- The fill gate was a binary indicator — discontinuous, in tension with the
  continuity the closed-loop analysis assumes — and was evaluated at the memory
  samples, where the KDE contains a self-kernel term.
- `r_w` and `s_w` partly compensated for `max|∇κ_h| = √(2/h)·e^{-1/2}` varying
  with h, confounding gains with bandwidths.

## Measured replacement

Three parameters (τ_M, a, k_M) with derived spatial scales. At 20000 steps over
seeds {43, 44, 45}: ergodic error 1.02e-07 vs this version's 1.13e-07, equivalent
dwelling behaviour, 1.18x per-step cost. Full study in
`results/multiscale_memory/summary.txt` and
`results/multiscale_memory_horizon/summary.txt`.

## Files here

- `legacy_memory_flow.py` — the `_legacy_memory_flow` function as it stood.
- `mppi_params_legacy.yaml` — the shipped config of that era, with the tuned
  `stein` block and its "frozen design toggles" note.

# NNMM.jl “Large SD” (Scale Blow‑Up) Fix Report

## Problem
When running the NNMM MCMC with the 2→3 layer enabled (omics → phenotype), the NNMM.jl chain produced **unrealistically large EBV/EPV dispersion**:

- EBV/EPV standard deviations were extremely large (e.g. EBV sd ≈ 67, EPV sd ≈ 127 in the “speed_julia_f32/f64” benchmark runs).
- The 2→3 marker effects (omics effects) drifted upward over iterations, and the prediction term `X * α` grew iteration by iteration.

This was not a normal “label switching / non-identifiability” issue; it was a **scale divergence** driven by a bug in how the 2→3 residual vector was constructed before sampling marker effects.

## Root cause
In the 2→3 sampling step, we pass a residual vector `ycorr2` into `BayesABC!`/`BayesC0!`/etc. Those samplers assume:

> `ycorr2 = y − X*b − X*α` (i.e., the phenotype corrected for *all* non-marker effects and for the *current* marker effects)

Inside these samplers, each locus update temporarily “adds back” the current locus effect (conceptually like `ycorr2 += x_j * α_j_old`) before resampling `α_j`, and then re-subtracts the updated locus effect.

However, NNMM.jl was recomputing `ycorr2` as:

> `ycorr2 = y − X*b`

right before 2→3 marker sampling, without subtracting the current `X*α`. This caused a mismatch between what the sampler expects and what it received:

- The sampler’s internal “add back” step effectively **double-counted** the previous iteration’s effect.
- This induced **systematic drift** in `α` (the 2→3 marker effects), so `X*α` grew over iterations.
- The growing `X*α` then inflated EPV/EBV scale, producing the “large SD” symptom.

## Why correlations were still ~1.0 (old vs fixed) and “accuracy” looked OK
The buggy outputs were almost a **pure rescaling** of the fixed outputs (same directions/rankings, inflated magnitude). Two key points explain why correlations remained high:

1. **Correlation is scale-invariant**. If `old ≈ c * fixed + a` with `c > 0`, then `cor(old, fixed) ≈ 1` even when `c` is huge. This is why we observed:
   - `cor(EBV_old, EBV_fixed) ≈ 0.998`
   - `cor(EPV_old, EPV_fixed) ≈ 0.999`

2. **This bug created a positive-feedback scaling drift, not a reshuffling**. BayesC/BayesABC locus updates conceptually do:
   - start from a residual that already removed the current marker signal: `ycorr2 = y - X*b - X*α`
   - for each locus `j`: temporarily “add back” the current locus (`ycorr2 += x_j * α_j_old`), sample `α_j_new`, then subtract it.

   If we incorrectly pass `ycorr2 = y - X*b`, then the “add back” step uses:
   - `ycorr2_bug += x_j * α_j_old = (y - X*b) + x_j*α_j_old`

   compared to the correct partial residual for updating `j`:
   - `r_j = y - X*b - X_{-j}*α_{-j}`

   so the buggy update effectively uses:
   - `r_bug_j = r_j + (X*α)` (an extra full predicted term)

   That extra `+(X*α)` term pushes `α_new` toward `α_old` (a positive feedback loop), inflating magnitudes while largely preserving the shape across individuals. As a result, EBV/EPV can blow up in SD but remain highly correlated with the correctly-scaled outputs.

Finally, most benchmark “accuracy” metrics we reported were also correlations (`cor(pred, truth)`), which are similarly scale-invariant—so a scale blow-up can still produce “good accuracy” correlation values.

## Not equivalent to “restarting layer 2→3 from α=0 each iteration”
It’s tempting to think “`ycorr2 = y - X*b` is what you’d have if `α=0`”, so the buggy code is like re-initializing layer 2→3 every iteration. But BayesC/BayesABC **still uses the current `α`** through its internal “add back current locus effect” step (`ycorr2 += x_j * α_j_old`).

So even though the residual *starts* as if `α=0`, the per-locus updates immediately re-introduce the previous `α` and (because of the missing `-X*α` in `ycorr2`) do so in a way that adds an extra `+(X*α)` term, producing the positive-feedback scaling drift described above.

## Fix
File: `src/nnmm/mcmc_bayesian.jl`

We ensured that `ycorr2` is *always* computed as:

> `ycorr2 = y − X*b − Σ(Xomics * α)`

Specifically:

1. **Initialization (before MCMC loop)**: build `ycorr2` from `y - X*b`, then subtract the current omics effects `Xomics * α`.
2. **Per-iteration recomputation (after updating/transforms of omics)**: rebuild `ycorr2` the same way (`y - X*b - X*α`) before calling the 2→3 marker samplers.

This removes the double-counting pathway and stabilizes the marker-effect updates.

### Debug instrumentation (optional)
We added lightweight, opt-in logging to confirm scale behavior:

- `NNMM_DEBUG_SCALE=1` enables printing a few scale stats.
- `NNMM_DEBUG_SCALE_ITERS=K` controls how many early iterations to print.

This was used to verify that `ycorr2` correctly shrinks from the phenotype’s raw scale to the residual scale after subtracting `X*α`, and that `α` no longer drifts upward.

## Verification
After the fix, the “speed_julia_f32/f64” benchmark outputs returned to a sensible scale:

- `benchmarks/ebv_julia_speed_julia_f32.csv`: `std(EBV) ≈ 0.605`
- `benchmarks/ebv_julia_speed_julia_f64.csv`: `std(EBV) ≈ 0.605`
- `benchmarks/epv_julia_speed_julia_f32.csv`: `std(EPV) ≈ 1.158`
- `benchmarks/epv_julia_speed_julia_f64.csv`: `std(EPV) ≈ 1.158`

Additionally, with `NNMM_DEBUG_SCALE=1`, the 2→3 marker effects stopped showing iteration-to-iteration “explosion” and `X*α` remained stable.

## How to reproduce
These are example commands used for local verification (adapt paths as needed):

```bash
JULIA_DEPOT_PATH="$PWD/.julia_depot:/Users/haocheng/.julia" \
/Users/haocheng/.julia/juliaup/julia-1.11.7+0.aarch64.apple.darwin14/bin/julia \
  --project=. benchmarks/save_ebv_for_comparison.jl \
  --seed=42 --chain-length=1000 --burnin=200 \
  --estimate-pi=false --estimate-var1=false --estimate-var2=false \
  --double-precision=true --suffix=speed_julia_f64
```

Debug scale prints for the first few iterations:

```bash
NNMM_DEBUG_SCALE=1 NNMM_DEBUG_SCALE_ITERS=5 \
JULIA_DEPOT_PATH="$PWD/.julia_depot:/Users/haocheng/.julia" \
/Users/haocheng/.julia/juliaup/julia-1.11.7+0.aarch64.apple.darwin14/bin/julia \
  --project=. benchmarks/save_ebv_for_comparison.jl \
  --seed=42 --chain-length=50 --burnin=0 \
  --estimate-pi=false --estimate-var1=false --estimate-var2=false \
  --double-precision=true --suffix=debug_scale
```

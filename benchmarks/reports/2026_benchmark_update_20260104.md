# Benchmark Update (2026‑01‑04): Post “Large SD” Fix

## What changed
NNMM.jl previously produced extremely large EBV/EPV standard deviations due to a 2→3 residual construction bug (`ycorr2` missing the `-X*α` term before BayesC updates). This has been fixed; see `benchmarks/reports/2026_large_sd_fix_report.md`.

## Latest accuracy (rerun)

### NNMM.jl (rerun)
Run log: `benchmarks/logs/benchmark_accuracy_run_20260104.log`

- MCMC: seed=42, chain_length=1000, burnin=200, BayesC (both layers), activation=linear
- EBV accuracy: `cor(EBV, genetic_total)=0.8571`, `cor(EBV, genetic_direct)=0.0379`, `cor(EBV, genetic_indirect)=0.9393`
- EPV accuracy: `cor(EPV, genetic_total)=0.4979`, `cor(EPV, genetic_direct)=0.0136`, `cor(EPV, genetic_indirect)=0.5498`, `cor(EPV, trait1)=0.8269`
- EBV scale: `std(EBV)=0.5979`
- EPV scale: `std(EPV)=1.1597`

### PyNNMM (rerun)
Run log: `benchmarks/logs/pynnmm_benchmark_accuracy_run_20260104.log`

- MCMC: seed=42, chain_length=1000, burnin=200, BayesC (both layers), activation=linear
- EBV accuracy: `cor(EBV, genetic_total)=0.7909`, `cor(EBV, genetic_direct)=0.3838`, `cor(EBV, genetic_indirect)=0.6924`
- EPV accuracy: `cor(EPV, genetic_total)=0.2608`, `cor(EPV, genetic_direct)=-0.0593`, `cor(EPV, genetic_indirect)=0.3213`, `cor(EPV, trait1)=0.4831`
- EBV scale: `std(EBV)=0.5737`
- EPV scale: `std(EPV)=0.1659`

### PyNNMM (update 2026‑01‑17): fixed “HMC on observed omics”

Root cause: PyNNMM was HMC‑updating the middle‑layer omics **even when no omics are missing**, which breaks EPV (and distorts EBV partitioning). PyNNMM now runs HMC **only when omics are actually missing** (matching NNMM.jl’s behavior of conditioning on observed omics).

With the same configuration (seed=42, chain_length=1000, burnin=200, BayesC/BayesC, activation=linear), PyNNMM now gives:

- EBV accuracy: `cor(EBV, genetic_total)=0.8139`, `cor(EBV, genetic_direct)=0.0305`, `cor(EBV, genetic_indirect)=0.8947`
- EPV accuracy: `cor(EPV, genetic_total)=0.4978`, `cor(EPV, genetic_direct)=0.0135`, `cor(EPV, genetic_indirect)=0.5498`, `cor(EPV, trait1)=0.8269`
- EBV scale: `std(EBV)=0.5772`
- EPV scale: `std(EPV)=1.1591`

### PyNNMM (update 2026‑01‑17): fixed π sampling + multi‑trait marker‑variance update

Root causes (systematic EBV mismatch):
- **π sampling inversion**: PyNNMM’s `PiSampling::sample_pi` sampled `Beta(included+1, excluded+1)` while BayesC uses `π = P(δ=0)` (excluded). NNMM.jl samples `Beta(excluded+1, included+1)`. This inversion systematically changed inclusion probabilities and shrank the 1→2 layer.
- **Multi‑trait BayesC `var_effect` update mismatch**: NNMM.jl samples the 1→2 marker‑variance using **β** (including excluded markers) and `n = nMarkers` (not `sum(δ)`). PyNNMM was using **α** and `n = sum(δ)`.

After aligning both to NNMM.jl, with the same benchmark config (seed=42, chain_length=1000, burnin=200, BayesC/BayesC, activation=linear), PyNNMM now gives:

- EBV accuracy: `cor(EBV, genetic_total)=0.8578`, `cor(EBV, genetic_direct)=0.0392`, `cor(EBV, genetic_indirect)=0.9395`
- EPV accuracy: `cor(EPV, genetic_total)=0.4978`, `cor(EPV, genetic_direct)=0.0136`, `cor(EPV, genetic_indirect)=0.5498`, `cor(EPV, trait1)=0.8269`
- EBV scale: `std(EBV)=0.5960`
- EPV scale: `std(EPV)=1.1591`

## Cross‑package parity (current)
Using the same dataset/seed settings, comparing NNMM.jl outputs from:
- `benchmarks/results/ebv_julia.csv`
- `benchmarks/results/epv_julia.csv`

to PyNNMM outputs saved into NNMM.jl’s benchmark folder (seed=42):
- `benchmarks/results/ebv_pynnmm_pynnmm_default_s42.csv`
- `benchmarks/results/epv_pynnmm_pynnmm_default_s42.csv`

Results:
- EBV parity: Pearson ≈ 0.9997; scale ratio (Py/Julia) ≈ 0.9970
- EPV parity: Pearson ≈ 1.0000; scale ratio (Py/Julia) ≈ 0.9998

Old PyNNMM snapshots (before the 2026‑01‑17 fix) are preserved as:
- `benchmarks/results/ebv_pynnmm_pynnmm_default_s42_before_20260117.csv`
- `benchmarks/results/epv_pynnmm_pynnmm_default_s42_before_20260117.csv`

And the “post‑HMC / pre‑π+variance” snapshot is preserved as:
- `benchmarks/results/ebv_pynnmm_pynnmm_default_s42_before_20260117_pi_variance_fix.csv`
- `benchmarks/results/epv_pynnmm_pynnmm_default_s42_before_20260117_pi_variance_fix.csv`

Conclusion: NNMM.jl and PyNNMM are now **numerically matching for both EBV and EPV** on this benchmark.

## Why the old “wrong” NNMM.jl still had good accuracy
The old buggy NNMM.jl outputs with huge SD are almost a **pure rescaling** of the fixed outputs:

- EBV(old) vs EBV(fixed): Pearson ≈ 0.998
- EPV(old) vs EPV(fixed): Pearson ≈ 0.999
- Scale factor was ~O(10^2) (i.e., SD blew up, but rankings/directions were nearly unchanged).

Most reported “accuracy” metrics are correlations (`cor(pred, truth)`), which are **invariant to multiplying predictions by a constant**. So a scale blow‑up can still show “good accuracy” even when the magnitude is wrong.

## Speed (status)
I have **not** completed a fresh rerun of the 5‑repeat speed benchmark after the fix (a rerun was started but interrupted). The last stable speed summary remains in:
- `benchmarks/reports/speed_benchmark_report.md`

If you want, we can rerun speed with 1 warm‑up + 5 repeats for:
- NNMM.jl f32/f64 (via `double_precision=false/true`)
- PyNNMM float32/float64 core (requires selecting the corresponding PyNNMM build)

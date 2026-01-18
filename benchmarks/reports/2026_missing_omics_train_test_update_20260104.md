# Missing Omics Train/Test Benchmark — Rerun + Comparison

Date: 2026-01-04

This note reruns the **full grid** benchmark (`0%:10%:100%` train missing × `{0%,100%}` test missing) after recent NNMM.jl algorithm updates, and compares results to the previous full-grid run.

## Runs compared

### Previous full-grid run (baseline)

- Log: `benchmarks/benchmark_missing_omics_train_test_run_20260105.log`
- Parsed summary CSV (from the log’s printed table): `benchmarks/missing_omics_train_test_results_baseline_20260105_summary.csv`

### Rerun (current)

- Log: `benchmarks/benchmark_missing_omics_train_test_run_20260104_212720.log`
- Results CSV (full metrics): `benchmarks/missing_omics_train_test_results.csv`
- Snapshot: `benchmarks/missing_omics_train_test_results_20260104_212720.csv`

Plot:

- Latest plot: `benchmarks/missing_omics_train_test_plot.png`
- Snapshot: `benchmarks/missing_omics_train_test_plot_20260104_212720.png`
- Previous plot backup: `benchmarks/archive_missing_omics_train_test/`

## How the rerun was executed

Command (same settings as the baseline full grid):

```bash
export JULIA_DEPOT_PATH="$PWD/.julia_depot:/Users/haocheng/.julia"
julia --project=. benchmarks/benchmark_missing_omics_train_test.jl \
  --seed=42 --chain-length=1000 --burnin=200 \
  --test-frac=0.2 \
  --missing-mode=individual \
  --train-missing-grid=0:0.1:1 --test-missing-pcts=0,1 | \
  tee benchmarks/benchmark_missing_omics_train_test_run_20260104_212720.log
```

Plot generation (this environment does **not** have Python `matplotlib`, so I used an R fallback):

```bash
Rscript benchmarks/plot_missing_omics_train_test.R
```

## Rerun summary (EBV on test; EPV on test)

| Train missing | Test missing | EBV(test,total) | EBV(test,indir) | EPV(test,total) | EPV(test,trait) | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 0% | 0.8528 | 0.9415 | 0.5008 | 0.8311 | 72.9 |
| 0% | 100% | 0.8412 | 0.9175 | 0.8419 | 0.4479 | 46.5 |
| 10% | 0% | 0.8538 | 0.9351 | 0.5005 | 0.8307 | 50.7 |
| 10% | 100% | 0.8420 | 0.9078 | 0.8407 | 0.4465 | 59.2 |
| 20% | 0% | 0.8558 | 0.9275 | 0.5006 | 0.8303 | 57.8 |
| 20% | 100% | 0.8452 | 0.8971 | 0.8432 | 0.4495 | 65.5 |
| 30% | 0% | 0.8614 | 0.9228 | 0.5000 | 0.8307 | 74.4 |
| 30% | 100% | 0.8499 | 0.8876 | 0.8489 | 0.4412 | 87.8 |
| 40% | 0% | 0.8620 | 0.9186 | 0.5003 | 0.8305 | 89.2 |
| 40% | 100% | 0.8539 | 0.8841 | 0.8521 | 0.4491 | 45.2 |
| 50% | 0% | 0.8598 | 0.9069 | 0.5001 | 0.8305 | 46.5 |
| 50% | 100% | 0.8467 | 0.8636 | 0.8459 | 0.4424 | 50.6 |
| 60% | 0% | 0.8588 | 0.8959 | 0.4990 | 0.8293 | 342.5 |
| 60% | 100% | 0.8472 | 0.8434 | 0.8450 | 0.4424 | 41.1 |
| 70% | 0% | 0.8518 | 0.8774 | 0.4962 | 0.8285 | 1044.2 |
| 70% | 100% | 0.8380 | 0.8115 | 0.8375 | 0.4453 | 43.6 |
| 80% | 0% | 0.8426 | 0.8633 | 0.4954 | 0.8254 | 44.9 |
| 80% | 100% | 0.8138 | 0.7706 | 0.8121 | 0.4277 | 44.2 |
| 90% | 0% | 0.8366 | 0.8480 | 0.4970 | 0.8269 | 43.7 |
| 90% | 100% | 0.7729 | 0.7073 | 0.7705 | 0.3997 | 48.6 |
| 100% | 0% | 0.8212 | 0.7827 | 0.4901 | 0.7671 | 46.6 |
| 100% | 100% | 0.7867 | 0.7048 | NaN | NaN | 44.4 |

![Missing omics train/test plot (rerun)](missing_omics_train_test_plot.png)

## Comparison vs baseline

Comparison CSV (old vs new + deltas): `benchmarks/missing_omics_train_test_comparison_20260104_212720.csv`

Key changes (finite rows only):

- `EBV(test,total)`: mean Δ `+0.00235`, max |Δ| `0.0556` (at train=100%, test=100%).
- `EBV(test,indir)`: mean Δ `+0.00213`, max |Δ| `0.0427` (at train=100%, test=100%).
- `EPV(test,total)`: mean Δ `+0.000407`, max |Δ| `0.0102` (at train=90%, test=100%).
- `EPV(test,trait)`: mean Δ `-0.00171`, max |Δ| `0.0446` (at train=100%, test=0%).

Timing is **not stable** across runs (JIT + OS noise). The rerun has two large runtime outliers:

- train=60%, test=0%: `342.5s`
- train=70%, test=0%: `1044.2s`

## Important anomaly: EPV is NaN at (train=100%, test=100%)

In the rerun, `EPV(test,*)` becomes `NaN` only for the extreme configuration where **all omics are missing in both train and test** (`train_missing_pct=1.0`, `test_missing_pct=1.0`).

This is **not** an “empty EPV” case, and it’s not just a “constant EPV” corner case:

- The EPV sample values themselves start to contain `NaN`s mid-chain, so the reported EPV(test) correlations become `NaN`.
- The first `NaN`s originate from the **2→3 residual variance** (`mme2.R.val`) becoming `NaN` during MCMC variance sampling (`sample_variance(ycorr2, ...)` in `src/core/variance_components.jl`), which can only happen if `ycorr2` already contains `NaN`s.
- Once `mme2.R.val` is `NaN`, the **BayesC/BayesABC** update for 2→3 effects uses `probDelta1 = ...` which becomes `NaN`, and Julia’s `rand() < NaN` is always `false`. That forces the sampler into the `δ=0` branch for every effect, collapsing `α` (and therefore `weights_NN`) to an all-zero vector (finite).
- After the collapse, **per-sample** EBV/EPV predictions for the test set become constant (all zeros), so **per-sample correlations** are undefined (`sd=0`). However, the **cumulative posterior mean** EBV can remain stable because correlation is scale-invariant and the chain’s early samples can dominate the running mean.

I diagnosed this with a local debug run using `--keep-output=true` to inspect per-iteration sample files. The resulting `benchmarks/epv_nan_debug_*` output folders are intentionally not tracked because they can be hundreds of MB.

## Update (2026-01-17): Fixed

This NaN issue was caused by numerical instability in the latent-omics sampler under **linear activation** when *all* omics are missing. NNMM.jl now uses a **closed-form conjugate Gaussian** update for the linear-activation latent-omics step (instead of HMC), and rerunning `(train_missing_pct=1.0, test_missing_pct=1.0)` with `chain_length=1000, burnin=200` no longer produces `NaN` EPV(test,*) correlations.

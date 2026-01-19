# Missing Omics Train/Test Benchmark — Results (2026-01-16)

Artifacts:
- Results CSV snapshot: `benchmarks/results/missing_omics_train_test_results_20260116_230206.csv`
- Full run log: `benchmarks/logs/benchmark_missing_omics_train_test_run_20260116_230206.log`

Configuration (from log):
- Dataset: `simulated_omics_data` (3534 individuals, 927 SNPs after QC, 10 omics)
- Seed: 42; chain length: 1000; burnin: 200; test fraction: 0.2
- Missing mode (train): `individual` (entire-omics missing per selected individual)
- Activation (2nd layer): `linear`
- Missing-latent sampler (2nd layer): conjugate Gaussian update (linear activation)

## Grid results (5 train missing × 2 test missing)

| Train missing | Test missing | EBV(test,total) | EBV(test,indir) | EPV(test,total) | EPV(test,trait) | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 0% | 0.8528 | 0.9415 | 0.5008 | 0.8311 | 59.4 |
| 0% | 100% | 0.8288 | 0.9094 | 0.8285 | 0.4390 | 41.7 |
| 25% | 0% | 0.8614 | 0.9263 | 0.5006 | 0.8305 | 34.7 |
| 25% | 100% | 0.8363 | 0.8788 | 0.8339 | 0.4339 | 38.0 |
| 50% | 0% | 0.8600 | 0.9061 | 0.5001 | 0.8304 | 37.6 |
| 50% | 100% | 0.7996 | 0.8181 | 0.7989 | 0.4285 | 48.9 |
| 75% | 0% | 0.8487 | 0.8714 | 0.4954 | 0.8263 | 40.0 |
| 75% | 100% | 0.4606 | 0.4459 | 0.4604 | 0.3140 | 51.5 |
| 100% | 0% | 0.8346 | 0.8058 | 0.4929 | 0.7930 | 48.5 |
| 100% | 100% | 0.6918 | 0.5961 | 0.6918 | 0.3527 | 48.3 |

Finiteness check: PASS (no NaNs/Infs in EBV/EPV outputs)

## Confirmation: train missing 75% vs 100% (test missing = 100%)

| Seed | EBV(test,total) @75% | EBV(test,total) @100% | Δ (75−100) | Time @75% (s) | Time @100% (s) |
| --- | --- | --- | --- | --- | --- |
| 41 | 0.3473 | 0.6501 | -0.3029 | 75.8 | 45.3 |
| 42 | 0.4606 | 0.6918 | -0.2311 | 77.2 | 39.9 |
| 43 | 0.3594 | 0.6472 | -0.2879 | 82.2 | 55.0 |
| 44 | 0.4012 | 0.6889 | -0.2877 | 88.3 | 44.7 |
| 45 | 0.2755 | 0.6757 | -0.4002 | 77.1 | 41.9 |

Δ summary over 5 seeds: mean -0.3020, std 0.0614

Per-seed CSV snapshots:
- `benchmarks/results/confirm_75_vs_100_train_missing_test100_seed41_20260116_233419.csv`
- `benchmarks/results/confirm_75_vs_100_train_missing_test100_seed42_20260116_233419.csv`
- `benchmarks/results/confirm_75_vs_100_train_missing_test100_seed43_20260116_233419.csv`
- `benchmarks/results/confirm_75_vs_100_train_missing_test100_seed44_20260116_233419.csv`
- `benchmarks/results/confirm_75_vs_100_train_missing_test100_seed45_20260116_233419.csv`

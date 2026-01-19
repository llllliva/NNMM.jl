# Missing Omics Train/Test Benchmark — NNMM.jl vs PyNNMM (seed=42)

Artifacts:
- NNMM.jl results CSV: `benchmarks/results/missing_omics_train_test_results_20260116_230206.csv`
- PyNNMM results CSV: `benchmarks/results/missing_omics_train_test_results_pynnmm_seed42_v3.csv`
- PyNNMM runner: `benchmarks/scripts/benchmark_missing_omics_train_test_pynnmm.py`

Notes:
- PyNNMM computes `EPV(test,*)` for test-missing=100% using EBV as a proxy (PyNNMM does not expose `EPV_Output_NonLinear` for unphenotyped IDs).
- PyNNMM no longer produces NaNs for the extreme case train=100%, test=100% (all omics missing everywhere).

## Grid results — NNMM.jl

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

## Grid results — PyNNMM (v3)

| Train missing | Test missing | EBV(test,total) | EBV(test,indir) | EPV(test,total) | EPV(test,trait) | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 0% | 0.8534 | 0.9411 | 0.5009 | 0.8311 | 44.5 |
| 0% | 100% | 0.8410 | 0.9162 | 0.8410 | 0.4463 | 35.4 |
| 25% | 0% | 0.8610 | 0.9259 | 0.5006 | 0.8305 | 44.0 |
| 25% | 100% | 0.8529 | 0.8940 | 0.8529 | 0.4470 | 36.3 |
| 50% | 0% | 0.8598 | 0.9065 | 0.5001 | 0.8304 | 44.7 |
| 50% | 100% | 0.8478 | 0.8634 | 0.8478 | 0.4448 | 36.8 |
| 75% | 0% | 0.8458 | 0.8691 | 0.4956 | 0.8263 | 46.8 |
| 75% | 100% | 0.8309 | 0.7962 | 0.8309 | 0.4442 | 38.5 |
| 100% | 0% | 0.8240 | 0.8130 | 0.4965 | 0.8016 | 47.4 |
| 100% | 100% | 0.8128 | 0.7215 | 0.8128 | 0.4327 | 41.0 |

## Δ (PyNNMM v3 − NNMM.jl)

| Train missing | Test missing | Δ EBV(test,total) | Δ EBV(test,indir) | Δ EPV(test,total) | Δ EPV(test,trait) | Time ratio (Py/JL) |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 0% | +0.0006 | -0.0004 | +0.0001 | -0.0000 | 0.748 |
| 0% | 100% | +0.0122 | +0.0069 | +0.0125 | +0.0073 | 0.849 |
| 25% | 0% | -0.0004 | -0.0005 | -0.0000 | +0.0000 | 1.268 |
| 25% | 100% | +0.0166 | +0.0152 | +0.0190 | +0.0131 | 0.955 |
| 50% | 0% | -0.0002 | +0.0004 | -0.0000 | +0.0000 | 1.188 |
| 50% | 100% | +0.0481 | +0.0453 | +0.0489 | +0.0163 | 0.752 |
| 75% | 0% | -0.0029 | -0.0023 | +0.0002 | +0.0001 | 1.170 |
| 75% | 100% | +0.3703 | +0.3503 | +0.3704 | +0.1302 | 0.747 |
| 100% | 0% | -0.0106 | +0.0072 | +0.0036 | +0.0086 | 0.977 |
| 100% | 100% | +0.1210 | +0.1254 | +0.1210 | +0.0800 | 0.848 |

## Takeaways

- Test-missing=0%: PyNNMM v3 is very close to NNMM.jl across the train-missing grid (Δ EBV/EPV generally ≤ ~0.01).
- Test-missing=100%: PyNNMM v3 remains materially higher than NNMM.jl, especially at train-missing=75% (Δ EBV ≈ +0.37). This indicates there is still a cross-implementation mismatch in the “all-omics-missing for unphenotyped IDs” regime.

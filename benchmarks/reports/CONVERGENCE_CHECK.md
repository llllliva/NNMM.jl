# MCMC Convergence Check Results

## Purpose

This document verifies that the MCMC chain has converged and that the benchmark results at `chain_length=1000` are reliable.

## Configuration

- **Seeds**: 42, 123, 456, 789, 2024
- **Chain length**: 1000
- **Burnin**: 200 (20% of chain length)
- **Method**: BayesC (both layers)
- **Activation**: linear
- **Constraint**: `true` (default) - independent variances per trait

## Dataset

- **Name**: simulated_omics_data
- **Individuals**: 3534
- **SNPs**: 1000 (927 after MAF filtering)
- **Omics**: 10
- **Target heritability**: 0.5 (20% direct, 80% indirect)

---

## Convergence Results (5 Seeds × 1000 iterations)

| Seed | cor(EBV, total) | cor(EBV, direct) | cor(EBV, indirect) | Time (s) |
|------|-----------------|------------------|---------------------|----------|
| 42 | 0.8549 | 0.0399 | 0.9358 | 301.6 |
| 123 | 0.8556 | 0.0411 | 0.9361 | 267.5 |
| 456 | 0.8547 | 0.0405 | 0.9354 | 380.8 |
| 789 | 0.8546 | 0.0394 | 0.9358 | 228.4 |
| 2024 | 0.8563 | 0.0407 | 0.9370 | 226.4 |
| **Mean** | **0.8552** | **0.0403** | **0.9360** | 281.0 |
| **Std** | **0.0007** | **0.0007** | **0.0006** | 63.9 |

---

## Analysis

### Variance Across Seeds

| Metric | Mean | Std Dev | CV (%) |
|--------|------|---------|--------|
| genetic_total | 0.8552 | 0.0007 | **0.08%** |
| genetic_direct | 0.0403 | 0.0007 | 1.7% |
| genetic_indirect | 0.9360 | 0.0006 | **0.06%** |

*CV = Coefficient of Variation*

### Conclusion

✅ **Chain has converged**

- Standard deviation across 5 seeds: **0.0007** (< 0.1%)
- Coefficient of variation: **0.08%** (extremely low)
- All metrics are stable across different random seeds
- Results are robust and reproducible

---

## Recommendations

1. **chain_length=1000** with **burnin=200** is sufficient for this dataset
2. Results are stable across different random seeds (σ = 0.0007)
3. The benchmark results at chain_length=1000 are reliable for PyNNMM comparison
4. **constraint=true** (default) provides same accuracy with parallelization benefit

---

## Running the Convergence Check

```bash
cd /Users/haocheng/Github/AFOCUS/NNMM.jl

# Single seed convergence check
julia --project=. benchmarks/check_convergence.jl

# Multi-seed convergence check (5 seeds)
julia --project=. benchmarks/check_convergence_seeds.jl
```

---

*Generated: 2025-12-31 (constraint=true benchmark)*


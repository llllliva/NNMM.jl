# MCMC Convergence Check Results - Missing Omics Data

## Purpose

This document verifies MCMC convergence when 30% of omics values are missing.
Missing values are imputed using HMC (Hamiltonian Monte Carlo) during MCMC.

## Configuration

- **Chain length**: 1000
- **Burnin**: 200
- **Missing percentage**: 30%
- **Seeds tested**: 42, 123, 456, 789, 2024
- **Method**: BayesC (both layers)
- **Activation**: linear

## Dataset

- **Name**: simulated_omics_data
- **Individuals**: 3534
- **SNPs**: 1000 (927 after MAF filtering)
- **Omics**: 10
- **Target heritability**: 0.5 (20% direct, 80% indirect)

---

## Convergence Results

| Seed | Missing Cells | cor(EBV, total) | cor(EBV, direct) | cor(EBV, indirect) | Time (s) |
|------|---------------|-----------------|------------------|---------------------|----------|
| 42 | 10600 | **0.7873** | 0.0685 | 0.8460 | 93.3 |
| 123 | 10600 | **0.7768** | 0.0757 | 0.8306 | 60.3 |
| 456 | 10600 | **0.7759** | 0.0500 | 0.8424 | 57.8 |
| 789 | 10600 | **0.7928** | 0.0844 | 0.8442 | 55.3 |
| 2024 | 10600 | **0.7723** | 0.0641 | 0.8314 | 55.0 |
| **Mean** | - | **0.7810** | 0.0685 | 0.8389 | 64.3 |
| **Std** | - | 0.0087 | 0.0129 | 0.0073 | 16.3 |

---

## Analysis

### Convergence Assessment

✅ **CONVERGED** - Standard deviation 0.0087 < 0.02

### Comparison to Full Omics Baseline

| Scenario | Mean Accuracy | Std Dev |
|----------|---------------|---------|
| Full omics (0% missing) | 0.8552 | 0.0007 |
| 30% missing omics | 0.7810 | 0.0087 |

**Accuracy degradation**: 8.67%

---

## Conclusion

1. The MCMC chain converges reliably with 30% missing omics data
2. HMC-based imputation produces consistent results across different seeds
3. Accuracy degrades by ~8.7% with 30% missing data, which is expected
4. These baseline values can be used for PyNNMM parity testing:
   - **Target accuracy**: 0.7810 ± 0.02 for 30% missing omics

---

*Generated: 2025-12-31*


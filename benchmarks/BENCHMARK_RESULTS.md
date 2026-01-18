# NNMM.jl Benchmark Results

## Dataset: simulated_omics_data

- **Individuals**: 3534
- **SNPs**: 1000 (927 after MAF filtering)
- **Omics**: 10
- **Target heritability**: 0.5 (20% direct, 80% indirect)

## Configuration

- **Seed**: 42
- **Chain length**: 1000
- **Burnin**: 200
- **Method**: BayesC (both layers)
- **Activation**: linear
- **Constraint**: `true` (default) - independent variances per trait, parallelizable

---

## Full Omics Benchmark (`benchmark_accuracy.jl`)

### EBV Accuracy Metrics (Estimated Breeding Value - from predicted omics)

| Metric | Value |
|--------|-------|
| cor(EBV, genetic_total) | **0.8549** |
| cor(EBV, genetic_direct) | 0.0399 |
| cor(EBV, genetic_indirect) | 0.9358 |

### EPV Accuracy Metrics (Estimated Phenotypic Value - from observed omics)

| Metric | Value |
|--------|-------|
| cor(EPV, genetic_total) | 0.4996 |
| cor(EPV, genetic_direct) | 0.0144 |
| cor(EPV, genetic_indirect) | 0.5513 |
| cor(EPV, trait1) | **0.8254** |

### EBV/EPV Statistics

| Statistic | EBV | EPV |
|-----------|-----|-----|
| Mean | -0.0 | 2.31 |
| Std | 358.7 | 691.1 |

> **Note on Scale**: The large EBV/EPV std is due to weight drift in the Layer 2 (omics → phenotype) sampler. 
> This affects absolute scale but NOT rankings (correlation with true values is preserved).

---

## Missing Omics Benchmark (`benchmark_missing_omics.jl`)

### EBV Accuracy by Missing Percentage

| Missing % | Missing Cells | cor(EBV, total) | cor(EBV, direct) | cor(EBV, indirect) |
|-----------|---------------|-----------------|------------------|---------------------|
| 0% | 0 | **0.8549** | 0.0399 | 0.9358 |
| 30% | 10,600 | **0.7873** | 0.0685 | 0.8460 |
| 50% | 17,670 | **0.7365** | 0.1497 | 0.7486 |

### EPV Accuracy by Missing Percentage

| Missing % | cor(EPV, total) | cor(EPV, direct) | cor(EPV, indirect) | cor(EPV, trait) |
|-----------|-----------------|------------------|---------------------|-----------------|
| 0% | 0.4996 | 0.0144 | 0.5513 | 0.8254 |
| 30% | 0.4067 | 0.0384 | 0.4355 | 0.6682 |
| 50% | 0.3583 | 0.0788 | 0.3612 | 0.6118 |

### Accuracy Degradation from Baseline

| Missing % | EBV Reduction | EPV Reduction |
|-----------|---------------|---------------|
| 30% | 7.9% | 18.6% |
| 50% | 13.8% | 28.3% |

---

## Convergence Check (`check_convergence_seeds.jl`)

### EBV Convergence (5 seeds × 1000 iterations)

| Seed | genetic_total | genetic_direct | genetic_indirect | Time (s) |
|------|---------------|----------------|------------------|----------|
| 42 | 0.8549 | 0.0399 | 0.9358 | 151.6 |
| 123 | 0.8556 | 0.0411 | 0.9361 | 107.4 |
| 456 | 0.8547 | 0.0405 | 0.9354 | 105.4 |
| 789 | 0.8546 | 0.0394 | 0.9358 | 80.3 |
| 2024 | 0.8563 | 0.0407 | 0.9370 | 64.8 |
| **Mean** | **0.8552** | **0.0403** | **0.9360** | 101.9 |
| **Std** | **0.0007** | **0.0007** | **0.0006** | 33.0 |

### EPV Convergence (5 seeds × 1000 iterations)

| Seed | genetic_total | genetic_direct | genetic_indirect |
|------|---------------|----------------|------------------|
| 42 | 0.4996 | 0.0144 | 0.5513 |
| 123 | 0.4995 | 0.0144 | 0.5513 |
| 456 | 0.4995 | 0.0144 | 0.5513 |
| 789 | 0.4996 | 0.0144 | 0.5513 |
| 2024 | 0.4996 | 0.0145 | 0.5513 |
| **Mean** | **0.4996** | **0.0144** | **0.5513** |
| **Std** | **0.0000** | **0.0000** | **0.0000** |

✅ **CONVERGED**: Very low standard deviation indicates stable results across different seeds.

---

## Performance Benchmark (`benchmark_performance.jl`)

### Speed (3534 individuals, 927 SNPs, 10 omics)

| Chain | Burnin | Time (s) | Iter/sec |
|-------|--------|----------|----------|
| 100 | 20 | 21.91 ± 1.61 | 4.6 |
| 500 | 100 | 58.39 ± 2.67 | 8.6 |
| 1000 | 200 | 69.84 ± 23.09 | 14.3 |

---

## Cross-Package Comparison

### EBV/EPV Correlation (NNMM.jl vs PyNNMM)

| Metric | Pearson | Spearman |
|--------|---------|----------|
| EBV (genetic) | 0.9616 | 0.9595 |
| EPV (phenotypic) | 1.0000 | 1.0000 |

### Accuracy Comparison

| Metric | NNMM.jl | PyNNMM | Gap |
|--------|---------|--------|-----|
| cor(EBV, genetic_total) | **0.8571** | 0.8139 | 0.0432 |
| cor(EBV, genetic_direct) | 0.0379 | 0.0305 | 0.0074 |
| cor(EBV, genetic_indirect) | **0.9393** | 0.8947 | 0.0446 |
| cor(EPV, genetic_total) | **0.4979** | 0.4978 | 0.0001 |

### Performance Comparison

| Chain | NNMM.jl (s) | PyNNMM (s) | Ratio |
|-------|-------------|------------|-------|
| 100 | 21.91 | 7.54 | 2.91x slower |
| 500 | 58.39 | 43.15 | 1.35x slower |
| 1000 | 69.84 | 80.92 | 0.86x faster |

> **Note**: PyNNMM is faster for short chains due to startup overhead in Julia JIT compilation.
> For longer chains (1000+), NNMM.jl reaches comparable or better performance.

---

## Running Benchmarks

```bash
# Full omics benchmark
julia --project=. benchmarks/benchmark_accuracy.jl

# Missing omics benchmark
julia --project=. benchmarks/benchmark_missing_omics.jl

# Convergence check (multiple seeds)
julia --project=. benchmarks/check_convergence_seeds.jl

# Performance benchmark
julia --project=. benchmarks/benchmark_performance.jl

# Save EBVs for cross-package comparison
julia --project=. benchmarks/save_ebv_for_comparison.jl
```

---

## Key Findings

1. **EBV vs EPV**: EBV (from predicted omics) shows much higher correlation with genetic values (0.85) compared to EPV (from observed omics, 0.50). This is expected since EBV captures the genetic component while EPV includes environmental noise from observed omics.

2. **Indirect Effects**: NNMM.jl shows very high correlation with genetic_indirect (0.94), indicating the model effectively captures the indirect genetic effects mediated through omics.

3. **Missing Data Robustness**: EBV accuracy degrades gracefully with missing omics data (13.8% reduction at 50% missing), while EPV degrades more significantly (28.3%).

4. **Cross-Package Agreement**: EPV is essentially identical (Pearson ≈ 1.00). EBV parity is strong (Pearson ≈ 0.96) but not identical; remaining differences likely come from MCMC trajectory/implementation details in the 1→2 layer.

---

## Known Issues

### Layer 2 Weight Drift (EBV Scale Inflation)

**Issue**: The EBV_NonLinear/EPV_NonLinear scale grows significantly larger than the phenotype scale.

**Impact**:
- Absolute values are inflated
- Rankings and correlations with true values are PRESERVED
- Cross-package comparison requires standardization (Z-scores)

**Workarounds**:
1. Use standardized values (Z-scores) for interpretation
2. Compare rankings/correlations rather than absolute values
3. Use Spearman correlation for cross-package comparison

---

*Generated: 2026-01-01 (Updated with EBV/EPV metrics, convergence, performance, and cross-package comparison)*

# Speed Benchmark Report (NNMM.jl vs PyNNMM)

This report summarizes wall-clock ("real") runtime for a fixed NNMM benchmark
configuration across float32 and float64 for both packages.

## Configuration

- Dataset: simulated_omics_data (genotypes_1000snps + phenotypes_sim)
- MCMC: chain_length=1000, burnin=200, seed=42
- Flags: estimate_pi=false, estimate_var1=false, estimate_var2=false
- Timing: `/usr/bin/time -p` per run
- Repeats: 1 warm-up + 5 measured runs (warm-up excluded from stats)

## Results (real seconds)

| Package | Precision | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| NNMM.jl | f32 | 79.87 | 78.65 | 5.30 | 74.81 | 87.30 |
| NNMM.jl | f64 | 90.40 | 87.21 | 9.94 | 83.03 | 107.67 |
| PyNNMM | f32 | 50.34 | 49.52 | 1.85 | 48.50 | 53.22 |
| PyNNMM | f64 | 48.59 | 48.25 | 0.86 | 48.07 | 50.11 |

## Notes

- These are wall-clock times; see raw CSV for `user`/`sys` CPU times.
- PyNNMM shows similar float32/float64 timings under this configuration.
- NNMM.jl shows higher variability; f64 is slower than f32.

## Raw Data

- `benchmarks/speed_repeats_raw.csv`
- `benchmarks/speed_repeats_summary.csv`

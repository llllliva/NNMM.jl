# Changelog

All notable changes to NNMM.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed variance estimation bug when using train/test split with missing phenotypes
  - `sample_variance` now uses effective sample size (n_effective) instead of total individuals (n_total)
  - Individuals with `invweights=0` (prediction-only) are correctly excluded from variance degrees of freedom
  - Layer 2→3 now correctly sets `invweights2=0` for individuals with missing phenotypes
  - This fix significantly improves agreement with JWAS (r increased from ~0.94 to ~0.996 on training set)
- Fixed Layer 2→3 MCMC incorrectly using partial-connected logic (`is_nnbayes_partial` branches)
  - Layer 2→3 now correctly uses fully-connected path regardless of Layer 1→2 connectivity
- Fixed output file handling for partial-connected networks
  - EBV output now uses consistent trait names (`mme.lhsVec`) for both file creation and writing

### Changed
- Refactored `wArray2` to always be initialized (consistent with `wArray1` in Layer 1→2)
  - `wArray2[1]` is now used for single-trait Layer 3 (equivalent to `ycorr2`)
  - Prepares codebase for future multi-trait Layer 3 support
- **API improvement**: Unified `omics_name` and `phenotype_name` into single `traits` parameter
  - Old: `Equation(..., omics_name=["omic1"])` or `Equation(..., phenotype_name=["trait1"])`
  - New: `Equation(..., traits=["omic1"])` or `Equation(..., traits=["trait1"])`
  - Backward compatible: `omics_name` and `phenotype_name` still work as deprecated aliases
- Changed `activation_function` type from `String` to `Union{String, Function}` in `Equation` struct
  - Prepares for future user-defined activation function support
  - Currently shows clear error message if a Function is passed (not yet implemented)

## [0.1.0] - 2026-01-18

### Added
- Initial release with `Layer`/`Equation`/`runNNMM` API
- Support for Bayesian methods: BayesA, BayesB, BayesC, BayesL, RR-BLUP, GBLUP
- Activation functions: `linear`, `sigmoid`, `tanh`, `relu`, `leakyrelu`
- Hamiltonian Monte Carlo (HMC) sampling for missing intermediate omics
- Built-in simulated datasets via `NNMM.Datasets`
- Fully-connected neural network architecture
- Flexible missing data handling in intermediate layers
- Multi-threaded parallel computing support
- GWAS analysis via model frequency
- Estimated Breeding Values (EBV) output

### Known Limitations
- User-defined activation functions not yet supported (use built-in functions)
- Multi-trait phenotypes in Layer 3 not yet supported
- GBLUP not supported for Layer 2→3 equations
- Binary traits not yet supported

## Links

- [Documentation](https://reworkhow.github.io/NNMM.jl/stable)
- [GitHub Repository](https://github.com/reworkhow/NNMM.jl)

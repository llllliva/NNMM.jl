# Changelog

All notable changes to NNMM.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Partial-connected networks have a bug (`wArray2` undefined) - use fully-connected as workaround
- User-defined activation functions not yet supported (use built-in functions)
- Multi-trait phenotypes in Layer 3 not yet supported
- GBLUP not supported for Layer 2→3 equations
- Binary traits not yet supported

## Links

- [Documentation](https://reworkhow.github.io/NNMM.jl/stable)
- [GitHub Repository](https://github.com/reworkhow/NNMM.jl)

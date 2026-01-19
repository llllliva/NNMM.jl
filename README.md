# NNMM.jl

[![CI](https://github.com/reworkhow/NNMM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/reworkhow/NNMM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/reworkhow/NNMM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/reworkhow/NNMM.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://reworkhow.github.io/NNMM.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://reworkhow.github.io/NNMM.jl/dev)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-blue?logo=julia)](https://julialang.org/)

**NNMM.jl** is an open-source Julia package for **Neural Network Mixed Models** that extend traditional linear mixed models to multilayer neural networks for genomic prediction and genome-wide association studies.

* **Documentation**: [https://reworkhow.github.io/NNMM.jl/stable](https://reworkhow.github.io/NNMM.jl/stable)
* **Authors**: [Hao Cheng](https://qtl.rocks), [Tianjing Zhao](https://animalscience.unl.edu/person/tianjing-zhao/)

## Key Features

- **Neural Network Architecture**: Extend linear mixed models to multilayer neural networks
- **Intermediate Omics Integration**: Incorporate gene expression, metabolomics, and other omics data
- **Flexible Missing Data**: Handle any pattern of missing data in intermediate layers
- **Bayesian Framework**: Full Bayesian inference using MCMC and Hamiltonian Monte Carlo
- **Multiple Bayesian Methods**: BayesA, BayesB, BayesC, BayesL, RR-BLUP, GBLUP
- **Multi-threaded**: Parallel computing support for large-scale analyses

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/reworkhow/NNMM.jl")
```

**Requirements**: Julia 1.9 or later

## Quick Start

```julia
using NNMM
using NNMM.Datasets
using DataFrames
using CSV

# Load built-in simulated data (3534 individuals, 1000 SNPs, 10 omics)
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

# Read phenotype file and prepare separate omics/trait files
pheno_df = CSV.read(pheno_path, DataFrame)

# Create omics file (10 omics features)
omics_df = pheno_df[:, vcat(:ID, [Symbol("omic$i") for i in 1:10])]
CSV.write("omics.csv", omics_df; missingstring="NA")

# Create trait file
CSV.write("traits.csv", pheno_df[:, [:ID, :trait1]]; missingstring="NA")

# Define network layers
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path="omics.csv", missing_value="NA"),
    Layer(layer_name="phenotypes", data_path="traits.csv", missing_value="NA")
]

# Define equations between layers
equations = [
    Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = intercept + geno",
        omics_name=["omic$i" for i in 1:10],
        method="BayesC"
    ),
    Equation(
        from_layer_name="omics",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + omics",
        phenotype_name=["trait1"],
        method="BayesC",
        activation_function="linear"
    )
]

# Run NNMM
results = runNNMM(layers, equations; chain_length=5000, burnin=1000)

# Get estimated breeding values
ebv = results["EBV_NonLinear"]
```

## Network Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Genotypes  │ ──► │   Omics     │ ──► │ Phenotypes  │
│  (Layer 1)  │     │  (Layer 2)  │     │  (Layer 3)  │
└─────────────┘     └─────────────┘     └─────────────┘
     SNPs           Gene Expression        Traits
                    Metabolomics
                    (can be missing)
```

NNMM supports:
- **Fully-connected** neural networks
- **Activation functions**: `linear`, `sigmoid`, `tanh`, `relu`, `leakyrelu`
- **Any pattern of missing data** in intermediate layers (sampled via HMC)

## Help

```julia
using NNMM
?NNMM      # Show package info
?runNNMM   # Help on specific function
?Layer     # Help on Layer type
?Equation  # Help on Equation type
```

## Citation

If you use NNMM.jl in your research, please cite:

> Zhao, T., Zeng, J., & Cheng, H. (2022). Extend mixed models to multilayer neural networks for genomic prediction including intermediate omics data. *GENETICS*, iyac034. https://doi.org/10.1093/genetics/iyac034

> Zhao, T., Fernando, R., & Cheng, H. (2021). Interpretable artificial neural networks incorporating Bayesian alphabet models for genome-wide prediction and association studies. *G3 Genes|Genomes|Genetics*, jkab228. https://doi.org/10.1093/g3journal/jkab228

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

MIT License - see [LICENSE](LICENSE) for details.

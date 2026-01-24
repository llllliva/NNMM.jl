
![NNMM](assets/NNMM.png)

# NNMM.jl - Neural Network Mixed Models

NNMM.jl is a Julia package for **Mixed Effect Neural Networks** that extend traditional linear mixed models to multilayer neural networks for genomic prediction and genome-wide association studies.

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.
    For earlier versions using `build_model`/`runMCMC`, see the [Legacy API](#legacy-api) section.

## Key Features

* **Generalizes Traditional Methods**: Traditional BayesC and other Bayesian alphabet methods are special cases of NNMM
* **Neural Network Architecture**: Extend linear mixed models to multilayer neural networks
* **Intermediate Omics Integration**: Incorporate known intermediate omics features (e.g., gene expression) in the middle layer
* **Flexible Missing Data Handling**: Allow any patterns of missing data in intermediate layers
* **Bayesian Framework**: Full Bayesian inference using MCMC and Hamiltonian Monte Carlo
* **Multi-threaded Parallelism**: Efficient parallel computing for large-scale analyses

## Network Architecture

NNMM models relationships between:
- **Input Layer**: Genotypes (SNP markers)
- **Middle Layer**: Intermediate traits (e.g., gene expression, metabolomics)
- **Output Layer**: Phenotypes

The package supports:
- Fully-connected neural networks
- Partial-connected neural networks
- User-defined nonlinear activation functions

## Installation

```julia
using Pkg
Pkg.add("NNMM")
```

Or for the latest development version:
```julia
Pkg.add(url="https://github.com/reworkhow/NNMM.jl")
```

**Requirements**: Julia 1.9 or later.

## Quick Start

This example uses built-in simulated data to demonstrate the NNMM workflow:

```julia
using NNMM
using NNMM.Datasets
using DataFrames, CSV
using Statistics
using Random

# Set seed for reproducibility
Random.seed!(42)

# Load built-in simulated dataset
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

# Read phenotype file and create separate omics/trait files
pheno_df = CSV.read(pheno_path, DataFrame)

# Create omics file (10 omics features)
omics_df = pheno_df[:, vcat(:ID, [Symbol("omic$i") for i in 1:10])]
omics_path = "omics_temp.csv"
CSV.write(omics_path, omics_df; missingstring="NA")

# Create trait file
trait_df = pheno_df[:, [:ID, :trait1]]
trait_path = "trait_temp.csv"
CSV.write(trait_path, trait_df; missingstring="NA")

# Define network layers (3-layer architecture)
layers = [
    # Layer 1: Genotypes (input layer)
    Layer(layer_name="geno", data_path=[geno_path]),
    # Layer 2: Omics (hidden/intermediate layer)  
    Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
    # Layer 3: Phenotypes (output layer)
    Layer(layer_name="phenotypes", data_path=trait_path, missing_value="NA")
]

# Define equations connecting layers
equations = [
    # Equation 1: Genotypes → Omics
    Equation(
        from_layer_name = "geno",
        to_layer_name = "omics",
        equation = "omics = intercept + geno",
        omics_name = ["omic$i" for i in 1:10],
        method = "BayesC",
        estimatePi = true
    ),
    # Equation 2: Omics → Phenotypes  
    Equation(
        from_layer_name = "omics",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + omics",
        phenotype_name = ["trait1"],
        method = "BayesC",
        activation_function = "linear"  # or "sigmoid", "tanh", "relu"
    )
]

# Run NNMM analysis
results = runNNMM(layers, equations;
    chain_length = 5000,        # Total MCMC iterations
    burnin = 1000,              # Burn-in iterations to discard
    output_folder = "nnmm_results"
)

# Access results
ebv = results["EBV_NonLinear"]  # Estimated breeding values
println("Mean EBV: ", mean(Float64.(ebv.EBV)))

# Cleanup temporary files
rm(omics_path, force=true)
rm(trait_path, force=true)
```

## API Summary Table

| Example | Required Packages | Entry Function(s) | What It Demonstrates |
|---------|-------------------|-------------------|---------------------|
| [Quick Start](#quick-start) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM` | Basic 3-layer NNMM with BayesC |
| [Tutorial](manual/tutorial.md) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM`, `GWAS` | Complete workflow with GWAS |
| [Part2: NNMM](nnmm/Part2_NNMM.md) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM` | Fully-connected NNMM |
| [Part3: Omics](nnmm/Part3_NNMMwithIntermediateOmics.md) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM` | NNMM with observed omics |
| [Part4: Partial](nnmm/Part4_PartialConnectedNeuralNetwork.md) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM` | Partial-connected networks |
| [Part5: Custom](nnmm/Part5_UserDefinedNonlinearFunction.md) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM` | User-defined activation functions |
| [Part6: Traditional](nnmm/Part6_TraditionalGenomicPrediction.md) | NNMM, DataFrames, CSV | `Layer`, `Equation`, `runNNMM` | Traditional BayesC as NNMM |

## Supporting and Citing

If you use NNMM.jl for your research, please cite:

> Zhao, T., Zeng, J., & Cheng, H. (2022). Extend mixed models to multilayer neural networks for genomic prediction including intermediate omics data. *GENETICS*. https://doi.org/10.1093/genetics/iyac034

> Zhao, T., Fernando, R., & Cheng, H. (2021). Interpretable artificial neural networks incorporating Bayesian alphabet models for genome-wide prediction and association studies. *G3 Genes|Genomes|Genetics*. https://doi.org/10.1093/g3journal/jkab228

Please star the repository [here](https://github.com/reworkhow/NNMM.jl) to help demonstrate community involvement.

## Getting Help

- [Open an issue](https://github.com/reworkhow/NNMM.jl/issues) on GitHub
- Contact: <qtlcheng@ucdavis.edu>

## Tutorials

### NNMM Tutorials
```@contents
Pages = [
  "nnmm/Part1_introduction.md",
  "nnmm/Part2_NNMM.md",
  "nnmm/Part3_NNMMwithIntermediateOmics.md",
  "nnmm/Part4_PartialConnectedNeuralNetwork.md",
  "nnmm/Part5_UserDefinedNonlinearFunction.md",
  "nnmm/Part6_TraditionalGenomicPrediction.md",
]
Depth = 2
```

### Manual
```@contents
Pages = [
  "manual/getstarted.md",
  "manual/tutorial.md",
  "manual/missing_data.md",
  "manual/skip_connection.md",
  "manual/public.md",
]
Depth = 2
```

## Legacy API

!!! warning "Deprecated API"
    The `build_model` / `get_genotypes` / `runMCMC` API from earlier versions is deprecated.
    Please use the new `Layer` / `Equation` / `runNNMM` API shown above.
    
    Legacy example (do not use for new code):
    ```julia
    # OLD API - DEPRECATED
    genotypes = get_genotypes("genotypes.csv", separator=',', method="BayesC")
    model = build_model("y = intercept + genotypes", num_hidden_nodes=3, nonlinear_function="tanh")
    out = runMCMC(model, phenotypes, chain_length=5000)
    ```

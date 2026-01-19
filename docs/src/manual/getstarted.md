# Get Started

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.

## Installation

To install Julia, please go to the [official Julia website](https://julialang.org/downloads/).
Please see [platform specific instructions](https://julialang.org/downloads/platform.html)
if you have trouble installing Julia.

**Requirements**: Julia 1.9 or later.

To install the NNMM package, use the following command inside the Julia REPL (or IJulia Notebook):
```julia
using Pkg
Pkg.add("NNMM")
```

To load the NNMM package:

```julia
using NNMM
```

### Development Version

To use the latest/beta features under development:
```julia
Pkg.add(url="https://github.com/reworkhow/NNMM.jl")
```

### Jupyter Notebook

If you prefer "reproducible research", an interactive Jupyter Notebook interface is available
for Julia (and therefore NNMM). The Jupyter Notebook is an open-source web application for creating
and sharing documents that contain live code, equations, visualizations and explanatory text.
To install IJulia for Jupyter Notebook, please go to [IJulia](https://github.com/JuliaLang/IJulia.jl).

## Multi-threaded Parallelism

NNMM supports multi-threaded parallelism for faster computation. The number of threads can be checked by:
```julia
Threads.nthreads()
```

To start Julia with multiple threads (requires Julia 1.5+):
```bash
julia --threads 4
```

Or set the environment variable:
```bash
export JULIA_NUM_THREADS=4
julia
```

## Access Documentation

!!! warning
    Please load the NNMM package first.

To show basic information about NNMM in REPL or IJulia notebook, use `?NNMM` and press enter.

For help on a specific function, type `?` followed by its name:
```julia
?runNNMM   # Help on runNNMM function
?Layer     # Help on Layer type
?Equation  # Help on Equation type
```

The full documentation is available [here](http://reworkhow.github.io/NNMM.jl/latest/index.html).

## Quick Example

```julia
using NNMM
using NNMM.Datasets
using DataFrames, CSV
using Random

# Set seed for reproducibility
Random.seed!(42)

# Load built-in example data
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

# Create omics and trait files
omics_df = pheno_df[:, vcat(:ID, [Symbol("omic$i") for i in 1:10])]
CSV.write("omics.csv", omics_df; missingstring="NA")
CSV.write("trait.csv", pheno_df[:, [:ID, :trait1]]; missingstring="NA")

# Define model architecture
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path="omics.csv", missing_value="NA"),
    Layer(layer_name="pheno", data_path="trait.csv", missing_value="NA")
]

equations = [
    Equation(
        from_layer_name="geno", to_layer_name="omics",
        equation="omics = intercept + geno",
        traits=["omic$i" for i in 1:10],
        method="BayesC"
    ),
    Equation(
        from_layer_name="omics", to_layer_name="pheno",
        equation="pheno = intercept + omics",
        traits=["trait1"],
        activation_function="linear"
    )
]

# Run MCMC analysis
results = runNNMM(layers, equations;
    chain_length=5000,
    burnin=1000,
    output_folder="my_results"
)

# Access EBV results
ebv = results["EBV_NonLinear"]
println("Mean EBV: ", mean(ebv.EBV))

# Cleanup
rm("omics.csv", force=true)
rm("trait.csv", force=true)
```

## Run Your Analysis

There are several ways to run your analysis:

### Interactive Session (REPL)

Start an interactive session by double-clicking the Julia executable or running `julia` from the command line:

```julia
julia> using NNMM
julia> # your analysis code here
```

To evaluate code written in a file `script.jl` in REPL:

```julia
julia> include("script.jl")
```

To exit the interactive session, type `^D` (control + d) or `quit()`.

### Command Line

To run code in a file non-interactively from the command line:

```bash
julia script.jl
```

If you want to pass arguments to your script:
```bash
julia script.jl arg1 arg2
```
where arguments `arg1` and `arg2` are passed as `ARGS[1]` and `ARGS[2]` of type *String*.

### Jupyter Notebook

To run code in Jupyter Notebook, please see [IJulia](https://github.com/JuliaLang/IJulia.jl).

## Key Concepts

### Three-Layer Architecture

NNMM uses a three-layer neural network:

1. **Layer 1 (Input)**: Genotypes (SNP markers)
2. **Layer 2 (Hidden)**: Omics/Latent traits (can have missing values)
3. **Layer 3 (Output)**: Phenotypes

### Core Types

| Type | Purpose |
|------|---------|
| `Layer` | Defines a layer with data file, layer name, and options |
| `Equation` | Connects two layers with a model equation and Bayesian method |

### Core Function

| Function | Purpose |
|----------|---------|
| `runNNMM(layers, equations; ...)` | Run the NNMM analysis |

## Next Steps

- [Tutorial](tutorial.md): Complete step-by-step analysis
- [Part 2: Basic NNMM](../nnmm/Part2_NNMM.md): Fully-connected networks
- [Part 3: NNMM with Omics](../nnmm/Part3_NNMMwithIntermediateOmics.md): Using observed omics data
- [API Reference](public.md): Full function documentation

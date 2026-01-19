# Step-by-Step Tutorial

!!! note "Version Compatibility"
    This tutorial uses NNMM.jl v0.3+ with the `Layer`/`Equation`/`runNNMM` API.

This tutorial walks through a complete NNMM analysis using the built-in simulated dataset.

## Overview

We'll perform genomic prediction using a three-layer Neural Network Mixed Model:

```
Genotypes (SNPs) → Omics (gene expression) → Phenotype (trait)
     Layer 1            Layer 2                  Layer 3
```

This architecture allows us to:
- Model the biological pathway from DNA to gene expression to phenotype
- Handle missing omics data via HMC sampling
- Obtain estimated breeding values (EBV) for selection

## Step 1: Load Packages and Data

```julia
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

# Set seed for reproducibility
Random.seed!(42)

# Load built-in simulated dataset paths
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

# Check the data
println("Genotype file: ", geno_path)
println("Phenotype file: ", pheno_path)
```

**Expected output:**
```
Genotype file: /path/to/NNMM.jl/src/datasets/data/simulated_omics_data/genotypes_1000snps.txt
Phenotype file: /path/to/NNMM.jl/src/datasets/data/simulated_omics_data/phenotypes_sim.txt
```

## Step 2: Prepare Data Files

The phenotype file contains both omics and trait data. NNMM requires separate files for omics (Layer 2) and phenotypes (Layer 3):

```julia
# Read the combined phenotype file
pheno_df = CSV.read(pheno_path, DataFrame)
println("Columns: ", names(pheno_df))
# Expected: [:ID, :omic1, ..., :omic10, :trait1, :genetic_total, ...]

# Create omics file (10 omics features)
omics_cols = vcat(:ID, [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
omics_file = "omics_data.csv"
CSV.write(omics_file, omics_df; missingstring="NA")

# Create phenotype file (single trait)
trait_df = pheno_df[:, [:ID, :trait1]]
trait_file = "phenotypes.csv"
CSV.write(trait_file, trait_df; missingstring="NA")

println("Created: $omics_file and $trait_file")
println("Number of individuals: ", nrow(pheno_df))
println("Number of omics features: ", length(omics_cols) - 1)
```

## Step 3: Define the Network Architecture

NNMM uses `Layer` structs to define each layer in the network:

```julia
# Define the three layers
layers = [
    # Layer 1: Genotypes (input layer)
    # Note: data_path is wrapped in [] for the genotype layer
    Layer(
        layer_name = "genotypes",
        data_path = [geno_path]
    ),
    
    # Layer 2: Omics (hidden/intermediate layer)
    # Can contain missing values that will be sampled via HMC
    Layer(
        layer_name = "omics",
        data_path = omics_file,
        missing_value = "NA"
    ),
    
    # Layer 3: Phenotypes (output layer)
    Layer(
        layer_name = "phenotypes",
        data_path = trait_file,
        missing_value = "NA"
    )
]
```

### Layer Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `layer_name` | Identifier used in equations | Required |
| `data_path` | Path to data file(s); use `[path]` for genotypes | Required |
| `separator` | Column delimiter | `','` |
| `header` | File has header row | `true` |
| `quality_control` | Perform MAF filtering (genotypes only) | `true` |
| `MAF` | Minor allele frequency threshold | `0.01` |
| `missing_value` | Value representing missing data | `9.0` |
| `center` | Center the data | `true` |

## Step 4: Define the Equations

`Equation` structs connect layers and specify the statistical method:

```julia
equations = [
    # Equation 1: Layer 1 → Layer 2 (Genotypes predict omics)
    Equation(
        from_layer_name = "genotypes",
        to_layer_name = "omics",
        equation = "omics = intercept + genotypes",
        omics_name = ["omic$i" for i in 1:10],  # Names of omics columns
        method = "BayesC",                       # Bayesian method
        estimatePi = true                        # Estimate marker inclusion probability
    ),
    
    # Equation 2: Layer 2 → Layer 3 (Omics predict phenotype)
    Equation(
        from_layer_name = "omics",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + omics",
        phenotype_name = ["trait1"],             # Names of phenotype columns
        method = "BayesC",
        estimatePi = true,
        activation_function = "linear"           # Activation: "linear", "sigmoid", "tanh", "relu"
    )
]
```

### Equation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `from_layer_name` | Source layer name | Required |
| `to_layer_name` | Target layer name | Required |
| `equation` | Model equation string | Required |
| `omics_name` | Names of omics variables (1→2 equation) | `false` |
| `phenotype_name` | Names of phenotype variables (2→3 equation) | `false` |
| `method` | Bayesian method: "BayesA", "BayesB", "BayesC", "BayesL", "RR-BLUP" | `"BayesC"` |
| `activation_function` | "linear", "sigmoid", "tanh", "relu", "leakyrelu" | `"linear"` |
| `estimatePi` | Estimate marker inclusion probability | `true` |
| `covariate` | Covariate variable names (optional) | `false` |
| `random` | Random effect specifications (optional) | `false` |

## Step 5: Run the Analysis

```julia
# Run NNMM with MCMC sampling
results = runNNMM(
    layers, 
    equations;
    chain_length = 5000,           # Number of MCMC iterations
    burnin = 1000,                 # Burn-in iterations to discard
    printout_frequency = 1000,     # Print progress every N iterations
    output_folder = "nnmm_output"  # Where to save results
)
```

!!! tip "For quick testing"
    Use `chain_length=100` for testing. For production, use at least `chain_length=50000`.

### runNNMM Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chain_length` | Total MCMC iterations | `100` |
| `burnin` | Burn-in iterations | `0` |
| `output_samples_frequency` | Save every Nth sample | auto |
| `outputEBV` | Output estimated breeding values | `true` |
| `output_heritability` | Calculate heritability | `true` |
| `output_folder` | Output directory | `"nnmm_results"` |
| `seed` | Random seed for reproducibility | `false` |
| `double_precision` | Use Float64 (slower, more precise) | `false` |

## Step 6: Examine Results

### Estimated Breeding Values (EBV)

```julia
# Get EBV for all individuals
ebv = results["EBV_NonLinear"]
println("EBV for first 10 individuals:")
display(first(ebv, 10))

# Summary statistics (convert to Float64 for statistics)
ebv_vals = Float64.(ebv.EBV)
println("\nEBV summary:")
println("Mean: ", round(mean(ebv_vals), digits=3))
println("Std:  ", round(std(ebv_vals), digits=3))
println("Min:  ", round(minimum(ebv_vals), digits=3))
println("Max:  ", round(maximum(ebv_vals), digits=3))
```

### Available Result Keys

```julia
# See all available results
println(keys(results))
```

Common result keys:
- `"EBV_NonLinear"`: Estimated breeding values (from predicted omics)
- `"EPV_NonLinear"`: Estimated phenotypic values (from observed omics)
- `"EBV_omic1"`, `"EBV_omic2"`, ...: EBV for each omics trait

### MCMC Output Files

The output folder contains MCMC samples for all parameters:

| File | Description |
|------|-------------|
| `MCMC_samples_marker_effects_*.txt` | Marker effect samples |
| `MCMC_samples_marker_effects_variances_*.txt` | Marker variance samples |
| `MCMC_samples_residual_variance.txt` | Residual variance samples |
| `MCMC_samples_pi_*.txt` | Marker inclusion probability samples |
| `MCMC_samples_EBV_*.txt` | EBV samples per trait/omics |
| `EBV_NonLinear.txt` | Posterior mean EBV |
| `EPV_NonLinear.txt` | Posterior mean EPV |

## Step 7: Post-Analysis with GWAS

Identify important markers using model frequency:

```julia
# GWAS based on model frequency (probability marker is included)
marker_file = "nnmm_output/MCMC_samples_marker_effects_genotypes_omic1.txt"
if isfile(marker_file)
    gwas_result = GWAS(marker_file)
    
    # Top markers for omic1
    sorted_gwas = sort(gwas_result, :modelfrequency, rev=true)
    println("Top 10 markers for omic1:")
    display(first(sorted_gwas, 10))
end
```

## Complete Script

Here's the complete analysis in one runnable script:

```julia
# Complete NNMM Analysis Script
# Requires: NNMM.jl v0.3+

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

# === Configuration ===
Random.seed!(42)
const CHAIN_LENGTH = 5000
const BURNIN = 1000

# === Data Preparation ===
println("Loading data...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

pheno_df = CSV.read(pheno_path, DataFrame)
println("  Individuals: ", nrow(pheno_df))

# Create omics file
omics_cols = vcat(:ID, [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
CSV.write("omics.csv", omics_df; missingstring="NA")

# Create trait file
trait_df = pheno_df[:, [:ID, :trait1]]
CSV.write("traits.csv", trait_df; missingstring="NA")

# === Define Model ===
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path="omics.csv", missing_value="NA"),
    Layer(layer_name="pheno", data_path="traits.csv", missing_value="NA")
]

equations = [
    Equation(
        from_layer_name="geno", 
        to_layer_name="omics",
        equation="omics = intercept + geno",
        traits=["omic$i" for i in 1:10],
        method="BayesC"
    ),
    Equation(
        from_layer_name="omics", 
        to_layer_name="pheno",
        equation="pheno = intercept + omics",
        traits=["trait1"],
        method="BayesC",
        activation_function="linear"
    )
]

# === Run MCMC ===
println("Running MCMC (chain_length=$CHAIN_LENGTH, burnin=$BURNIN)...")
results = runNNMM(layers, equations; 
                  chain_length=CHAIN_LENGTH, 
                  burnin=BURNIN,
                  output_folder="results")

# === Results ===
ebv = results["EBV_NonLinear"]
println("\nAnalysis complete!")
println("EBV saved in results folder.")
println("Mean EBV: ", round(mean(Float64.(ebv.EBV)), digits=4))
println("Std EBV:  ", round(std(Float64.(ebv.EBV)), digits=4))

# === Accuracy (if true breeding values available) ===
if hasproperty(pheno_df, :genetic_total)
    # Convert ID types for joining
    ebv.ID = string.(ebv.ID)
    pheno_df.ID = string.(pheno_df.ID)
    merged = innerjoin(ebv, pheno_df[:, [:ID, :genetic_total]], on=:ID)
    accuracy = cor(Float64.(merged.EBV), merged.genetic_total)
    println("Accuracy (cor with true BV): ", round(accuracy, digits=4))
end

# Cleanup temp files
rm("omics.csv", force=true)
rm("traits.csv", force=true)
```

## Next Steps

- **Different activation functions**: Try `activation_function="sigmoid"` or `"tanh"` for nonlinear relationships
- **Different Bayesian methods**: Try `method="BayesA"`, `"BayesB"`, `"BayesL"`, or `"RR-BLUP"`
- **Add pedigree**: Include random effects with pedigree relationships
- **Multi-omics**: Add more omics features in `omics_name`
- **Partial-connected networks**: See [Part 4](../nnmm/Part4_PartialConnectedNeuralNetwork.md)

## Troubleshooting

### Common Issues

1. **"There is data already stored in layer X"**: Re-run the `layers = [...]` definition to reset.

2. **"omics_name or phenotype_name must be provided"**: Ensure the first equation has `omics_name` and the second has `phenotype_name`.

3. **Folder already exists**: NNMM auto-increments folder names (e.g., `nnmm_results1`, `nnmm_results2`).

4. **Memory issues**: Use `double_precision=false` (default) for large datasets.

See the [NNMM Examples](../nnmm/Part2_NNMM.md) for more advanced use cases.

# Part 4: Partial-Connected Neural Networks

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.

## Model Architecture

In partial-connected neural networks, SNPs can be divided into groups by users, and each group connects to its own intermediate trait in the middle layer. This is useful when:
- Different genomic regions affect different intermediate traits
- You want to model pathway-specific effects
- You have prior biological knowledge about SNP-trait relationships

```
  Genotype Group 1 ─────► Omics/Latent 1 ─┐
  Genotype Group 2 ─────► Omics/Latent 2 ─┼──► Phenotype
  Genotype Group 3 ─────► Omics/Latent 3 ─┘
```

!!! important "Partial Connection Setup"
    To create a partial-connected network, provide **multiple genotype files** in the `data_path` parameter as a vector. The number of genotype files must equal the number of omics features.

!!! important "Data Requirements"
    All genotype files and phenotype files must have matching individual IDs. This example uses 
    the `simulated_omics_data` dataset which has 3534 aligned individuals.

## Example: Partial-Connected Network with Observed Omics

This example demonstrates:
- **Number of genotype groups**: 3 (simulated by splitting SNPs)
- **Number of omics features**: 3 (one per genotype group)
- **Bayesian method**: BayesC
- **Activation function**: `sigmoid`

![](https://github.com/zhaotianjing/figures/blob/main/part4_partial_omics.png?raw=true)

```julia
# Step 1: Load packages
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

Random.seed!(123)

# Step 2: Load simulated data
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

# Step 3: Split genotypes into 3 groups (for demonstration)
# Read full genotype file and split into 3 parts
geno_full = CSV.read(geno_path, DataFrame)
n_markers = ncol(geno_full) - 1  # Exclude ID column
markers_per_group = div(n_markers, 3)

# Create 3 genotype group files
geno1_cols = vcat(:ID, names(geno_full)[2:markers_per_group+1])
geno2_cols = vcat(:ID, names(geno_full)[markers_per_group+2:2*markers_per_group+1])
geno3_cols = vcat(:ID, names(geno_full)[2*markers_per_group+2:end])

geno1_df = geno_full[:, geno1_cols]
geno2_df = geno_full[:, geno2_cols]
geno3_df = geno_full[:, geno3_cols]

genofile1 = "geno_group1.csv"
genofile2 = "geno_group2.csv"
genofile3 = "geno_group3.csv"

CSV.write(genofile1, geno1_df)
CSV.write(genofile2, geno2_df)
CSV.write(genofile3, geno3_df)

# Step 4: Create omics file (3 features, one per genotype group)
omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3]]
omics_file = "omics_partial.csv"
CSV.write(omics_file, omics_df; missingstring="NA")

# Step 5: Create phenotype file
trait_df = pheno_df[:, [:ID, :trait1]]
trait_file = "trait_partial.csv"
CSV.write(trait_file, trait_df; missingstring="NA")

# Step 6: Define layers
# KEY: Provide multiple genotype files as a vector for partial connection
layers = [
    # Layer 1: Multiple genotype files = partial-connected network
    Layer(
        layer_name = "geno",
        data_path = [genofile1, genofile2, genofile3]  # Vector of 3 files
    ),
    # Layer 2: Observed omics (one per genotype group)
    Layer(
        layer_name = "omics",
        data_path = omics_file,
        missing_value = "NA"
    ),
    # Layer 3: Phenotypes
    Layer(
        layer_name = "phenotypes",
        data_path = trait_file,
        missing_value = "NA"
    )
]

# Step 7: Define equations
# The equation uses a single layer name ("geno"), but NNMM internally
# creates separate equations for each genotype group (geno1, geno2, geno3)
equations = [
    Equation(
        from_layer_name = "geno",
        to_layer_name = "omics",
        equation = "omics = intercept + geno",  # Internal: omic1=intercept+geno1; omic2=intercept+geno2; omic3=intercept+geno3
        traits = ["omic1", "omic2", "omic3"],  # Must match number of genotype files
        method = "BayesC",
        estimatePi = true
    ),
    Equation(
        from_layer_name = "omics",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + omics",
        traits = ["trait1"],
        method = "BayesC",
        activation_function = "sigmoid"
    )
]

# Step 8: Run analysis
out = runNNMM(layers, equations;
    chain_length = 5000,
    burnin = 1000,
    output_folder = "nnmm_partial_results"
)

# Step 9: Check accuracy
ebv = out["EBV_NonLinear"]
ebv.ID = string.(ebv.ID)
pheno_df.ID = string.(pheno_df.ID)
results = innerjoin(ebv, pheno_df[:, [:ID, :genetic_total]], on=:ID)
accuracy = cor(Float64.(results.EBV), results.genetic_total)
println("Prediction accuracy: ", round(accuracy, digits=4))

# Cleanup
rm(genofile1, force=true)
rm(genofile2, force=true)
rm(genofile3, force=true)
rm(omics_file, force=true)
rm(trait_file, force=true)
```

## Example: Partial-Connected Network with Latent Traits

When you don't have observed omics data but want to use a partial-connected architecture:

```julia
# Step 1: Load packages
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

Random.seed!(123)

# Step 2: Load simulated data and split genotypes (same as above)
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

geno_full = CSV.read(geno_path, DataFrame)
n_markers = ncol(geno_full) - 1
markers_per_group = div(n_markers, 3)

geno1_cols = vcat(:ID, names(geno_full)[2:markers_per_group+1])
geno2_cols = vcat(:ID, names(geno_full)[markers_per_group+2:2*markers_per_group+1])
geno3_cols = vcat(:ID, names(geno_full)[2*markers_per_group+2:end])

CSV.write("geno_group1.csv", geno_full[:, geno1_cols])
CSV.write("geno_group2.csv", geno_full[:, geno2_cols])
CSV.write("geno_group3.csv", geno_full[:, geno3_cols])

# Step 3: Create latent trait file (all missing)
n_individuals = nrow(pheno_df)
latent_df = DataFrame(
    ID = pheno_df.ID,
    latent1 = fill(missing, n_individuals),
    latent2 = fill(missing, n_individuals),
    latent3 = fill(missing, n_individuals)
)
CSV.write("latent_partial.csv", latent_df; missingstring="NA")

# Step 4: Create phenotype file
CSV.write("trait_partial.csv", pheno_df[:, [:ID, :trait1]]; missingstring="NA")

# Step 5: Define layers
layers = [
    Layer(layer_name="geno", data_path=["geno_group1.csv", "geno_group2.csv", "geno_group3.csv"]),
    Layer(layer_name="latent", data_path="latent_partial.csv", missing_value="NA"),
    Layer(layer_name="phenotypes", data_path="trait_partial.csv", missing_value="NA")
]

# Step 6: Define equations
equations = [
    Equation(from_layer_name="geno", to_layer_name="latent",
             equation="latent = intercept + geno",
             traits=["latent1", "latent2", "latent3"], method="BayesC"),
    Equation(from_layer_name="latent", to_layer_name="phenotypes",
             equation="phenotypes = intercept + latent",
             traits=["trait1"], activation_function="tanh")
]

# Step 7: Run analysis
out = runNNMM(layers, equations; chain_length=5000, burnin=1000, output_folder="nnmm_partial_latent")

# Step 8: Check accuracy
ebv = out["EBV_NonLinear"]
ebv.ID = string.(ebv.ID)
pheno_df.ID = string.(pheno_df.ID)
results = innerjoin(ebv, pheno_df[:, [:ID, :genetic_total]], on=:ID)
println("Accuracy: ", round(cor(Float64.(results.EBV), results.genetic_total), digits=4))

# Cleanup
for f in ["geno_group1.csv", "geno_group2.csv", "geno_group3.csv", "latent_partial.csv", "trait_partial.csv"]
    rm(f, force=true)
end
```

## Key Points

1. **Number of files = Number of omics**: The number of genotype files must exactly match the number of omics/latent features.

2. **Automatic naming**: When you provide 3 genotype files and `layer_name="geno"`, NNMM internally names them `geno1`, `geno2`, `geno3`.

3. **Independent models**: Each genotype group gets its own independent Bayesian model for predicting its corresponding omics feature.

4. **Mixed with residuals**: You can add extra latent nodes for residual effects by including additional (all-missing) columns in the omics file.

## Comparison: Fully-Connected vs Partial-Connected

| Aspect | Fully-Connected | Partial-Connected |
|--------|-----------------|-------------------|
| Genotype input | Single file (vector with 1 element) | Multiple files (vector with N elements) |
| Connectivity | All SNPs → All omics | SNP group i → Omics i |
| Marker effects | Shared across omics | Group-specific |
| Use case | General prediction | Pathway-specific modeling |

## Output Files

Output files are similar to fully-connected networks, but with group-specific marker effect files:
- `MCMC_samples_marker_effects_geno1_omic1.txt`
- `MCMC_samples_marker_effects_geno2_omic2.txt`
- `MCMC_samples_marker_effects_geno3_omic3.txt`

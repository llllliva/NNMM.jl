# Part 2: Mixed Effect Neural Network (NNMM)

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.

## Model Architecture

NNMM extends traditional genomic prediction by adding an intermediate layer:

```
Genotypes → Unobserved Intermediate Traits → Phenotypes
 (Layer 1)         (Layer 2)                 (Layer 3)
```

When the intermediate traits are unobserved (latent), they are sampled via Hamiltonian Monte Carlo (HMC).

!!! tip "Data Preparation"
    It is recommended to center the phenotypes to have zero mean before running NNMM.

!!! important "Data Requirements"
    Genotype and phenotype files must have matching individual IDs. This example uses the 
    `simulated_omics_data` dataset which has 3534 aligned individuals.

## Example: Fully-Connected Neural Network with Unobserved Latent Traits

This example demonstrates:
- **Activation function**: `tanh` (other options: `"sigmoid"`, `"relu"`, `"leakyrelu"`, `"linear"`)
- **Number of latent nodes**: 3 (unobserved intermediate traits)
- **Bayesian model**: BayesC (multiple independent single-trait models for marker effects)
- **Latent trait sampling**: Hamiltonian Monte Carlo

![](https://github.com/zhaotianjing/figures/blob/main/part2_example.png?raw=true)

```julia
# Step 1: Load packages
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

Random.seed!(123)

# Step 2: Read data (using simulated_omics_data with aligned individuals)
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

# Read phenotypes (output layer)
phenotypes = CSV.read(pheno_path, DataFrame)

# Step 3: Create latent trait file (all values missing)
# When no omics data is observed, create a file with all missing values
# NNMM will sample these latent traits via HMC
n_individuals = nrow(phenotypes)
n_latent = 3  # Number of latent nodes in middle layer

latent_df = DataFrame(ID = phenotypes.ID)
for i in 1:n_latent
    latent_df[!, Symbol("latent$i")] = fill(missing, n_individuals)
end

# Write to file with "NA" as missing string
latent_file = "latent_traits.csv"
CSV.write(latent_file, latent_df; missingstring="NA")

# Step 4: Create phenotype file for Layer 3
# Center phenotype for better convergence
pheno_mean = mean(skipmissing(phenotypes.trait1))
phenotypes.trait1_centered = phenotypes.trait1 .- pheno_mean

trait_file = "phenotypes_layer3.csv"
trait_df = phenotypes[:, [:ID, :trait1_centered]]
rename!(trait_df, :trait1_centered => :y1)
CSV.write(trait_file, trait_df; missingstring="NA")

# Step 5: Define layers
layers = [
    # Layer 1: Genotypes (input layer)
    Layer(
        layer_name = "genotypes",
        data_path = [geno_path]
    ),
    # Layer 2: Latent traits (all missing, will be sampled via HMC)
    Layer(
        layer_name = "latent",
        data_path = latent_file,
        missing_value = "NA"
    ),
    # Layer 3: Phenotypes (output layer)
    Layer(
        layer_name = "phenotypes",
        data_path = trait_file,
        missing_value = "NA"
    )
]

# Step 6: Define equations
equations = [
    # Genotypes → Latent traits (BayesC)
    Equation(
        from_layer_name = "genotypes",
        to_layer_name = "latent",
        equation = "latent = intercept + genotypes",
        omics_name = ["latent1", "latent2", "latent3"],  # Names match CSV columns
        method = "BayesC",
        estimatePi = true
    ),
    # Latent traits → Phenotypes (with tanh activation)
    Equation(
        from_layer_name = "latent",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + latent",
        phenotype_name = ["y1"],
        method = "BayesC",
        activation_function = "tanh"  # Nonlinear relationship
    )
]

# Step 7: Run analysis
out = runNNMM(layers, equations;
    chain_length = 5000,
    burnin = 1000,
    output_folder = "nnmm_latent_results"
)

# Step 8: Check accuracy (simulated data has true breeding values)
ebv = out["EBV_NonLinear"]

# Convert to proper types and merge
ebv.ID = string.(ebv.ID)
phenotypes.ID = string.(phenotypes.ID)
results = innerjoin(ebv, phenotypes[:, [:ID, :genetic_total]], on=:ID)
accuracy = cor(Float64.(results.EBV), results.genetic_total)
println("Prediction accuracy: ", round(accuracy, digits=4))

# Cleanup temporary files
rm(latent_file, force=true)
rm(trait_file, force=true)
```

## Output Files

When latent traits are named "latent1", "latent2", "latent3", the output files use these names directly.

### Estimate Files (Posterior Means)

| File Name | Description |
|-----------|-------------|
| `EBV_NonLinear.txt` | Estimated breeding values for observed phenotype |
| `EBV_latent1.txt` | EBV for latent node 1 |
| `EBV_latent2.txt` | EBV for latent node 2 |
| `EBV_latent3.txt` | EBV for latent node 3 |
| `genetic_variance.txt` | Genetic variance-covariance of all latent nodes |
| `heritability.txt` | Heritability estimates for all latent nodes |
| `location_parameters.txt` | Intercept estimates for all latent nodes |
| `neural_networks_bias_and_weights.txt` | Bias and weights between latent nodes and phenotypes |
| `pi_genotypes.txt` | Marker inclusion probability (π) for all latent nodes |
| `marker_effects_genotypes.txt` | Marker effects for all latent nodes |
| `residual_variance.txt` | Residual variance-covariance for all latent nodes |

### MCMC Sample Files

| File Name | Description |
|-----------|-------------|
| `MCMC_samples_EBV_NonLinear.txt` | MCMC samples for phenotype breeding values |
| `MCMC_samples_EBV_latent1.txt` | MCMC samples for latent node 1 EBV |
| `MCMC_samples_EBV_latent2.txt` | MCMC samples for latent node 2 EBV |
| `MCMC_samples_EBV_latent3.txt` | MCMC samples for latent node 3 EBV |
| `MCMC_samples_genetic_variance.txt` | MCMC samples for genetic variance-covariance |
| `MCMC_samples_heritability.txt` | MCMC samples for heritability |
| `MCMC_samples_marker_effects_genotypes_latent1.txt` | Marker effect samples for latent node 1 |
| `MCMC_samples_marker_effects_genotypes_latent2.txt` | Marker effect samples for latent node 2 |
| `MCMC_samples_marker_effects_genotypes_latent3.txt` | Marker effect samples for latent node 3 |
| `MCMC_samples_marker_effects_variances_genotypes.txt` | Marker effect variance samples |
| `MCMC_samples_neural_networks_bias_and_weights.txt` | Neural network parameter samples |
| `MCMC_samples_pi_genotypes.txt` | π samples for all latent nodes |
| `MCMC_samples_residual_variance.txt` | Residual variance samples |

## Different Number of Latent Nodes

The number of latent nodes is determined by the `omics_name` parameter in the first equation:

```julia
# For 5 latent nodes:
omics_name = ["latent1", "latent2", "latent3", "latent4", "latent5"]

# For 10 latent nodes:
omics_name = ["latent$i" for i in 1:10]
```

## Different Activation Functions

```julia
# Linear (traditional regression)
Equation(..., activation_function = "linear")

# Sigmoid (outputs between 0 and 1)
Equation(..., activation_function = "sigmoid")

# Tanh (outputs between -1 and 1)
Equation(..., activation_function = "tanh")

# ReLU (sparse activation)
Equation(..., activation_function = "relu")

# Leaky ReLU (sparse with gradient flow)
Equation(..., activation_function = "leakyrelu")
```

## Tips for Best Results

1. **Center phenotypes**: Subtract the mean from phenotypes before running NNMM.

2. **Choose number of latent nodes**: More nodes capture more variance but increase computational cost. Start with 3-5 nodes.

3. **Chain length**: For production, use `chain_length=50000` or more. For testing, `chain_length=1000` is sufficient.

4. **Convergence**: Check MCMC samples for convergence. If variance estimates are unstable, increase chain length.

5. **Activation function**: Use `"linear"` for simple additive models, `"tanh"` or `"sigmoid"` for nonlinear relationships.

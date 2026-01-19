# Part 5: User-Defined Nonlinear Functions

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.

!!! warning "Known Limitation"
    **User-defined activation functions are not yet fully implemented.** The `Equation` struct now 
    accepts `Union{String, Function}`, but the MCMC sampling logic for custom functions requires 
    additional development. Currently, passing a Function will result in a clear error message.
    
    Use the built-in activation functions (`"linear"`, `"sigmoid"`, `"tanh"`, `"relu"`, `"leakyrelu"`) 
    for now. This feature is planned for a future release.

## Overview

The intention is to allow custom nonlinear relationships between the middle layer (intermediate traits) and the output layer (phenotype), instead of using built-in activation functions.

This would be useful when:
- You have domain knowledge about the biological relationship
- The phenotype follows a specific mechanistic model
- Standard activation functions don't capture the true relationship

## Current Status

The `Equation` struct now accepts:
```julia
activation_function::Union{String, Function}  # Accepts strings or custom functions
```

However, passing a custom function will currently raise an error:
```
ERROR: User-defined activation functions are not yet supported. 
Please use a built-in activation function: "tanh", "sigmoid", "relu", "leakyrelu", or "linear".
```

The type change prepares for future implementation of this feature.

## Workaround: Use Built-in Activation Functions

Until this feature is implemented, use the closest built-in activation function:

```julia
# For bounded outputs (0 to 1)
Equation(..., activation_function = "sigmoid")

# For bounded outputs (-1 to 1)
Equation(..., activation_function = "tanh")

# For unbounded linear relationships
Equation(..., activation_function = "linear")

# For sparse/rectified outputs
Equation(..., activation_function = "relu")
Equation(..., activation_function = "leakyrelu")
```

## Example: Using tanh as Alternative to Custom Function

If you need a custom function like:
```math
y = \sqrt{\frac{x_1^2}{x_1^2 + x_2^2}}
```

Consider using `tanh` which provides similar bounded nonlinear behavior:

```julia
# Step 1: Load packages
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

Random.seed!(123)

# Step 2: Load data
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

# Step 3: Create latent trait file (2 latent nodes)
n_individuals = nrow(pheno_df)
latent_df = DataFrame(
    ID = pheno_df.ID,
    latent1 = fill(missing, n_individuals),
    latent2 = fill(missing, n_individuals)
)
latent_file = "latent_custom.csv"
CSV.write(latent_file, latent_df; missingstring="NA")

# Step 4: Create phenotype file
trait_file = "trait_custom.csv"
CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")

# Step 5: Define layers
layers = [
    Layer(layer_name = "genotypes", data_path = [geno_path]),
    Layer(layer_name = "latent", data_path = latent_file, missing_value = "NA"),
    Layer(layer_name = "phenotypes", data_path = trait_file, missing_value = "NA")
]

# Step 6: Define equations with built-in tanh (workaround for custom function)
equations = [
    Equation(
        from_layer_name = "genotypes",
        to_layer_name = "latent",
        equation = "latent = intercept + genotypes",
        traits = ["latent1", "latent2"],
        method = "BayesC",
        estimatePi = true
    ),
    Equation(
        from_layer_name = "latent",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + latent",
        traits = ["trait1"],
        method = "BayesC",
        activation_function = "tanh"  # Use built-in function instead
    )
]

# Step 7: Run analysis
out = runNNMM(layers, equations;
    chain_length = 5000,
    burnin = 1000,
    output_folder = "nnmm_tanh_results"
)

# Step 8: Check results
ebv = out["EBV_NonLinear"]
ebv.ID = string.(ebv.ID)
pheno_df.ID = string.(pheno_df.ID)
results = innerjoin(ebv, pheno_df[:, [:ID, :genetic_total]], on=:ID)
accuracy = cor(Float64.(results.EBV), results.genetic_total)
println("Prediction accuracy: ", round(accuracy, digits=4))

# Cleanup
rm(latent_file, force=true)
rm(trait_file, force=true)
```

## Comparison: Built-in Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| `"linear"` | f(x) = x | (-∞, ∞) | Traditional regression, additive models |
| `"sigmoid"` | f(x) = 1/(1+e^(-x)) | (0, 1) | Bounded outputs, probabilities |
| `"tanh"` | f(x) = tanh(x) | (-1, 1) | Centered bounded outputs |
| `"relu"` | f(x) = max(0, x) | [0, ∞) | Sparse activation, positive outputs |
| `"leakyrelu"` | f(x) = max(0.01x, x) | (-∞, ∞) | Sparse with gradient flow |

## Future: Planned User-Defined Function Support

When implemented, the syntax would be:

```julia
# NOT YET SUPPORTED - Future syntax
custom_fn(x1, x2) = sqrt(x1^2 / (x1^2 + x2^2 + 1e-8))

equations = [
    ...,
    Equation(
        ...,
        activation_function = custom_fn  # Will be supported in future
    )
]
```

### Expected Behavior (Future)

When using a user-defined function:
- **Missing latent traits**: Would be sampled via **Metropolis-Hastings** (not HMC)
- **Reason**: Automatic differentiation for HMC may not work with arbitrary user functions

## Tips

1. **Choose closest built-in**: Select the built-in function that best approximates your desired behavior.

2. **Transform data**: Pre-transform your phenotype data to better match available activation functions.

3. **Check documentation**: Monitor NNMM.jl releases for user-defined function support.

4. **Feature request**: If you need custom functions urgently, open an issue on the GitHub repository.

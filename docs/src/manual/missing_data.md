# Missing Data Handling

This page documents how NNMM.jl handles missing data across different layers and scenarios.

## Overview

NNMM.jl uses a three-layer neural network architecture:

```
Layer 1 (Genotypes X) → Layer 2 (Omics/Latent L) → Layer 3 (Phenotype y)
        ↓                      ↓                         ↓
   Always observed      Can be observed/missing      Can be observed/missing
```

The package handles various combinations of missing data in the middle layer (omics/latent) and output layer (phenotype) for both training and test individuals.

## Summary Table

| Scenario | Layer 2 Status | Phenotype Status | L1→L2 Contribution | L2→L3 Contribution | L2 Sampling |
|----------|---------------|------------------|-------------------|-------------------|-------------|
| **Training** | All observed | Observed | ✅ Full | ✅ Full | Not needed |
| **Training** | Partial missing | Observed | ✅ Full | ✅ Full | HMC for missing |
| **Training** | All missing | Observed | ✅ Full | ✅ Full | HMC for all |
| **Test** | All observed | Missing | ✅ Full | ❌ Excluded | Not needed |
| **Test** | Partial missing | Missing | ✅ Full | ❌ Excluded | Prior for missing |
| **Test** | All missing | Missing | ❌ Excluded | ❌ Excluded | Prior for all |

## Detailed Breakdown

### Layer 1→2 (Genotypes → Omics/Latent)

This layer estimates marker effects that predict omics or latent traits from genotypes.

| Individual Type | Contribution to Marker Effects | Contribution to Variance |
|-----------------|-------------------------------|-------------------------|
| **Training with observed L2** | ✅ Yes (observed omics as target) | ✅ Yes |
| **Training with partial L2** | ✅ Yes (observed + HMC-sampled as targets) | ✅ Yes |
| **Training with all L2 missing** | ✅ Yes (HMC-sampled latent as target) | ✅ Yes |
| **Test with observed L2** | ✅ Yes (observed omics as target) | ✅ Yes |
| **Test with partial L2** | ✅ Yes (observed + prior-sampled as targets) | ✅ Yes |
| **Test with all L2 missing** | ❌ No (excluded via `invweights=0`) | ❌ No |

**Key Points:**
- Training individuals always contribute to Layer 1→2, regardless of phenotype status
- Test individuals with observed or partial omics still contribute to marker effect estimation
- For test individuals with partial L2: **both** observed values and prior-sampled missing values are used as targets
- Test individuals with completely missing Layer 2 are excluded from parameter updates (no observed "anchor")

### Layer 2→3 (Omics/Latent → Phenotype)

This layer estimates neural network weights connecting the middle layer to phenotypes.

| Individual Type | Contribution to NN Weights | Contribution to Variance |
|-----------------|---------------------------|-------------------------|
| **Training (any L2 status)** | ✅ Yes | ✅ Yes |
| **Test (any L2 status)** | ❌ No (excluded via `invweights=0`) | ❌ No |

**Key Points:**
- Only individuals with observed phenotypes contribute to Layer 2→3 parameter estimation
- Test individuals (missing phenotype) are excluded regardless of their omics status
- Variance estimation uses effective sample size (individuals with `invweights > 0`)

### Layer 2 Value Handling

| Scenario | How L2 Values Are Obtained |
|----------|---------------------------|
| **Observed omics** | Direct from data |
| **Missing omics (training)** | HMC sampling using phenotype likelihood |
| **Missing latent (training)** | HMC sampling using phenotype likelihood |
| **Missing omics (test, partial)** | Prior sampling for missing parts |
| **Missing latent (test, all missing)** | Prior sampling for all |

**Hamiltonian Monte Carlo (HMC) Sampling:**

For training individuals with observed phenotypes, missing omics/latent values are sampled using HMC with the phenotype likelihood:

```
log_likelihood = -0.5 * (y - f(L))² / σ²_residual
```

This allows the phenotype information to guide the latent trait sampling.

**Prior Sampling:**

For test individuals without phenotypes, missing values are sampled from the prior (genotype prediction only):

```
L_sampled = X * marker_effects + randn() * sqrt(residual_var)
```

## Detailed: L2 Sampling for Test Individuals

This section explains exactly how L2 values are sampled and used for test individuals.

### Test with Partial L2 Missing

**Step 1: Sample missing L2 values from prior**
```julia
# For missing omics in test individuals (no phenotype available)
L_missing_sampled = X * β + ε    # where ε ~ N(0, σ²_residual)
```

**Step 2: Combine observed and sampled values**
```julia
# Final L2 for this individual:
L2 = [observed_omics..., sampled_omics...]
```

**Step 3: Use ALL L2 values for marker effect estimation**
```julia
# Both observed and sampled values contribute to β update
# invweights1 = 1 (not excluded)
β_new = update_marker_effects(X, L2)  # uses full L2
```

**Example:**
```
Test Individual with Partial L2:
├── Genotypes X = [1, 0, 2, 1]
├── Omics (before sampling):
│   ├── omics1 = 0.5 (observed)
│   ├── omics2 = ?   (missing)
│   └── omics3 = 0.4 (observed)
│
├── Step 1: Sample omics2 from prior
│   └── omics2_sampled = X × β₂ + ε = 0.3
│
├── Step 2: Final L2 = [0.5, 0.3, 0.4]
│
└── Step 3: ALL values used for marker effect update
    ├── β₁ updated using (X, omics1=0.5)  ✅
    ├── β₂ updated using (X, omics2=0.3)  ✅ (sampled value!)
    └── β₃ updated using (X, omics3=0.4)  ✅
```

### Test with All L2 Missing

**Step 1: Sample ALL L2 values from prior**
```julia
# All omics/latent values sampled from prior
L_sampled = X * β + ε    # for all L2 nodes
```

**Step 2: Do NOT contribute to marker effect estimation**
```julia
# invweights1 = 0 (excluded from parameter updates)
# These sampled values are NOT used to update β
```

**Step 3: Use sampled values for prediction only**
```julia
# EPV = f(L_sampled) × weights_NN
```

**Example:**
```
Test Individual with All L2 Missing:
├── Genotypes X = [0, 1, 1, 2]
├── Latent (before sampling):
│   ├── latent1 = ? (missing)
│   ├── latent2 = ? (missing)
│   └── latent3 = ? (missing)
│
├── Step 1: Sample ALL from prior
│   ├── latent1 = X × β₁ + ε₁ = 0.2
│   ├── latent2 = X × β₂ + ε₂ = -0.1
│   └── latent3 = X × β₃ + ε₃ = 0.4
│
├── Step 2: invweights1 = 0 (EXCLUDED from β update)
│   └── Sampled values do NOT update marker effects
│
└── Step 3: Use for prediction only
    └── EPV = f([0.2, -0.1, 0.4]) × weights_NN
```

### Why the Difference?

| Scenario | Why this behavior? |
|----------|-------------------|
| **Partial L2 missing** | Observed values provide "anchors" that constrain the estimation. Sampled values fill gaps but the individual still provides useful information. |
| **All L2 missing** | No observed anchors. Using purely prior-sampled values for β update would be circular (sample from β, then update β) and only add noise without information. |

### MCMC Iteration Flow

```
Each MCMC iteration:
═══════════════════════════════════════════════════════════════

1. SAMPLE MISSING L2 VALUES (for all individuals with missing L2):
   ┌─────────────────────────────────────────────────────────────┐
   │ Training (has phenotype):                                   │
   │   → HMC sampling using phenotype likelihood                 │
   │                                                             │
   │ Test (no phenotype):                                        │
   │   → Prior sampling: L = X × β + noise                       │
   └─────────────────────────────────────────────────────────────┘

2. RESTORE OBSERVED VALUES (for partial missing):
   ┌─────────────────────────────────────────────────────────────┐
   │ Replace sampled values with observed values where available │
   │ (ensures observed data is preserved)                        │
   └─────────────────────────────────────────────────────────────┘

3. UPDATE MARKER EFFECTS (β):
   ┌─────────────────────────────────────────────────────────────┐
   │ Include: Training + Test with observed/partial L2           │
   │          (invweights1 > 0)                                  │
   │                                                             │
   │ Exclude: Test with ALL L2 missing                           │
   │          (invweights1 = 0)                                  │
   └─────────────────────────────────────────────────────────────┘

4. UPDATE VARIANCE using n_effective (individuals with invweights > 0)

5. COMPUTE PREDICTIONS (EPV) for all individuals
```

## Prediction for Test Individuals

| Test Scenario | EBV (L1→L2) | EPV (L2→L3) |
|---------------|-------------|-------------|
| **All L2 observed** | From marker effects | `f(observed_L2) * weights_NN` |
| **Partial L2 observed** | From marker effects | `f(mixed_L2) * weights_NN` |
| **All L2 missing** | From marker effects | `f(prior_sampled_L2) * weights_NN` |

Where:
- `f()` is the activation function (e.g., `tanh`, `sigmoid`, `linear`)
- `mixed_L2` = combination of observed values and prior-sampled values for missing parts
- `prior_sampled_L2` = `X * marker_effects + noise` (sampled from prior)

## Visual Summary

**Training Individual (has phenotype y):**
```
┌─────────────────────────────────────────────────────────────┐
│  Genotypes X  ──► Omics L (observed/HMC) ──► Phenotype y   │
│       │                    │                      │         │
│  contributes to       contributes to        contributes to  │
│  marker effects       NN weights            variance        │
│  (invweights=1)       (invweights=1)        (n_effective)   │
└─────────────────────────────────────────────────────────────┘
```

**Test Individual (missing phenotype y):**
```
┌─────────────────────────────────────────────────────────────┐
│  Genotypes X  ──► Omics L (if observed) ──► Phenotype ?    │
│       │                    │                      │         │
│  contributes to       DOES NOT             DOES NOT         │
│  marker effects*      contribute           contribute       │
│  (invweights=1/0)     (invweights=0)       (excluded)       │
└─────────────────────────────────────────────────────────────┘
* Only if L2 has observed data; otherwise invweights=0
```

## Implementation Details

### Inverse Weights (`invweights`)

NNMM.jl uses inverse weights to control which individuals contribute to parameter estimation:

- `invweights = 1.0`: Full contribution to parameter updates
- `invweights = 0.0`: Excluded from parameter updates (prediction-only)

For Layer 1→2:
```julia
# Individuals with missing phenotype AND no observed omics are excluded
invweights1[prediction_only] .= 0
```

For Layer 2→3:
```julia
# Individuals with missing phenotype are excluded
invweights2[missing_pheno] .= 0
```

### Effective Sample Size for Variance

Variance estimation uses the effective sample size (number of individuals with `invweights > 0`):

```julia
n_effective = count(invweights .> 0)
sample_variance(residuals, n_effective, df, scale, invweights)
```

This ensures unbiased variance estimates when some individuals are excluded from parameter updates.

## Example: Train/Test Split

```julia
using NNMM, NNMM.Datasets, DataFrames, CSV, Random

Random.seed!(12345)

# Load data
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

n = nrow(pheno_df)
n_train = Int(floor(n * 0.8))

# Create train/test split (set test phenotypes to missing)
pheno_df.ID = string.(pheno_df.ID)
pheno_train = DataFrame(
    ID = pheno_df.ID,
    trait1 = Vector{Union{Missing, Float64}}(pheno_df.trait1)
)
test_idx = (n_train+1):n
pheno_train[test_idx, :trait1] .= missing  # Test individuals have missing phenotype

# Save modified phenotype file
tmpdir = mktempdir()
CSV.write(joinpath(tmpdir, "pheno.csv"), pheno_train; missingstring="NA")

# Create completely missing latent layer
latent_df = DataFrame(ID = pheno_df.ID, latent1 = fill(missing, n))
CSV.write(joinpath(tmpdir, "latent.csv"), latent_df; missingstring="NA")

# Define layers and equations
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="latent", data_path=joinpath(tmpdir, "latent.csv"), missing_value="NA"),
    Layer(layer_name="pheno", data_path=joinpath(tmpdir, "pheno.csv"), missing_value="NA")
]

equations = [
    Equation(from_layer_name="geno", to_layer_name="latent",
             equation="latent = intercept + geno",
             traits=["latent1"], method="BayesC"),
    Equation(from_layer_name="latent", to_layer_name="pheno",
             equation="pheno = intercept + latent",
             traits=["trait1"], activation_function="tanh")
]

# Run NNMM - test individuals will be automatically handled
out = runNNMM(layers, equations;
              chain_length=5000, burnin=1000,
              output_folder=joinpath(tmpdir, "results"))

# Get predictions for all individuals (including test)
epv = out["EBV_NonLinear"]
```

The model will automatically:
1. Use training individuals for parameter estimation
2. Exclude test individuals from variance estimation
3. Generate predictions (EPV) for all individuals

## See Also

- [Tutorial](@ref) - Step-by-step guide to using NNMM.jl
- [Part 2: NNMM with Missing Latent Traits](@ref) - Example with completely missing middle layer
- [Part 3: NNMM with Intermediate Omics](@ref) - Example with observed omics

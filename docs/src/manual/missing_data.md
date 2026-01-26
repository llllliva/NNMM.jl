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

## NNMM vs “Classic” Bayesian Neural Networks (BNN)

If you come from a Bayesian neural network (BNN) perspective, NNMM looks unusual because it explicitly models and samples the *middle-layer values* (omics/latent traits) instead of treating hidden activations as purely deterministic.

### “Classic” BNN: hidden activations are deterministic

In a conventional BNN, the *weights* are random, but (given weights and inputs) the hidden activations are deterministic:

```
h = g(W₁*x + b₁)
y = W₂*h + b₂ + e
```

You do **not** separately sample `h`; MCMC/HMC targets the posterior over weights `W` (and other parameters), and `h` is just a computed value given `W` and `x`.

If you have observed omics, a “classic” approach is to treat omics as additional inputs (e.g., concatenate `[X, O]`), not as a stochastic intermediate layer generated from `X`.

### NNMM: the middle layer is a stochastic “intermediate trait” layer

NNMM is closer to a hierarchical model used in quantitative genetics: each middle-layer node (omic/latent trait) is treated like an intermediate phenotype with its own residual noise:

```
h = μ_h + X*α + e_h
y = μ_y + f(h)*w + e_y
```

Here we use the same symbol `h` for the “middle layer” in both BNN and NNMM. In NNMM, `h` represents the middle-layer *intermediate traits* (observed omics and/or latent nodes). Some entries of `h` may be observed (omics), and the rest can be missing/latent.

That extra term `e_h` is the key difference versus deterministic hidden activations. It lets NNMM represent the fact that omics are typically **not** perfectly determined by genotype (environment, measurement noise, batch effects, etc.).

You can interpret the “strength” of the genotype→omics link using a heritability-like ratio for a given omic node:

```
h²_omic = Var(X*α) / Var(h) = σ²_g / (σ²_g + σ²_e)
```

- If `h²_omic ≈ 1`: the omic is almost deterministic from genotype (`σ²_e ≈ 0`), so `h ≈ X*α`.
- If `h²_omic ≈ 0`: genotype carries almost no signal for that omic (`σ²_g ≈ 0`), so the omic is mostly residual noise (`h ≈ e_h`).

### Why this is necessary when you put observed omics in the middle layer

NNMM’s stochastic middle layer is what makes it natural to handle the key “omics in the middle” use cases:

- **Observed + missing omics in the same layer**: treat observed entries as fixed anchors and sample only the missing entries (conditioning on the observed ones).
- **Prediction when phenotypes are missing** (test IDs): missing middle-layer values are drawn from the genotype-based prior `X*α + noise` (no phenotype conditioning).
- **Realistic genotype→omics uncertainty**: without `e_h`, you are implicitly assuming `h²_omic = 1` (omics are fully genetic), which is often unrealistic and can distort downstream phenotype prediction.

**Concrete 1-omic example (why `h²_omic = 0` matters):**

```
h_i = μ_h + x_i*α + e_{h,i}
y_i = μ_y + b*h_i + e_{y,i}
```

If `h²_omic = 0` (equivalently `α ≈ 0`), then this middle-layer omic `h` is unrelated to genotype. In that case:
- Genotypes cannot predict `h` (so they cannot impute it accurately when it’s missing).
- But if `h` is observed at prediction time, it can still predict `y` through `b*h_i`.

This is exactly the situation NNMM is designed for: omics can be *observed intermediates* that help predict phenotypes, while still being only partially (or not at all) explained by genotypes.

### BNN-style vs NNMM-style: when to use which

Both approaches are “Bayesian neural” in the sense that parameters are random and inferred from `y`. The difference is whether the middle layer `h` is treated as a deterministic activation (BNN style) or a stochastic intermediate trait (NNMM style).

**BNN style (deterministic `h`)**
- **What is sampled:** weights/parameters (e.g., `W₁, W₂, …`), not `h_i` values directly.
- **Best when:** the middle layer is purely latent (all missing) and represents a flexible mapping `X → y` (possibly nonlinear).
- **Pros:** no per-individual latent sampling; avoids “fitted-value” behavior from drawing `h_i | y_i`.
- **Cons:** partially observed/missing omics inside `h` are not handled natively; you typically need imputation or explicit missingness-aware modeling.

**NNMM style (stochastic `h`)**
- **What is sampled:** model parameters **and** missing/latent entries of `h` (conditioning on observed omics, and on `y` for training IDs).
- **Best when:** `h` contains observed intermediate traits (omics) that can be partially missing, and you want uncertainty-aware imputation within MCMC.
- **Pros:** principled missing-data handling for omics in the middle layer; interpretable genotype→omics uncertainty.
- **Cons:** if a node in `h` is **all missing**, the genotype→that-node mapping is weakly/non-identifiable unless you add additional structure; training `h_i | y_i` updates can behave like an in-sample fitted component.

**Hybrid (common in practice)**
- Use **BNN style** for "genetic hidden units" that are fully latent (deterministic `h_g = g(X*W + b)` learned from `y`).
- Use **NNMM style** for omics entries in `h` that can be observed/missing (sample missing parts conditional on observed ones).

## Why "Deterministic Latent" Doesn't Work in NNMM

A natural question arises: can we make latent traits **deterministic** (like BNN-style hidden units) to avoid the "EPV inflation" behavior observed on training data?

### The EPV vs EBV Discrepancy

In NNMM, you may notice:
- **Training data**: EPV (Estimated Phenotypic Value) has much higher correlation with observed phenotype than EBV (Estimated Breeding Value)
- **Test data**: EPV ≈ EBV (both show similar correlation with true phenotype)

This is **expected behavior**, not a bug:

| Metric | Training | Test |
|--------|----------|------|
| EPV | High accuracy (inflated) | ≈ EBV (true predictive) |
| EBV | True genetic merit | True genetic merit |

The EPV inflation on training occurs because latent values are sampled **conditioned on the observed phenotype** via HMC:

```
Training: L_i sampled from posterior p(L | X, y, β, w)
          → L incorporates information from y_i
          → EPV_i = f(L_i) * w is "fitted" to y_i
          
Test:     L_i sampled from prior p(L | X, β) only
          → L has no y information (y is missing)
          → EPV_i = f(L_i) * w ≈ EBV_i
```

### The "Deterministic Latent" Idea

To eliminate this EPV inflation, one might try making latent traits **deterministic**:

```
L = X * β    (no residual noise, no sampling)
```

This would mean:
- L is a **direct function** of genotypes
- No HMC sampling needed for latent traits
- EPV should equal EBV (both purely genotype-based)

### Why It Doesn't Work

**The fundamental problem**: when L is deterministic, the marker effects β **cannot be estimated**.

Consider the NNMM model structure:

```
Layer 1→2:  L = X*β           (latent = genotypes × marker effects)
Layer 2→3:  y = intercept + w*L + e_y
```

In the current MCMC algorithm:

1. **Layer 1→2 update**: Estimate β by fitting L = X*β
   - Target: L (latent trait values)
   - Prediction: X*β
   - **Residual: L - X*β**

2. **Layer 2→3 update**: Estimate w by fitting y = w*L + e

**The critical issue**: With deterministic L = X*β:
```
Residual = L - X*β = X*β - X*β = 0
```

**Zero residual = no information to update β!**

The marker effects β need a "target" to fit against. When L is deterministic, that target **equals the prediction by definition**, so there's nothing to learn.

### Why Sampling Works

With HMC sampling, L is drawn from a **posterior** that considers **both** genotypes and phenotype:

```
p(L | X, y, β, w) ∝ p(L | X, β) × p(y | L, w)
                   \_________/   \________/
                     Prior        Likelihood

Posterior mean ≈ weighted average of:
  - X*β (genetic prediction)
  - (y - intercept)/w (phenotype-derived value)
```

So **L_sampled ≠ X*β**! The residual `L - X*β ≠ 0` carries phenotype information, which allows β to be updated.

### Information Flow Diagram

```
WITH SAMPLING (current NNMM):
┌─────────────────────────────────────────────────────────────┐
│  y (phenotype)                                              │
│    ↓                                                        │
│  HMC samples L from posterior p(L|y,β,w)                    │
│    ↓                                                        │
│  L_sampled ≠ X*β  (pulled toward explaining y)              │
│    ↓                                                        │
│  r = L_sampled - X*β ≠ 0  (contains y info!)                │
│    ↓                                                        │
│  β updated using r  ← y information reaches β!              │
└─────────────────────────────────────────────────────────────┘

WITH DETERMINISTIC (doesn't work):
┌─────────────────────────────────────────────────────────────┐
│  y (phenotype)                                              │
│    ↓                                                        │
│  Used only to update w in Layer 2→3                         │
│    ✗ (dead end)                                             │
│                                                             │
│  L = X*β  (by definition, no sampling)                      │
│    ↓                                                        │
│  r = X*β - X*β = 0  (no information!)                       │
│    ↓                                                        │
│  β not updated  ← y information BLOCKED!                    │
└─────────────────────────────────────────────────────────────┘
```

### Would Fixing the Weight (w=1) Help?

Another idea: fix the weight between latent and phenotype to 1:

```
L = X*β        (deterministic)
y = intercept + 1*L + e_y
```

This is mathematically equivalent to standard BayesC: `y = intercept + X*β + e_y`

**But it still doesn't work** because:
- Layer 1→2 residual = L - X*β = 0 (still zero!)
- The model structure treats layers **separately**
- Phenotype information doesn't "back-propagate" to β

To make this work, you'd need to **restructure the MCMC** to propagate the phenotype residual back through the layers (like neural network back-propagation). But this would break NNMM's ability to handle **mixed observed/missing omics** in the same layer.

### Comparison: Deterministic vs Sampling (Experimental)

When tested empirically with real data:

| Approach | Train Acc (EPV) | Train Acc (EBV) | Test Acc | Weight (w) |
|----------|-----------------|-----------------|----------|------------|
| **Sampling (default)** | 0.958 | 0.689 | 0.328 | -0.21 |
| **Deterministic** | 0.066 | 0.066 | 0.057 | ~0.003 |
| **JWAS BayesC** | - | 0.694 | 0.330 | - |

With sampling, NNMM matches JWAS BayesC (EBV ≈ 0.69). With deterministic latent, the weight collapses to ~0, giving essentially random predictions.

### Practical Recommendation

**Use the default sampling approach.** The EPV inflation on training data is expected behavior:

1. **For prediction accuracy**: Evaluate on **test set** results (EPV ≈ EBV on test)
2. **For genetic merit estimation**: Use **EBV** (not EPV on training)
3. **Don't worry about EPV inflation on training**: It's a natural consequence of phenotype-conditioned sampling

### Why This Design is Necessary for NNMM

NNMM's strength is handling the "omics in the middle" use case:
- **Observed omics** (some individuals): Use observed values directly
- **Missing omics** (some individuals): Sample from posterior (with phenotype info)
- **Latent traits** (all missing): Sample from posterior (with phenotype info)

A deterministic approach can't handle this mixed case elegantly. The sampling-based approach provides a **unified framework** that naturally handles all combinations of observed/missing data in the middle layer.

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
# - `EPV_*` uses observed middle-layer values (plus sampled missing/latent values)
# - `EBV_*` uses genotype-predicted middle-layer values
epv = out["EPV_Output_NonLinear"]
```

The model will automatically:
1. Use training individuals for parameter estimation
2. Exclude test individuals from variance estimation
3. Generate predictions (EPV) for all individuals

## See Also

- [Step-by-Step Tutorial](@ref) - Step-by-step guide to using NNMM.jl
- [Part 2: Mixed Effect Neural Network (NNMM)](@ref) - Example with completely missing middle layer
- [Part 3: NNMM with Intermediate Omics Features](@ref) - Example with observed omics

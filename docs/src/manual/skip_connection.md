# Skip Connection (Genotypes → Phenotypes) and Multi-Class Priors

This tutorial shows how to add a **Layer 1 → Layer 3 skip/shortcut term** so your 2→3 equation can include both:

- **Middle-layer features** (observed and/or partially missing omics), and
- a **direct genotype effect** on the phenotype.

This is useful when you want NNMM.jl to behave more like **JWAS multi-class BayesC** in the fully observed-omics case:

```
y = intercept + omics + genotypes
```

while still keeping NNMM’s ability to sample missing omics in the middle layer.

---

## Model overview

We assume a 3-layer NNMM:

```
Layer 1: Genotypes (X)
Layer 2: Omics/Intermediate traits (O, may contain missing values)
Layer 3: Phenotype (y)
```

### With skip connection

The 2→3 equation includes both marker classes:

```
y = intercept + O + X
```

Conceptually, NNMM fits two marker-effect “classes” in the 2→3 step:

- an **omics class** (effects for Layer 2 features), and
- a **genotype-skip class** (effects for Layer 1 markers directly on y).

---

## Step-by-step example

This example uses the built-in simulated dataset (same data source as `manual/tutorial.md`).

### 1) Load data and write separate files

```julia
using NNMM
using NNMM.Datasets
using DataFrames, CSV
using Random

Random.seed!(42)

geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

pheno_df = CSV.read(pheno_path, DataFrame)

# Layer 2 (omics)
omics_cols = vcat(:ID, [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
omics_file = "omics_data.csv"
CSV.write(omics_file, omics_df; missingstring="NA")

# Layer 3 (phenotype)
trait_df = pheno_df[:, [:ID, :trait1]]
trait_file = "phenotypes.csv"
CSV.write(trait_file, trait_df; missingstring="NA")
```

### 2) Define layers

```julia
layers = [
    Layer(layer_name="genotypes", data_path=[geno_path]),
    Layer(layer_name="omics", data_path=omics_file, missing_value="NA"),
    Layer(layer_name="phenotypes", data_path=trait_file, missing_value="NA"),
]
```

### 3) Define equations (including the skip term)

Key idea: the 2→3 equation explicitly includes `+ genotypes`.

```julia
eq1 = Equation(
    from_layer_name = "genotypes",
    to_layer_name   = "omics",
    equation        = "omics = intercept + genotypes",
    traits          = ["omic$i" for i in 1:10],
    method          = "BayesC",
    estimatePi      = true,
)

# Optional: set different priors/hyperparameters for each marker class in 2->3.
# The keys must match the layer names appearing in the equation ("omics" and "genotypes").
class_priors_23 = Dict(
    "omics" => (
        method = "BayesC",
        Pi = 0.0,
        estimatePi = true,
        # You can also set (G, df_G, estimate_variance_G, ...) here if desired.
    ),
    "genotypes" => (
        method = "BayesC",
        Pi = 0.0,
        estimatePi = true,
    ),
)

eq2 = Equation(
    from_layer_name = "omics",
    to_layer_name   = "phenotypes",
    equation        = "phenotypes = intercept + omics + genotypes",
    traits          = ["trait1"],
    class_priors    = class_priors_23,
    activation_function = "linear",
)
```

### 4) Run MCMC

```julia
results = runNNMM(layers, [eq1, eq2];
    chain_length = 5000,
    burnin = 1000,
    output_folder = "nnmm_skip_example",
    seed = 42,
)
```

---

## Which output should you compare to JWAS?

With observed omics available for an individual, the closest “JWAS-like” fitted value is:

- `results["EPV_Output_NonLinear"]` (uses **observed** Layer 2 values on output IDs, plus the genotype-skip term)

If you want a genotype-only prediction that does **not** assume omics will be available at prediction time, use:

- `results["EBV_NonLinear"]` (genotype-only **total EBV** when skip is present)

When a skip term is present, NNMM also reports the decomposition:

- `results["EBV_Indirect_NonLinear"]`: mediated component (Genotypes → predicted MiddleLayer → y)
- `results["EBV_Direct_Skip"]`: direct component from the 2→3 genotype-skip marker class only

and by definition:

```
EBV_NonLinear = EBV_Indirect_NonLinear + EBV_Direct_Skip
```

---

## Omics-class scaling for BayesC priors (JWAS-like)

When NNMM sets default BayesC(-like) priors for a marker class, it needs to convert a **target genetic variance** (`σ_g²`) into a **marker-effect variance** (`σ_a²`).

For a marker class with design matrix `X` (columns `x_j`), BayesC can be written (conceptually) as:

```math
g = X (\delta \odot a) \\
\delta_j \sim \mathrm{Bernoulli}(1-\pi) \\
a_j \sim \mathcal{N}(0, \sigma_a^2)
```

Assuming centered predictors and (approximately) independent columns, we have:

```math
\mathrm{Var}(g) \approx \sum_j \mathrm{Var}(x_j)\,\mathrm{Var}(\delta_j a_j)
               = \sum_j \mathrm{Var}(x_j)\,(1-\pi)\,\sigma_a^2
```

so the usual scaling is:

```math
\sigma_a^2 \approx \frac{\sigma_g^2}{(1-\pi)\,\sum_j \mathrm{Var}(x_j)}
```

### Why `nFeatures` worked before (and when it still does)

If your predictors are standardized so `Var(x_j) ≈ 1` for all features, then:

```math
\sum_j \mathrm{Var}(x_j) \approx n_\text{features}
```

which recovers the older NNMM scaling:

```math
\sigma_a^2 \approx \frac{\sigma_g^2}{(1-\pi)\,n_\text{features}}
```

### Why omics needs `∑ var(feature_j)` (unstandardized features)

Omics features are often **not standardized** and can have very different units/scales, so `∑ Var(x_j)` can be far from `nFeatures`.
Using `nFeatures` in that situation makes the prior (and therefore the posterior) depend on arbitrary measurement units, and it can prevent NNMM from matching **JWAS multi-class BayesC** in the complete-omics case.

To make the omics marker class behave more JWAS-like, NNMM now computes the omics analogue of JWAS’s genotype `sum2pq` as:

```math
\sum_j \mathrm{Var}(\text{omics feature}_j)
```

and uses that value in the `σ_g² → σ_a²` conversion for omics.

### Implementation notes

- The change is implemented in `src/markers/genotype_tools.jl` in `genetic2marker(::Omics, π)`.
- Per-feature variances are computed **ignoring `missing` values**, and the denominator is clamped away from 0 to avoid numerical issues.
- If you pre-standardize omics so each feature has variance ≈ 1, the new scaling reduces to the old `nFeatures` scaling.

## How this differs from the older “all-missing node” pattern

Before skip connections, a common workaround to represent a genotype shortcut was:

1. Add a **latent node** (e.g., `latent1`) in Layer 2.
2. Set that node to be **missing for all individuals**.
3. Fit:
   - 1→2: `omics = intercept + genotypes` (including `latent1` as an “omic” trait), and
   - 2→3: `phenotypes = intercept + omics` (no direct `+ genotypes` term).

This creates an *indirect* genotype path `genotypes → latent1 → phenotype` rather than a direct `genotypes → phenotype` class of marker effects.

### Statistical perspective (why “latent-only” is not the same as “skip”)

The “all-missing node” workaround can look similar to a skip term in a **linear-Gaussian** setting, but it is generally a **different statistical model**.

Consider the simplest case:

- Middle-layer latent node `z` is missing for everyone.
- Linear activation.
- Single trait.

**No skip + one fully-missing latent node**:

```
z = X a + e_z
y = w z + e_y
```

Substituting out `z` gives:

```
y = X (w a) + (w e_z + e_y)
```

This resembles a direct genotype effect with `β = w a`, but with important differences:

- **Factorized (rank-1) effect**: the marker effect is the product `β = w a`. The induced prior on `β` is not the same as putting BayesC directly on `β`.
- **Weak identifiability / scaling**: `a`, `w`, and the latent residual variance can trade off (e.g., scaling `a` up and `w` down leaves `w a` similar), which can slow MCMC mixing.
- **Different error model**: the effective noise includes `w e_z`, so the likelihood is not the same as a standard skip model unless you impose strong constraints.

**Skip (direct genotype class in 2→3)**:

```
y = Z_obs w_obs + X β + e_y
```

Here `β` is a directly parameterized marker-effect class with its own prior/hyperparameters (via `class_priors`), and it is typically more identifiable and stable.

So, **no-skip + fully-missing latent node is not guaranteed to produce the same posterior or predictions** as skip (especially under BayesC-style priors and when variances are learned).

### When some omics are partially missing

With partially missing omics nodes in the MiddleLayer, NNMM is a joint model that both:

- imputes missing omics using 1→2 (Genotypes → MiddleLayer), and
- fits 2→3 (MiddleLayer → Phenotypes) using the current/imputed MiddleLayer values.

Key difference:

- **No skip (full mediation)** forces genotype information to influence `y` *only through* the (partly imputed) MiddleLayer. If omics are very missing, prediction can become sensitive to how well that imputation is identified.
- **Skip (partial mediation)** adds a direct `X β` path, so prediction can remain stable even when omics are missing at training time and/or prediction time.

### Practical guidance

- If your goal is **JWAS-like multi-class BayesC** behavior and stable inference, prefer:
  - `phenotypes = intercept + omics + genotypes` in 2→3, and
  - do **not** add a fully-missing latent node unless you have a specific modeling reason.
- Use a fully-missing latent node only if you intentionally want an extra unobserved mediator dimension—and be aware it can introduce identifiability/mixing issues.

### Pros of the skip connection (new approach)

- **JWAS-like multi-class structure** in 2→3: `y = intercept + omics + genotypes` (two marker classes).
- **Separate priors per class** via `class_priors` (e.g., different `Pi`, different variance hyperparameters).
- **Uses observed omics directly** when they are available (via `EPV_Output_NonLinear`), without forcing genotype effects to go through a latent mediator node.
- **Still supports partially missing omics**: missing Layer 2 values are sampled (for `activation_function="linear"`) while conditioning on phenotype residuals that include the skip term.

### Pros of the all-missing node (older workaround)

- Keeps the model strictly in the original 2→3 form `phenotypes = intercept + omics` (no extra marker class in 2→3).
- If you truly want *all* genotype influence on `y` to be **mediated** through the middle layer, the latent-node pattern is closer to that biological assumption.

### Cons / tradeoffs

- **Interpretability**: with both `omics` and `genotypes` in 2→3, they can explain overlapping variance; “direct” vs “mediated” effects depend on modeling assumptions and priors.
- **MCMC mixing**: adding a second large marker block to 2→3 (genotype skip) can increase computation and autocorrelation; longer chains may be needed.
- **Prediction target choice**: `EPV_Output_NonLinear` (omics available) and `EBV_NonLinear` (genotype-only) answer different questions; make sure you evaluate the one that matches your deployment setting.

---

## Real-data reference script (repository)

For an end-to-end real-data comparison against JWAS multi-class BayesC, see:

- `test_real_data_codex.jl` (uses `TempTestData/nnmm_small_dataset`)

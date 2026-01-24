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

- `results["EBV_NonLinear"]` (propagates genotype information through the network to the output)

---

## How this differs from the older “all-missing node” pattern

Before skip connections, a common workaround to represent a genotype shortcut was:

1. Add a **latent node** (e.g., `latent1`) in Layer 2.
2. Set that node to be **missing for all individuals**.
3. Fit:
   - 1→2: `omics = intercept + genotypes` (including `latent1` as an “omic” trait), and
   - 2→3: `phenotypes = intercept + omics` (no direct `+ genotypes` term).

This creates an *indirect* genotype path `genotypes → latent1 → phenotype` rather than a direct `genotypes → phenotype` class of marker effects.

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


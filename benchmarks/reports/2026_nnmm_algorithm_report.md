# NNMM.jl Algorithm Walkthrough (MCMC) — Implementation-Level Report

Date: 2026-01-05

This report describes the **NNMM.jl** algorithm *as implemented* (not as an abstract statistical derivation), using the **same variable names as the code** and referencing the relevant source locations.

Scope:

- End-to-end flow from `runNNMM(...)` to the MCMC sampler `nnmm_MCMC_BayesianAlphabet(...)`.
- Starting values / initialization for the key state variables.
- The per-iteration MCMC update schedule (1→2, latent-omics, 2→3).
- How outputs are produced (including **EBV** and **EPV**).

---

## 0) Entry Points and Call Graph

The user-facing entry point is:

- `runNNMM(layers, equations; ...)` (`src/nnmm/run_mcmc.jl:76`)

The MCMC sampler is:

- `nnmm_MCMC_BayesianAlphabet(mme1, df1, mme2, df2)` (`src/nnmm/mcmc_bayesian.jl:44`)

The high-level call chain is:

1. `runNNMM(...)` reads input data, builds two `MME` objects (`mme_all[1]`, `mme_all[2]`), initializes missing omics, and builds MME matrices.
2. `runNNMM(...)` calls `nnmm_MCMC_BayesianAlphabet(mme_all[1], df1, mme_all[2], df2)` (`src/nnmm/run_mcmc.jl:1013`).
3. `nnmm_MCMC_BayesianAlphabet(...)` runs the MCMC loop and writes MCMC sample files.
4. After MCMC finishes, it calls `output_result(...)` to read/aggregate EBV/EPV samples and return a dictionary (`src/nnmm/mcmc_bayesian.jl:942`, `src/io/output.jl:108`).

---

## 1) Model Objects and Key Variables (Code Names)

### Core model objects

- `mme1`: the **1→2** MME object (genotypes → omics / latent traits) (`src/nnmm/mcmc_bayesian.jl:31`).
- `mme2`: the **2→3** MME object (omics → phenotypes) (`src/nnmm/mcmc_bayesian.jl:34`).

Both are `MME` objects that (conceptually) represent a linear mixed model of the form:

- `ySparse ≈ X*sol + (random/genetic terms)` with residual variance `R.val`

Key fields used heavily in the sampler:

- `mme*.ySparse`: response vector (for `mme1`, this is omics stacked as a vector; for `mme2`, this is phenotype values) (`src/nnmm/mcmc_bayesian.jl:147`, `src/nnmm/mcmc_bayesian.jl:296`).
- `mme*.X`: fixed-effect design matrix.
- `mme*.sol`: fixed-effect location parameters.
- `mme*.R.val`: residual variance (scalar or matrix).
- `mme*.M`: list of “marker-like” terms:
  - For `mme1`, `Mi` is typically **genotypes** (`Genotypes`), with `Mi.genotypes` and marker effects `Mi.α`.
  - For `mme2`, `Mi` is **omics** (`Omics`), treated as “marker-like covariates” with effects `Mi.α`.

### Residual working vectors

Within the MCMC loop, the code maintains “corrected phenotype” residuals:

- `ycorr1`: residual for the 1→2 model (`src/nnmm/mcmc_bayesian.jl:147`)
  - Intended invariant: `ycorr1 = y1 - X1*sol1 - Σ(Mi.genotypes * Mi.α[trait])`
- `ycorr2`: residual for the 2→3 model (`src/nnmm/mcmc_bayesian.jl:296`)
  - Intended invariant: `ycorr2 = y2 - X2*sol2 - Xomics*Mi.α[1]`

### “Neural network” coupling variables

NNMM couples the 2→3 regression into the latent-omics sampling step:

- `nonlinear_function`: activation function `g(·)` used in the NNBayes path (`src/nnmm/mcmc_bayesian.jl:59`).
- `mme1.weights_NN`: the 2→3 omics effect vector used in the NNBayes likelihood term (`src/nnmm/mcmc_bayesian.jl:203`, updated at `src/nnmm/mcmc_bayesian.jl:833`).
- `mme1.σ2_yobs`: set from `mme2.R.val` (`src/nnmm/mcmc_bayesian.jl:833`).
- `mme1.σ2_weightsNN`: set from the 2→3 effect variance `mme2.M[1].G.val` (`src/nnmm/mcmc_bayesian.jl:834`).

### Missingness bookkeeping for omics

Missingness is tracked in `mme1` (because omics live in the 1→2 response):

- `mme1.missingPattern`: boolean matrix, `true` where omics are *observed* (`src/nnmm/check.jl:187`).
- `mme1.incomplete_omics`: boolean vector, `true` for individuals with *any* missing omics (`src/nnmm/check.jl:222`).

---

## 2) Starting Values and Initialization

### 2.1 MCMC settings (`MCMCinfo`)

`runNNMM` constructs `MCMCinfo` for both layers (and stores it into each `mme`):

- 1→2: `missing_phenotypes=false` (`src/nnmm/run_mcmc.jl:842`, `src/nnmm/run_mcmc.jl:846`)
- 2→3: `missing_phenotypes=true` (`src/nnmm/run_mcmc.jl:855`, `src/nnmm/run_mcmc.jl:856`)

Key values used in the sampler:

- `chain_length`, `burnin`, `output_samples_frequency`, `output_folder` (`src/nnmm/mcmc_bayesian.jl:48`)

### 2.2 Aligning IDs and constructing “output” matrices

NNMM supports predicting for `output_ID` individuals (often *all genotyped IDs*).

- `align_genotypes(...)` creates `Mi.output_genotypes` aligned to `mme.output_ID` (`src/markers/genotype_tools.jl:93`).
- `align_omics(...)` creates omics `Mi.output_genotypes` aligned to `mme.output_ID` for the 2→3 layer (`src/markers/genotype_tools.jl:133`).
- `align_transformed_omics_with_phenotypes(...)` creates `Mi.aligned_omics_w_phenotype` aligned to `mme.obsID` and transformed by `nonlinear_function` (`src/markers/genotype_tools.jl:158`).

This alignment is performed during setup (`src/nnmm/run_mcmc.jl:988`, `src/nnmm/run_mcmc.jl:995`, `src/nnmm/run_mcmc.jl:997`) and also repeatedly inside MCMC after omics are updated (`src/nnmm/mcmc_bayesian.jl:637`).

### 2.3 Initializing missing omics values (data augmentation initial state)

Before MCMC starts, missing omics cells are filled so the sampler has an initial `mme1.ySparse`:

- `nnlmm_initialize_missing_with_mean(mme, df)` (`src/nnmm/check.jl:182`)

Behavior:

- For each omics column `i ∈ mme.lhsVec`:
  - If the column is **not** all missing, replace missing cells with `mean_omics_i` (`src/nnmm/check.jl:191`–`src/nnmm/check.jl:200`).
  - If the column is all missing:
    - If the phenotype `mme.yobs[j]` exists, initialize with `mme.yobs[j]` (`src/nnmm/check.jl:205`–`src/nnmm/check.jl:207`).
    - If phenotype is missing too (e.g., test), initialize with `0.0` (`src/nnmm/check.jl:202`–`src/nnmm/check.jl:205`).
- Sets:
  - `mme.missingPattern = .!ismissing.(Matrix(df[!, mme.lhsVec]))` (`src/nnmm/check.jl:187`)
  - `mme.incomplete_omics` as “any missing in row” (`src/nnmm/check.jl:219`–`src/nnmm/check.jl:223`)

### 2.4 Initializing `mme.sol` and `Mi.α`

MME matrices and starting values are created by:

- `init_mixed_model_equations(mme, df, starting_value)` (`src/io/input_validation.jl:434`)

Starting values when `starting_value == false`:

- `mme.sol = zeros(T, nsol)` (`src/io/input_validation.jl:441`)
  - Intercept is initialized to the trait mean for better chain alignment (`src/io/input_validation.jl:443`–`src/io/input_validation.jl:450`).
- Each marker-like term `Mi.α` is initialized to zeros (`src/io/input_validation.jl:472`–`src/io/input_validation.jl:480`).

### 2.5 Initial residual vectors (`ycorr1`, `ycorr2`)

At the start of `nnmm_MCMC_BayesianAlphabet`, residual working vectors are initialized from current state:

- `ycorr1 = vec(Matrix(mme1.ySparse) - mme1.X*mme1.sol) - Σ(Mi.genotypes*Mi.α)` (`src/nnmm/mcmc_bayesian.jl:147`)
- `ycorr2 = vec(Matrix(mme2.ySparse) - mme2.X*mme2.sol) - Xomics*Mi.α[1]` (`src/nnmm/mcmc_bayesian.jl:296`)

These are the residuals that marker-effect samplers (BayesC/BayesABC etc.) operate on.

### 2.6 Initial NN weights used by latent-omics sampler

If NNBayes is active (`nonlinear_function != false`), the sampler starts with:

- `mme1.weights_NN = zeros(mme1.nModels)` (`src/nnmm/mcmc_bayesian.jl:202`–`src/nnmm/mcmc_bayesian.jl:205`)

This is later overwritten each iteration from the 2→3 regression (`src/nnmm/mcmc_bayesian.jl:833`–`src/nnmm/mcmc_bayesian.jl:836`).

---

## 3) One Iteration of the MCMC Sampler (Detailed Schedule)

The loop is:

- `for iter = 1:chain_length` (`src/nnmm/mcmc_bayesian.jl:355`)

### 3.1 Block A — 1→2 updates (`mme1`)

#### A1) Update non-marker location parameters (`mme1.sol`)

The code updates `mme1.sol` via Gibbs sampling:

1. Temporarily restore full response: `ycorr1 += mme1.X*mme1.sol` (`src/nnmm/mcmc_bayesian.jl:373`).
2. Build RHS (`mme1.mmeRhs`) using `mme1.X' * ycorr1` (and optional weights) (`src/nnmm/mcmc_bayesian.jl:375`).
3. Call `Gibbs(...)` to sample `mme1.sol` (`src/nnmm/mcmc_bayesian.jl:375`).
4. Restore residual: `ycorr1 -= mme1.X*mme1.sol` (`src/nnmm/mcmc_bayesian.jl:378`).

This keeps the invariant form “ycorr = y - X*b - genetic terms” without recomputing from scratch.

Gibbs update formula (single-trait path):

- For each element `x[i]`:
  - `μ = invlhs * (b[i] - A[:,i]'x) + x[i]`
  - `x[i] = randn() * sqrt(invlhs * vare) + μ`
- Implemented in `Gibbs(A, x, b, vare)` (`src/solvers/iterative_solver.jl:39`).

#### A2) Update marker effects (`Mi.α`) for 1→2

For each genotype term `Mi` in `mme1.M`, the sampler updates marker effects using Bayesian Alphabet kernels (BayesC/BayesABC/RR-BLUP/etc.) operating on `ycorr1`:

- Entry to this section: `# 2. Marker Effects` (`src/nnmm/mcmc_bayesian.jl:388`)

The most common configuration in your benchmarks is `Mi.method == "BayesC"`, which calls:

- `BayesABC!(Mi, ycorr1, mme1.R.val, locus_effect_variances)` (`src/nnmm/mcmc_bayesian.jl:407`, core kernel in `src/markers/BayesABC.jl:41`)

Implementation details of `BayesABC!` (single-trait, variable names match `src/markers/BayesABC.jl:47`):

- Inputs:
  - `yCorr`: the residual vector (`ycorr1` here)
  - `α`: current marker effects (vector)
  - `β`: auxiliary effects (used by BayesB/BayesC implementations)
  - `δ`: inclusion indicators (0/1)
  - `vare`: residual variance (scalar; uses `invVarRes = 1/vare`)
  - `varEffects`: marker effect variances (vector; BayesC uses a constant variance replicated across loci)
  - `π`: inclusion probability (scalar for single trait)
- For each marker `j = 1:nMarkers` (`src/markers/BayesABC.jl:60`):
  1. Compute the conditional least-squares quantities:
     - `rhs = (dot(xRinv, yCorr) + xpRinvx[j]*α[j]) * invVarRes` (`src/markers/BayesABC.jl:62`)
     - `lhs = xpRinvx[j]*invVarRes + invVarEffects[j]` (`src/markers/BayesABC.jl:63`)
     - `invLhs = 1/lhs`, `gHat = rhs*invLhs` (`src/markers/BayesABC.jl:64`–`src/markers/BayesABC.jl:65`)
  2. Compute inclusion probability for `δ[j]` via the mixture log-odds:
     - `logDelta0 = log(π)` and `logDelta1 = -0.5*(log(lhs) + logVarEffects[j] - gHat*rhs) + log(1-π)` (`src/markers/BayesABC.jl:52`–`src/markers/BayesABC.jl:67`)
     - `probDelta1 = 1/(1 + exp(logDelta0 - logDelta1))` (`src/markers/BayesABC.jl:67`)
  3. Sample `δ[j]`:
     - If `rand() < probDelta1`:
       - Set `δ[j] = 1`
       - Sample `β[j] ~ Normal(gHat, invLhs)` and set `α[j] = β[j]` (`src/markers/BayesABC.jl:70`–`src/markers/BayesABC.jl:73`)
       - Update residual *in place* to reflect the α change:
         - `yCorr += (oldAlpha - α[j]) * x` via `BLAS.axpy!` (`src/markers/BayesABC.jl:74`)
     - Else:
       - Set `δ[j] = 0`, set `α[j] = 0` (`src/markers/BayesABC.jl:79`–`src/markers/BayesABC.jl:82`)
       - If `oldAlpha != 0`, restore it in the residual (`src/markers/BayesABC.jl:76`–`src/markers/BayesABC.jl:78`)

This residual-in-place update is why the main loop can keep using `ycorr1` without recomputing from scratch each time.

#### A3) Update variance components for 1→2

Key variance updates include:

- Marker-effect prior variance `Mi.G.val`:
  - Called as `sample_marker_effect_variance(Mi)` (`src/nnmm/mcmc_bayesian.jl:465`)
  - For BayesC single-trait: uses `nloci = sum(Mi.δ[1])` and samples:
    - `Mi.G.val = (dot(Mi.α[1], Mi.α[1]) + Mi.G.df*Mi.G.scale) / rand(Chisq(nloci + Mi.G.df))`
    - See `sample_marker_effect_variance(Mi)` and `sample_variance(...)` (`src/core/variance_components.jl:140`, `src/core/variance_components.jl:60`, `src/core/variance_components.jl:149`–`src/core/variance_components.jl:152`)
- Inclusion probability `Mi.π` (if `Mi.estimatePi == true`):
  - Single-trait: `Mi.π = samplePi(sum(Mi.δ[1]), Mi.nMarkers)` (`src/nnmm/mcmc_bayesian.jl:458`)
  - `samplePi(nEffects, nTotal) = rand(Beta(nTotal-nEffects+1, nEffects+1))` (`src/markers/Pi.jl:7`)
- Residual variance `mme1.R.val`:
  - Single-trait: `mme1.R.val = sample_variance(ycorr1, length(ycorr1), mme1.R.df, mme1.R.scale, invweights1)` (`src/nnmm/mcmc_bayesian.jl:521`–`src/nnmm/mcmc_bayesian.jl:523`)
  - `sample_variance(x,n,df,scale) = (dot(x,x) + df*scale) / rand(Chisq(n+df))` (`src/core/variance_components.jl:60`)

### 3.2 Block B — Sample/impute latent omics (`mme1.ySparse`)

This is the “data augmentation” step: impute missing omics entries by sampling from their conditional distribution.

#### B1) Define the current state (`yobs`, `ylats_old`, `μ_ylats`)

Within each iteration:

- `yobs = mme1.yobs` (`src/nnmm/mcmc_bayesian.jl:535`)
- `ylats_old = mme1.ySparse` (vector, then reshaped to `nobs × ntraits`) (`src/nnmm/mcmc_bayesian.jl:536`, `src/nnmm/mcmc_bayesian.jl:545`)
- `μ_ylats = vcat((getEBV(mme1, i) for i in 1:mme1.nModels)...)` (`src/nnmm/mcmc_bayesian.jl:539`)
  - `getEBV` computes the current 1→2 fitted values for output IDs from `mme.sol`, `Mi.output_genotypes`, and `Mi.α` (`src/io/output.jl:278`).

Interpretation:

- `μ_ylats` is the **geno→omics conditional mean** given current 1→2 parameters.
- Missing omics are sampled around this mean, with variance `mme1.R.val` and optionally informed by the phenotype likelihood (via HMC/MH) if `yobs` is available.

#### B2) Select which individuals to update

- `incomplete_omics = mme1.incomplete_omics` (`src/nnmm/mcmc_bayesian.jl:551`)
- Partition:
  - `incomplete_with_yobs = incomplete_omics .& .!ismissing.(yobs)`
  - `incomplete_no_yobs   = incomplete_omics .&  ismissing.(yobs)` (`src/nnmm/mcmc_bayesian.jl:554`–`src/nnmm/mcmc_bayesian.jl:556`)

This matters because the phenotype likelihood term is only defined for observed `yobs` (and for phenotyped IDs in `mme2.obsID`).

#### B3) Sample missing omics for `incomplete_with_yobs`

If `mme1.is_activation_fcn == true`, NNMM uses HMC:

- Call: `hmc_one_iteration(10, 0.1, ...)` (`src/nnmm/mcmc_bayesian.jl:564`)
- HMC kernel: `src/nnmm/hmc.jl:92`

The HMC target corresponds to (up to constants):

- 1→2 term: `-(ycorr_reshape)^2 / (2*mme1.R.val)`
- 2→3 term: `-(ycorr_yobs)^2 / (2*σ2_yobs)`

Where:

- `weights_NN` is `mme1.weights_NN` (the current 2→3 omics effects) (`src/nnmm/mcmc_bayesian.jl:567`).
- `ycorr_yobs` is passed in as the relevant slice of `ycorr2` (`src/nnmm/mcmc_bayesian.jl:569`).
- Gradients/log-probability are computed in:
  - `calc_gradient_z(...)` (`src/nnmm/hmc.jl:62`)
  - `calc_log_p_z(...)` (`src/nnmm/hmc.jl:76`)

If `mme1.is_activation_fcn == false` (user-defined `nonlinear_function`), NNMM uses a Metropolis-Hastings style update (see the `else` branch after the HMC call in `src/nnmm/mcmc_bayesian.jl:563`).

#### B4) Sample missing omics for `incomplete_no_yobs`

For phenotype-missing individuals (e.g., test), the sampler uses only the 1→2 conditional Normal:

- `candidates = μ_ylats + Normal(0, mme1.R.val)` with scalar/Diagonal/full-matrix cases (`src/nnmm/mcmc_bayesian.jl:605`–`src/nnmm/mcmc_bayesian.jl:620`)

#### B5) Preserve observed omics in partial-missing rows

For individuals with partially observed omics, the observed entries must not be overwritten:

- `ylats_old[mme1.missingPattern] .= ylats_old2[mme1.missingPattern]` (`src/nnmm/mcmc_bayesian.jl:623`–`src/nnmm/mcmc_bayesian.jl:624`)

This means only missing cells are “imputed”; observed cells remain fixed to their observed values.

#### B6) Write updated omics back into both layers and update residuals

After sampling:

- Update 1→2 response:
  - `mme1.ySparse = vec(ylats_old)` (`src/nnmm/mcmc_bayesian.jl:626`–`src/nnmm/mcmc_bayesian.jl:627`)
  - Update residual efficiently:
    - `ycorr1[:] += mme1.ySparse - vec(ylats_old2)` (`src/nnmm/mcmc_bayesian.jl:631`)
- Update omics data backing the 2→3 model:
  - `mme2.M[1].data[!, featureID] = ylats_old` (`src/nnmm/mcmc_bayesian.jl:635`)
  - `align_transformed_omics_with_phenotypes(mme2, nonlinear_function)` (`src/nnmm/mcmc_bayesian.jl:637`)
- Recompute 2→3 residual from scratch to avoid drift:
  - `ycorr2 = y - X*b - Xomics*α` (`src/nnmm/mcmc_bayesian.jl:638`–`src/nnmm/mcmc_bayesian.jl:645`)

### 3.3 Block C — 2→3 updates (`mme2`)

Now the model uses the (possibly updated) omics to update the phenotype regression.

#### C1) Update non-marker location parameters (`mme2.sol`)

Analogous to the 1→2 case:

- Entry to this section: `# 1. Non-Marker Location Parameters` (`src/nnmm/mcmc_bayesian.jl:653`)

#### C2) Update omics effects (`mme2.M[1].α[1]`) and priors

This is the 2→3 “marker effects” update, except the “markers” are omics features:

- Entry: `# 2. Marker Effects` (`src/nnmm/mcmc_bayesian.jl:693`)

In the common single-trait BayesC configuration, it calls:

- `BayesABC!(Mi, ycorr2, mme2.R.val, locus_effect_variances)` (`src/nnmm/mcmc_bayesian.jl:712`, kernel in `src/markers/BayesABC.jl:41`)

and then updates:

- `Mi.π` via `Mi.π = samplePi(sum(Mi.δ[1]), Mi.nFeatures)` (`src/nnmm/mcmc_bayesian.jl:771`, Beta sampler in `src/markers/Pi.jl:7`)
- `Mi.G.val` via `sample_marker_effect_variance(Mi)` (`src/nnmm/mcmc_bayesian.jl:778`, implementation in `src/core/variance_components.jl:140`)

This block produces:

- `mme2.M[1].α[1]`: effect of each transformed omics feature on the phenotype (the NN “weights”).
- `mme2.M[1].G.val`: effect prior variance for 2→3.
- `mme2.R.val`: phenotype residual variance.

#### C3) Synchronize NNBayes coupling parameters back into `mme1`

After 2→3 updates, NNMM copies the current 2→3 parameters into `mme1` so the latent-omics sampling step can include the phenotype likelihood term:

- `mme1.σ2_yobs      = mme2.R.val` (`src/nnmm/mcmc_bayesian.jl:833`)
- `mme1.σ2_weightsNN = mme2.M[1].G.val` (`src/nnmm/mcmc_bayesian.jl:834`)
- `mme1.weights_NN   = mme2.M[1].α[1]` (`src/nnmm/mcmc_bayesian.jl:835`–`src/nnmm/mcmc_bayesian.jl:836`)

---

## 4) How Outputs Are Generated (EBV + EPV)

There are two phases:

1. **During MCMC**: write samples to text files.
2. **After MCMC**: read those sample files and compute posterior means/variances for EBV/EPV.

### 4.1 Output file setup (headers)

When `output_samples_frequency != 0`, the sampler creates output files:

- `outfile = output_MCMC_samples_setup(...)` (`src/nnmm/mcmc_bayesian.jl:177`, setup function in `src/io/output.jl:325`)
- Additional NNMM-specific files:
  - `MCMC_samples_layer2_residual_variance.txt` and `MCMC_samples_layer2_effect_variance.txt` (`src/nnmm/mcmc_bayesian.jl:181`–`src/nnmm/mcmc_bayesian.jl:185`)
  - `MCMC_samples_EPV_NonLinear.txt` (`src/nnmm/mcmc_bayesian.jl:188`–`src/nnmm/mcmc_bayesian.jl:190`)
  - `MCMC_samples_EPV_Output_NonLinear.txt` (`src/nnmm/mcmc_bayesian.jl:191`–`src/nnmm/mcmc_bayesian.jl:195`)

EPV headers are written once:

- `EPV_NonLinear` header uses the **phenotyped IDs** (2→3 `obsID`) (`src/nnmm/mcmc_bayesian.jl:346`–`src/nnmm/mcmc_bayesian.jl:349`).
- `EPV_Output_NonLinear` header uses `mme1.output_ID` (`src/nnmm/mcmc_bayesian.jl:350`–`src/nnmm/mcmc_bayesian.jl:352`).

### 4.2 Saving samples each iteration

On iterations that satisfy:

- `iter > burnin && (iter-burnin) % output_samples_frequency == 0` (`src/nnmm/mcmc_bayesian.jl:841`)

NNMM writes:

#### A) Parameter samples and EBV samples (`output_MCMC_samples`)

- Called as: `output_MCMC_samples(mme1, mme1.R.val, ..., outfile)` (`src/nnmm/mcmc_bayesian.jl:859`)
- EBV sampling:
  - Computes per-trait “EBV” vectors via `getEBV(mme, traiti)` (`src/io/output.jl:278`, used at `src/io/output.jl:520`–`src/io/output.jl:528`).
  - For NNMM, it computes `EBV_NonLinear` using:
    - `BV_NN = mme.nonlinear_function.(EBVmat) * mme.weights_NN` (activation-function path) (`src/io/output.jl:546`–`src/io/output.jl:556`)
  - Saves into `MCMC_samples_EBV_NonLinear.txt` (`src/io/output.jl:556`).

Important detail:

- `mme.weights_NN` used inside `output_MCMC_samples` is `mme1.weights_NN`, which was set from the latest 2→3 update (`src/nnmm/mcmc_bayesian.jl:833`–`src/nnmm/mcmc_bayesian.jl:836`).

So **EBV_NonLinear** is:

- “geno → predicted omics (via 1→2)” then “omics → phenotype (via activation + current 2→3 weights)”

#### B) EPV samples (computed in `mcmc_bayesian.jl`)

EPV is written outside of `output_MCMC_samples` because it needs access to `mme2`’s aligned/phenotyped omics:

- `EPV_NonLinear` (phenotyped IDs only):
  - Uses `observed_omics = mme2.M[1].aligned_omics_w_phenotype` (`src/nnmm/mcmc_bayesian.jl:872`)
  - For activation-function path: `EPV_NN = observed_omics * mme1.weights_NN` (`src/nnmm/mcmc_bayesian.jl:873`–`src/nnmm/mcmc_bayesian.jl:880`)
  - Writes to `MCMC_samples_EPV_NonLinear.txt` (`src/nnmm/mcmc_bayesian.jl:880`)
- `EPV_Output_NonLinear` (all `output_ID`):
  - Starts from `omics_out = ylats_old` (current sampled/imputed omics for everyone) (`src/nnmm/mcmc_bayesian.jl:887`–`src/nnmm/mcmc_bayesian.jl:892`)
  - Aligns to output IDs via `mkmat_incidence_factor(...)` if needed (`src/nnmm/mcmc_bayesian.jl:889`–`src/nnmm/mcmc_bayesian.jl:893`)
  - Applies the activation and weights:
    - `omics_out = nonlinear_function.(omics_out)` then `EPV_out = omics_out * mme1.weights_NN` (`src/nnmm/mcmc_bayesian.jl:894`–`src/nnmm/mcmc_bayesian.jl:897`)
  - Writes to `MCMC_samples_EPV_Output_NonLinear.txt` (`src/nnmm/mcmc_bayesian.jl:900`)

So EPV is:

- “use observed/sampled omics values” then “omics → phenotype (via activation + current 2→3 weights)”

### 4.3 Post-MCMC aggregation into returned results

At the end of MCMC:

- `output = output_result(mme1, output_folder, ...)` (`src/nnmm/mcmc_bayesian.jl:942`)

`output_result`:

- Returns location parameters and variances from *online posterior means* (`src/io/output.jl:108`–`src/io/output.jl:152`).
- For EBV/EPV, reads the MCMC sample text files and computes posterior mean/variance:
  - EBV: `output[EBVkey] = DataFrame([ID, mean(samples), var(samples)])` (`src/io/output.jl:154`–`src/io/output.jl:169`)
  - EPV (phenotyped): `output["EPV_NonLinear"] = ...` (`src/io/output.jl:187`–`src/io/output.jl:196`)
  - EPV (all output IDs): `output["EPV_Output_NonLinear"] = ...` (`src/io/output.jl:198`–`src/io/output.jl:209`)

---

## 5) Summary: EBV vs EPV (Implemented Definitions)

Using the code’s naming:

- **EBV_NonLinear** (saved in `output["EBV_NonLinear"]`) is computed from **genotype-predicted omics** (`getEBV(mme1, ...)`) and then mapped to phenotype using `nonlinear_function` and `mme1.weights_NN` (`src/io/output.jl:520`, `src/io/output.jl:546`).
- **EPV_NonLinear** is computed from **phenotyped individuals’ observed/sampled omics** (`mme2.M[1].aligned_omics_w_phenotype`) and then multiplied by `mme1.weights_NN` (`src/nnmm/mcmc_bayesian.jl:872`–`src/nnmm/mcmc_bayesian.jl:880`).
- **EPV_Output_NonLinear** is computed from **all output individuals’ current sampled omics** (`ylats_old`, aligned to `output_ID`), then mapped the same way (`src/nnmm/mcmc_bayesian.jl:887`–`src/nnmm/mcmc_bayesian.jl:900`).

---

## 6) Pseudocode (Matches Code Schedule)

This pseudocode follows the *exact control flow* in `nnmm_MCMC_BayesianAlphabet` (`src/nnmm/mcmc_bayesian.jl:44`):

```text
setup: ycorr1, ycorr2, outfile, mme1.weights_NN

for iter in 1:chain_length:
  # 1->2 block (mme1)
  update mme1.sol (Gibbs) using ycorr1
  update mme1 marker effects Mi.α (Bayesian Alphabet) using ycorr1
  update mme1 variance components (Mi.G.val, Mi.π, mme1.R.val, ...)

  # latent omics update (mme1.ySparse)
  compute μ_ylats from getEBV(mme1, ...)
  reshape mme1.ySparse -> ylats_old (nobs×ntraits)
  sample missing omics for incomplete_with_yobs using HMC/MH (uses mme1.weights_NN, ycorr2)
  sample missing omics for incomplete_no_yobs from Normal(μ_ylats, mme1.R.val)
  restore observed cells using mme1.missingPattern
  write back mme1.ySparse and update ycorr1

  # update 2->3 data and residuals
  write ylats_old into mme2.M[1].data
  align_transformed_omics_with_phenotypes(mme2, nonlinear_function)
  recompute ycorr2 = y - X*b - Xomics*α

  # 2->3 block (mme2)
  update mme2.sol (Gibbs) using ycorr2
  update mme2 omics effects Mi.α (Bayesian Alphabet) using ycorr2
  update mme2 variance components (mme2.R.val, mme2.M[1].G.val, ...)

  # synchronize coupling parameters for next iteration
  mme1.σ2_yobs = mme2.R.val
  mme1.σ2_weightsNN = mme2.M[1].G.val
  mme1.weights_NN = mme2.M[1].α[1]

  # save samples if requested
  if iter is a saved iteration:
    output_MCMC_samples(mme1, ...)     # includes EBV_NonLinear
    write EPV_NonLinear and EPV_Output_NonLinear

return output_result(mme1, output_folder, ...)
```

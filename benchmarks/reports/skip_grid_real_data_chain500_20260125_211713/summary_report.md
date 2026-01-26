# NNMM Skip Connection Validation Report (Real Data, chain length = 500)

Run directory: `dev_workspace/skip_grid_real_data_chain500_20260125_211713`

## 1) Experiment setup

- **Data**: `TempTestData/nnmm_small_dataset/input_files/data1`
  - Genotypes after QC: **4669 markers**, **1637 individuals**
  - Omics: **20 observed features** (and an optional extra `latent1` node that is all missing)
  - Phenotype: `FIP` with **1310 training** + **327 validation** individuals
- **Grid** (3×3): training-omics missing rate × testing-omics missing rate ∈ {0%, 50%, 100%}
- **Models**
  - **omics**: `Phenotypes = intercept + MiddleLayer`
  - **skip**:  `Phenotypes = intercept + MiddleLayer + Genotypes` (multi-class BayesC; separate priors per class; MiddleLayer includes `latent1`)
  - **skip_nolatent**: same as **skip**, but MiddleLayer excludes `latent1` (only the 20 observed omics nodes)
- **MCMC**: `chain_length=500`, `burnin=125`, `output_samples_frequency=25`, `seed=12345`
- **Evaluation metric**: Pearson correlation(`pred`, `true y`) on validation IDs

## 2) Full model specification (NNMM skip_nolatent)

This section describes the exact NNMM model used in the **JWAS sanity check** (complete case: train=0%, test=0%).

- **Layers**
  - Layer 1 (**Genotypes**): 4669 SNP markers
  - Layer 2 (**MiddleLayer**): 20 observed omics nodes (no `latent1`)
  - Layer 3 (**Phenotypes**): `y` (=`FIP`)
- **Equations**
  - **1→2 (Genotypes → MiddleLayer)**: for each omics node `m_j`  
    `m_j = intercept + Genotypes` (BayesC)
  - **2→3 (MiddleLayer → Phenotypes)**:  
    `y = intercept + MiddleLayer + Genotypes` (BayesC, **multi-class** with separate priors for `MiddleLayer` vs `Genotypes`)
- **Activation**: linear

## 3) Prediction outputs (names used in this report)

- **JWAS EBV**: `EBV_y` from JWAS multi-class BayesC (`y = intercept + geno + omics_class`)
- **NNMM EPV_Output**: `EPV_Output_NonLinear` (posterior mean)  
  Predicted phenotypic value on **output IDs** using the current MiddleLayer values (observed/imputed) and the 2→3 mapping; includes the genotype-skip term. (No intercept.)
- **NNMM EBV_total**: `EBV_NonLinear` (posterior mean)  
  Redefined as total EBV when skip is present:
  `EBV_total = EBV_Indirect_NonLinear + EBV_Direct_Skip`
- **NNMM EBV_Indirect**: `EBV_Indirect_NonLinear` (posterior mean)  
  Mediated component (Genotypes → predicted MiddleLayer → y)
- **NNMM EBV_Direct**: `EBV_Direct_Skip` (posterior mean)  
  Direct 2→3 genotype-skip marker class contribution only

## 4) EBV definitions (NNMM output)

When a skip term is present (2→3 includes `+ Genotypes`), NNMM outputs:

- `EBV_Indirect_NonLinear`: **mediated** component (genotype → predicted omics → phenotype via NN weights)
- `EBV_Direct_Skip`: **direct** component from the 2→3 genotype-skip marker class blocks only
- `EBV_NonLinear` (**total**): `EBV_Indirect_NonLinear + EBV_Direct_Skip`

When no skip is present (omics-only model), `EBV_Direct_Skip` is identically 0.

## 5) Main results (validation accuracy)

Each cell is reported as `omics / skip(latent1) / skip(no-latent)`.

### 5.1 Validation Acc: EPV_Output_NonLinear

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.1988 / 0.3148 / 0.3142 | 0.1388 / 0.3153 / 0.3048 | 0.0504 / 0.3241 / 0.3295 |
| 50% | 0.1989 / 0.3532 / 0.2955 | 0.0977 / 0.3271 / 0.3339 | -0.0322 / 0.2902 / 0.3402 |
| 100% | 0.1454 / 0.2854 / 0.3052 | 0.0117 / 0.3369 / 0.2998 | 0.2399 / 0.2594 / 0.2663 |

### 5.2 Validation Acc: EBV_NonLinear (Total)

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.1601 / 0.3094 / 0.3096 | 0.1021 / 0.3092 / 0.3038 | 0.0275 / 0.3230 / 0.3302 |
| 50% | 0.1780 / 0.3512 / 0.2754 | 0.0366 / 0.3266 / 0.3200 | -0.0327 / 0.2953 / 0.3413 |
| 100% | 0.2692 / 0.2891 / 0.3024 | 0.0145 / 0.3370 / 0.2998 | 0.2399 / 0.2594 / 0.2663 |

### 5.3 Validation Acc: EBV_Indirect_NonLinear

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.1601 / 0.1210 / 0.1098 | 0.1021 / 0.0880 / 0.1594 | 0.0275 / 0.0430 / 0.0768 |
| 50% | 0.1780 / 0.0909 / 0.2061 | 0.0366 / 0.1127 / 0.1414 | -0.0327 / 0.0275 / 0.0506 |
| 100% | 0.2692 / -0.0735 / -0.0242 | 0.0145 / -0.1098 / 0.1156 | 0.2399 / 0.1991 / 0.2099 |

### 5.4 Validation Acc: EBV_Direct_Skip

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.0000 / 0.3078 / 0.3087 | 0.0000 / 0.3077 / 0.3027 | 0.0000 / 0.3391 / 0.3298 |
| 50% | 0.0000 / 0.3491 / 0.2675 | 0.0000 / 0.3263 / 0.3133 | 0.0000 / 0.3126 / 0.3401 |
| 100% | 0.0000 / 0.2899 / 0.3031 | 0.0000 / 0.3373 / 0.2997 | 0.0000 / 0.2687 / 0.2663 |

## 6) Key takeaways

- **Skip helps consistently**: across all 9 scenarios, **skip** beats **omics** on both `EPV_Output_NonLinear` and `EBV_NonLinear (Total)`.
- **skip_nolatent is mixed vs skip(latent1)**: it can be better in some settings (notably with `test_missing=100%`), but worse in others (notably `train_missing=50%, test_missing=0%`).
- **Best-performing scenario (skip)**:
  - `train_missing=50%`, `test_missing=0%`: `val_acc_epv=0.3532`, `val_acc_ebv_total=0.3512`.
- **Best-performing scenario (skip_nolatent)**:
  - `train_missing=50%`, `test_missing=100%`: `val_acc_epv=0.3402`, `val_acc_ebv_total=0.3413`.
- **Direct skip dominates most scenarios**:
  - When `test_missing ∈ {0%, 50%}`, `corr(EBV_total, EBV_direct) ≈ 0.9997–1.0000`, and the direct term explains ~all variance in `EBV_total`.
  - When `test_missing=100%`, the indirect part becomes more important, especially at `train_missing=100%, test_missing=100%`:
    - `corr(EBV_total, EBV_direct)=0.6785` and indirect variance share is large (latent-only middle layer).

## 7) Why the “skip EBV” numbers changed vs the previous report

In the earlier chain-500 grid report (`dev_workspace/skip_grid_real_data_chain500_20260124_185645`),
`EBV_NonLinear` for **skip** effectively reflected only the **indirect** (omics-mediated) component.
After redefining `EBV_NonLinear` as **total** (= indirect + direct), skip-model validation EBV increased substantially:

Δ `val_acc_ebv(skip)` = new total EBV − old EBV_NonLinear (indirect-only)

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | +0.1884 | +0.2213 | +0.2800 |
| 50% | +0.2603 | +0.2139 | +0.2678 |
| 100% | +0.3626 | +0.4468 | +0.0604 |

## 8) JWAS sanity check (complete-omics case only: train=0%, test=0%)

**Models compared**

- **JWAS baseline** (two marker classes):
  - `y = intercept + geno + omics_class`
  - `geno`: SNP genotypes (4669 markers)
  - `omics_class`: the 20 complete observed omics features treated as a second marker class
- **NNMM**: `skip_nolatent` (no `latent1`), with:
  - 1→2: `m_j = intercept + Genotypes` (for each of 20 omics nodes)
  - 2→3: `y = intercept + MiddleLayer + Genotypes` (multi-class BayesC)

For the JWAS-vs-NNMM comparison, we report correlation between **JWAS EBV** and **NNMM EPV_Output** (and also NNMM EBV decompositions) on the same ID sets.

Correlation with JWAS EBV (`y = intercept + geno + omics_class`) on the same IDs:

- `r(JWAS EBV, NNMM EPV_Output)` train: **0.9611**, val: **0.9542**
- `r(JWAS EBV, NNMM EBV_total)`  train: **0.9606**, val: **0.9525**
- `r(JWAS EBV, NNMM EBV_direct)` train: **0.9604**, val: **0.9524**
- `r(JWAS EBV, NNMM EBV_indirect)` train: **0.1714**, val: **0.1386**

Accuracy vs true y (complete case):

- JWAS EBV: train **0.6861**, val **0.3345**
- NNMM EPV_Output: train **0.6581**, val **0.3142**
- NNMM EBV_total: train **0.6567**, val **0.3096**

## 9) Runtime (chain length = 500)

- Total NNMM runtime for 27 fits (9 grid cells × 3 models): **~61.1 minutes** (sum of per-run wall times).
- Per-run wall time (mean over the 9 scenarios):
  - **omics**: ~159.5 s
  - **skip**:  ~152.6 s
  - **skip_nolatent**: ~95.0 s

# Skip Connection 3×3 Grid (Real Data)

- Data: `TempTestData/nnmm_small_dataset/input_files/data1`
- Omics features: 20 observed; the default MiddleLayer includes an extra `latent1` node (all missing), and `skip_nolatent` removes it.
- MCMC: chain_length=500, burnin=125, seed=12345
- Models:
  - **omics**: `y = intercept + MiddleLayer`
  - **skip**:  `y = intercept + MiddleLayer + Genotypes` (multi-class BayesC via `class_priors`, MiddleLayer includes `latent1`)
  - **skip_nolatent**: same as **skip**, but MiddleLayer excludes `latent1` (only the 20 observed omics nodes)

The tables below report **validation accuracy** (correlation with true y on the validation IDs),
as `omics / skip(latent1) / skip(no-latent)` in each cell.

EBV definitions:
- `EBV_NonLinear` = `EBV_Indirect_NonLinear` + `EBV_Direct_Skip` (total EBV; direct term is nonzero only under skip models).

### Validation Acc: EPV_Output_NonLinear (omics / skip(latent1) / skip(no-latent))

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.1988 / 0.3148 / 0.3142 | 0.1388 / 0.3153 / 0.3048 | 0.0504 / 0.3241 / 0.3295 |
| 50% | 0.1989 / 0.3532 / 0.2955 | 0.0977 / 0.3271 / 0.3339 | -0.0322 / 0.2902 / 0.3402 |
| 100% | 0.1454 / 0.2854 / 0.3052 | 0.0117 / 0.3369 / 0.2998 | 0.2399 / 0.2594 / 0.2663 |

### Validation Acc: EBV_NonLinear (Total) (omics / skip(latent1) / skip(no-latent))

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.1601 / 0.3094 / 0.3096 | 0.1021 / 0.3092 / 0.3038 | 0.0275 / 0.323 / 0.3302 |
| 50% | 0.178 / 0.3512 / 0.2754 | 0.0366 / 0.3266 / 0.32 | -0.0327 / 0.2953 / 0.3413 |
| 100% | 0.2692 / 0.2891 / 0.3024 | 0.0145 / 0.337 / 0.2998 | 0.2399 / 0.2594 / 0.2663 |

### Validation Acc: EBV_Indirect_NonLinear (omics / skip(latent1) / skip(no-latent))

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.1601 / 0.121 / 0.1098 | 0.1021 / 0.088 / 0.1594 | 0.0275 / 0.043 / 0.0768 |
| 50% | 0.178 / 0.0909 / 0.2061 | 0.0366 / 0.1127 / 0.1414 | -0.0327 / 0.0275 / 0.0506 |
| 100% | 0.2692 / -0.0735 / -0.0242 | 0.0145 / -0.1098 / 0.1156 | 0.2399 / 0.1991 / 0.2099 |

### Validation Acc: EBV_Direct_Skip (omics / skip(latent1) / skip(no-latent))

| Train \\ Test | 0% | 50% | 100% |
|---:|---:|---:|---:|
| 0% | 0.0 / 0.3078 / 0.3087 | 0.0 / 0.3077 / 0.3027 | 0.0 / 0.3391 / 0.3298 |
| 50% | 0.0 / 0.3491 / 0.2675 | 0.0 / 0.3263 / 0.3133 | 0.0 / 0.3126 / 0.3401 |
| 100% | 0.0 / 0.2899 / 0.3031 | 0.0 / 0.3373 / 0.2997 | 0.0 / 0.2687 / 0.2663 |

## JWAS sanity (complete omics, skip_nolatent model)

- r(JWAS EBV, NNMM EPV_Output) train: 0.9611
- r(JWAS EBV, NNMM EPV_Output) val:   0.9542


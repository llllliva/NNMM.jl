#!/usr/bin/env julia
#=
Test: NNMM with Latent + Observed Omics vs JWAS NNMM
WITH Train/Test Split (Test has observed omics but missing phenotype)

Scenario:
- Layer 1: Genotypes (X)
- Layer 2: Missing latent (L) + Observed omics (O)  
- Layer 3: Phenotype (y)

Training (80%): y observed, O observed, L missing
Test (20%): y MISSING, O observed, L missing
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra

println("="^60)
println("Test: NNMM (Latent + Observed Omics) - WITH Train/Test Split")
println("="^60)

Random.seed!(12345)

# ============================================================
# Step 1: Create simulated data
# ============================================================
println("\n--- Step 1: Creating simulated data ---")

n = 500
p = 100  # SNPs

# Generate genotypes
X = rand([0.0, 1.0, 2.0], n, p)

# True marker effects for latent
β_latent = randn(p) .* 0.1

# True latent values (genotype effects)
L_true = X * β_latent

# Generate observed omics (correlated with genotypes)
β_omics = randn(p) .* 0.05
O_true = X * β_omics + randn(n) .* 0.3

# True weights from middle layer to phenotype
w_latent = 0.6
w_omics = 0.4

# Generate phenotypes
y_true = w_latent .* L_true + w_omics .* O_true + randn(n) .* 0.5

# Create IDs
ids = ["ind_$i" for i in 1:n]

# ============================================================
# Step 2: Train/Test Split (80/20)
# ============================================================
println("\n--- Step 2: Train/Test Split ---")

n_train = Int(round(0.8 * n))
n_test = n - n_train

shuffle_idx = randperm(n)
train_idx = shuffle_idx[1:n_train]
test_idx = shuffle_idx[n_train+1:end]

println("Training: $n_train individuals")
println("Test: $n_test individuals (missing phenotype, observed omics)")

# ============================================================
# Step 3: Save data files
# ============================================================
println("\n--- Step 3: Saving data files ---")

tmpdir = mktempdir()
println("Using temp directory: $tmpdir")

# Genotypes
geno_df = DataFrame(ID = ids)
for j in 1:p
    geno_df[!, "snp$j"] = X[:, j]
end
geno_path = joinpath(tmpdir, "genotypes.csv")
CSV.write(geno_path, geno_df)

# Phenotypes with missing for test set (use NA string for missing)
pheno_df = DataFrame(ID = ids, y = y_true)
pheno_path = joinpath(tmpdir, "phenotypes.csv")

# Write phenotypes - replace test indices with NA after writing
open(pheno_path, "w") do io
    println(io, "ID,y")
    for i in 1:n
        if i in test_idx
            println(io, "$(ids[i]),NA")
        else
            println(io, "$(ids[i]),$(y_true[i])")
        end
    end
end

# Latent + Omics
# latent1: mostly missing (95% missing, 5% observed to ensure numeric type detection)
# omics1: ALL observed for both train and test
# This setup tests: test individuals have observed omics but missing phenotype
latent_omics_path = joinpath(tmpdir, "latent_omics.csv")

# Pick a small subset of training individuals to have "observed" latent1
# (simulates partially observed intermediate traits)
n_latent_observed = max(5, Int(round(0.05 * n_train)))  # 5% of training
latent_observed_idx = train_idx[1:n_latent_observed]

open(latent_omics_path, "w") do io
    println(io, "ID,latent1,omics1")
    for i in 1:n
        latent_val = i in latent_observed_idx ? L_true[i] : "NA"
        println(io, "$(ids[i]),$latent_val,$(O_true[i])")
    end
end

println("Latent1: $(n_latent_observed) observed, $(n - n_latent_observed) missing")
println("Omics1: all $n observed")

println("Files saved:")
println("  - $geno_path")
println("  - $pheno_path")
println("  - $latent_omics_path")

# ============================================================
# Step 4: Run NNMM.jl
# ============================================================
println("\n--- Step 4: Running NNMM.jl ---")

using NNMM

# Layer 1: Genotypes
layer1 = Layer(
    layer_name = "Genotypes",
    data_path = geno_path,
    header = true
)

# Layer 2: Latent + Observed Omics
layer2 = Layer(
    layer_name = "MiddleLayer",
    data_path = latent_omics_path,
    header = true,
    missing_value = "NA"
)

# Layer 3: Phenotypes
layer3 = Layer(
    layer_name = "Phenotypes",
    data_path = pheno_path,
    header = true,
    missing_value = "NA"
)

# Equations
# Layer 1 -> Layer 2: Genotypes predict BOTH latent1 and omics1 (BayesC)
# This way, observed omics1 in test set contributes to L1→L2 marker effect estimation
eq1 = Equation(
    from_layer_name = "Genotypes",
    to_layer_name = "MiddleLayer",
    equation = "MiddleLayer = intercept + Genotypes",
    traits = ["latent1", "omics1"],  # BOTH are latent traits!
    method = "BayesC"
)

# Layer 2 -> Layer 3: Both latent1 and omics1 predict y (linear activation)
# NO covariate - omics1 is part of the hidden layer
eq2 = Equation(
    from_layer_name = "MiddleLayer",
    to_layer_name = "Phenotypes",
    equation = "Phenotypes = intercept + MiddleLayer",
    traits = ["y"],
    activation_function = "linear"
)

# Run NNMM
output_nnmm = joinpath(tmpdir, "nnmm_out")
out_nnmm = runNNMM(
    [layer1, layer2, layer3],
    [eq1, eq2],
    chain_length = 50000,
    burnin = 10000,
    output_folder = output_nnmm,
    seed = 12345
)

# Get NNMM predictions from result dictionary
println("NNMM output keys: $(keys(out_nnmm))")

# For comparison with a traditional model that uses observed omics (O),
# use EPV (uses observed middle-layer values), not EBV (uses genotype-predicted omics).
epv_nnmm_df = out_nnmm["EPV_Output_NonLinear"]
println("NNMM EPV columns: $(names(epv_nnmm_df))")
epv_nnmm = Dict(zip(epv_nnmm_df.ID, epv_nnmm_df.EPV))

# Also get EBV (purely genotype-predicted) for fair comparison
ebv_nnmm_df = out_nnmm["EBV_NonLinear"]
println("NNMM EBV columns: $(names(ebv_nnmm_df))")
ebv_nnmm = Dict(zip(ebv_nnmm_df.ID, ebv_nnmm_df.EBV))

# ============================================================
# Step 5: Run JWAS Multi-Class BayesC (Genotypes + Omics)
# ============================================================
println("\n--- Step 5: Running JWAS Multi-Class BayesC ---")
println("Model: y = X*a + O*b + e (two genomic classes)")

using JWAS

# Prepare JWAS phenotype data (training only has observed y)
y_jwas = Vector{Union{Float64, Missing}}(y_true)
y_jwas[test_idx] .= missing

jwas_pheno_df = DataFrame(
    ID = ids,
    y = y_jwas
)
jwas_pheno_path = joinpath(tmpdir, "jwas_pheno.csv")
CSV.write(jwas_pheno_path, jwas_pheno_df, missingstring="NA")

# Save omics as a "genotype-like" file for JWAS (second class)
omics_geno_path = joinpath(tmpdir, "omics_as_geno.csv")
omics_geno_df = DataFrame(ID = ids)
omics_geno_df[!, "omics1"] = O_true
CSV.write(omics_geno_path, omics_geno_df)

# Load genotypes as class 1
geno = JWAS.get_genotypes(geno_path, separator=',', header=true)

# Load omics as class 2 (like a second set of markers)
omics_class = JWAS.get_genotypes(omics_geno_path, separator=',', header=true, quality_control=false)
# For continuous omics covariates, JWAS's default `sum2pq` (based on allele frequencies)
# is not meaningful; replace it with the sum of column variances to keep BayesC priors finite.
omics_class.sum2pq = max(float(sum(var(omics_class.genotypes, dims=1))), eps(Float64))

# JWAS multi-class model: y = geno (class 1) + omics_class (class 2)
model_equation = "y = intercept + geno + omics_class"
mme_jwas = JWAS.build_model(model_equation)

# Read phenotypes
jwas_data = CSV.read(jwas_pheno_path, DataFrame, missingstring="NA")

# Run JWAS multi-class BayesC
output_jwas = joinpath(tmpdir, "jwas_out")
out_jwas = JWAS.runMCMC(
    mme_jwas,
    jwas_data,
    chain_length = 50000,
    burnin = 10000,
    output_folder = output_jwas,
    seed = 12345,
    outputEBV = true
)

# Get JWAS predictions (EBV_y for standard model)
println("JWAS output keys: $(keys(out_jwas))")
ebv_jwas_df = out_jwas["EBV_y"]
println("JWAS EBV columns: $(names(ebv_jwas_df))")
ebv_jwas = Dict(zip(ebv_jwas_df.ID, ebv_jwas_df.EBV))

# ============================================================
# Step 6: Compare Results
# ============================================================
println("\n--- Step 6: Comparing Results ---")

# Extract predictions for common IDs
common_ids = intersect(keys(epv_nnmm), keys(ebv_jwas))
println("Common IDs: $(length(common_ids))")

pred_nnmm = [epv_nnmm[id] for id in common_ids]
pred_jwas = [ebv_jwas[id] for id in common_ids]
true_vals = [y_true[findfirst(==(id), ids)] for id in common_ids]

# Overall correlation
r_overall = cor(pred_jwas, pred_nnmm)
println("\n=== Overall Results ===")
println("r(JWAS_NNMM, NNMM.jl) = $(round(r_overall, digits=4))")

# Training set
train_ids_set = Set(ids[train_idx])
train_common = [id for id in common_ids if id in train_ids_set]
pred_nnmm_train = [epv_nnmm[id] for id in train_common]
pred_jwas_train = [ebv_jwas[id] for id in train_common]
true_train = [y_true[findfirst(==(id), ids)] for id in train_common]

r_train = cor(pred_jwas_train, pred_nnmm_train)
acc_jwas_train = cor(pred_jwas_train, true_train)
acc_nnmm_train = cor(pred_nnmm_train, true_train)

println("\n=== Training Set (n=$(length(train_common))) ===")
println("r(JWAS, NNMM) = $(round(r_train, digits=4))")
println("Accuracy JWAS  = $(round(acc_jwas_train, digits=4))")
println("Accuracy NNMM  = $(round(acc_nnmm_train, digits=4))")

# Test set
test_ids_set = Set(ids[test_idx])
test_common = [id for id in common_ids if id in test_ids_set]
pred_nnmm_test = [epv_nnmm[id] for id in test_common]
pred_jwas_test = [ebv_jwas[id] for id in test_common]
true_test = [y_true[findfirst(==(id), ids)] for id in test_common]

r_test = cor(pred_jwas_test, pred_nnmm_test)
acc_jwas_test = cor(pred_jwas_test, true_test)
acc_nnmm_test = cor(pred_nnmm_test, true_test)

println("\n=== Test Set (n=$(length(test_common))) ===")
println("r(JWAS, NNMM) = $(round(r_test, digits=4))")
println("Accuracy JWAS  = $(round(acc_jwas_test, digits=4))")
println("Accuracy NNMM  = $(round(acc_nnmm_test, digits=4))")

# === EBV Comparison (purely genotype-based) ===
println("\n" * "="^60)
println("=== NNMM EBV (genotype-predicted) Comparison ===")
println("="^60)

# Training set - EBV
ebv_common_train = [id for id in train_common if haskey(ebv_nnmm, id)]
pred_ebv_train = [ebv_nnmm[id] for id in ebv_common_train]
pred_jwas_train2 = [ebv_jwas[id] for id in ebv_common_train]
true_train2 = [y_true[findfirst(==(id), ids)] for id in ebv_common_train]

r_ebv_train = cor(pred_jwas_train2, pred_ebv_train)
acc_ebv_train = cor(pred_ebv_train, true_train2)
println("\n=== Training Set (EBV) ===")
println("r(JWAS, NNMM_EBV) = $(round(r_ebv_train, digits=4))")
println("Accuracy NNMM (EBV) = $(round(acc_ebv_train, digits=4))")
println("Accuracy JWAS       = $(round(acc_jwas_train, digits=4))")

# Test set - EBV
ebv_common_test = [id for id in test_common if haskey(ebv_nnmm, id)]
pred_ebv_test = [ebv_nnmm[id] for id in ebv_common_test]
pred_jwas_test2 = [ebv_jwas[id] for id in ebv_common_test]
true_test2 = [y_true[findfirst(==(id), ids)] for id in ebv_common_test]

r_ebv_test = cor(pred_jwas_test2, pred_ebv_test)
acc_ebv_test = cor(pred_ebv_test, true_test2)
println("\n=== Test Set (EBV) ===")
println("r(JWAS, NNMM_EBV) = $(round(r_ebv_test, digits=4))")
println("Accuracy NNMM (EBV) = $(round(acc_ebv_test, digits=4))")
println("Accuracy JWAS       = $(round(acc_jwas_test, digits=4))")

# Summary
println("\n" * "="^60)
if r_train > 0.95 && r_test > 0.90
    println("✓ SUCCESS: JWAS NNMM ≈ NNMM.jl (with train/test split)")
else
    println("⚠ Results differ - investigate further")
end
println("="^60)

# Cleanup
println("\nCleaning up temp directory...")
rm(tmpdir, recursive=true)
println("Done!")

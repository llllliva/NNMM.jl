#!/usr/bin/env julia
#=
Test: Compare NNMM.jl vs JWAS Multi-Class BayesC
Goal: Make them as equivalent as possible

JWAS Multi-class: y = X*β_geno + O*β_omics + e
NNMM: Remove latent, use ONLY observed omics in L2
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra

println("="^60)
println("Test: NNMM vs JWAS Multi-Class (exact comparison)")
println("="^60)

Random.seed!(12345)

# Simpler data setup
n = 300
p = 50  # SNPs

# Generate genotypes
X = rand([0.0, 1.0, 2.0], n, p)

# Generate omics (1 omics feature, observed for all)
β_omics_from_geno = randn(p) .* 0.05
O_true = X * β_omics_from_geno + randn(n) .* 0.3

# Generate phenotype: y = X*β_geno + O*β_omics + e
β_geno = randn(p) .* 0.1  # direct genotype effects on y
β_omics = 0.5  # omics effect on y
y_true = X * β_geno + O_true .* β_omics + randn(n) .* 0.5

ids = ["ind_$i" for i in 1:n]

tmpdir = mktempdir()
println("Temp dir: $tmpdir")

# Save genotypes
geno_df = DataFrame(ID = ids)
for j in 1:p
    geno_df[!, "snp$j"] = X[:, j]
end
geno_path = joinpath(tmpdir, "genotypes.csv")
CSV.write(geno_path, geno_df)

# Save phenotypes
pheno_path = joinpath(tmpdir, "phenotypes.csv")
CSV.write(pheno_path, DataFrame(ID = ids, y = y_true))

# Save omics (ONLY observed omics, no latent)
omics_path = joinpath(tmpdir, "omics.csv")
CSV.write(omics_path, DataFrame(ID = ids, omics1 = O_true))

# ============================================================
# NNMM.jl: Only observed omics in L2 (no latent)
# ============================================================
println("\n--- Running NNMM.jl (only observed omics, no latent) ---")

using NNMM

layer1 = Layer(layer_name="Genotypes", data_path=geno_path, header=true)
layer2 = Layer(layer_name="MiddleLayer", data_path=omics_path, header=true, missing_value="NA")
layer3 = Layer(layer_name="Phenotypes", data_path=pheno_path, header=true, missing_value="NA")

# Only omics1 in L2 (no latent)
eq1 = Equation(
    from_layer_name="Genotypes", to_layer_name="MiddleLayer",
    equation="MiddleLayer = intercept + Genotypes",
    traits=["omics1"],  # ONLY observed omics
    method="BayesC"
)
eq2 = Equation(
    from_layer_name="MiddleLayer", to_layer_name="Phenotypes",
    equation="Phenotypes = intercept + MiddleLayer",
    traits=["y"],
    activation_function="linear"
)

output_nnmm = joinpath(tmpdir, "nnmm_out")
out_nnmm = runNNMM([layer1, layer2, layer3], [eq1, eq2],
    chain_length=5000, burnin=1000, output_folder=output_nnmm, seed=12345)

# Use EPV (observed omics values)
epv_nnmm_df = out_nnmm["EPV_Output_NonLinear"]
epv_nnmm = Dict(zip(epv_nnmm_df.ID, epv_nnmm_df.EPV))

# ============================================================
# JWAS Multi-Class BayesC: y = geno + omics_class
# ============================================================
println("\n--- Running JWAS Multi-Class BayesC ---")

using JWAS

# Save omics as "genotype" file for JWAS
omics_geno_path = joinpath(tmpdir, "omics_as_geno.csv")
CSV.write(omics_geno_path, DataFrame(ID = ids, omics1 = O_true))

# Load genotypes
geno = JWAS.get_genotypes(geno_path, separator=',', header=true)

# Load omics as second class with proper prior
omics_class = JWAS.get_genotypes(omics_geno_path, separator=',', header=true, quality_control=false)
omics_class.sum2pq = max(float(sum(var(omics_class.genotypes, dims=1))), eps(Float64))

# Multi-class model
mme_jwas = JWAS.build_model("y = intercept + geno + omics_class")
jwas_data = CSV.read(pheno_path, DataFrame)

output_jwas = joinpath(tmpdir, "jwas_out")
out_jwas = JWAS.runMCMC(mme_jwas, jwas_data,
    chain_length=5000, burnin=1000, output_folder=output_jwas, seed=12345, outputEBV=true)

ebv_jwas_df = out_jwas["EBV_y"]
ebv_jwas = Dict(zip(ebv_jwas_df.ID, ebv_jwas_df.EBV))

# ============================================================
# Compare
# ============================================================
println("\n--- Comparing Results ---")

common_ids = collect(intersect(keys(epv_nnmm), keys(ebv_jwas)))
pred_nnmm = [epv_nnmm[id] for id in common_ids]
pred_jwas = [ebv_jwas[id] for id in common_ids]
true_vals = [y_true[findfirst(==(id), ids)] for id in common_ids]

r_models = cor(pred_jwas, pred_nnmm)
acc_nnmm = cor(pred_nnmm, true_vals)
acc_jwas = cor(pred_jwas, true_vals)

println("\n=== Results ===")
println("r(JWAS, NNMM) = $(round(r_models, digits=4))")
println("Accuracy NNMM = $(round(acc_nnmm, digits=4))")
println("Accuracy JWAS = $(round(acc_jwas, digits=4))")

# Check variance estimates
println("\n=== Model Structure Comparison ===")
println("NNMM: X → [omics1] → y (hierarchical)")
println("JWAS: X + O → y directly (flat)")
println("\nKey difference: In NNMM, genotype effects go through omics layer.")
println("In JWAS, both X and O directly predict y.")

if r_models > 0.99
    println("\n✓ Models are essentially equivalent!")
elseif r_models > 0.95
    println("\n≈ Models are similar but structurally different")
else
    println("\n✗ Models give different predictions")
end

rm(tmpdir, recursive=true)
println("\nDone!")

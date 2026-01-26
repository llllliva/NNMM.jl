#!/usr/bin/env julia
#=
Test: Compare NNMM.jl with JWAS Multi-Class BayesC
WITHOUT Train/Test Split - all individuals have observed phenotypes

NNMM.jl: X → [latent1, omics1] → y
JWAS: y = X*a + O*b + e (two genomic classes)
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra

println("="^60)
println("Test: NNMM vs JWAS Multi-Class BayesC (NO train/test split)")
println("="^60)

Random.seed!(12345)

# Create simulated data
n = 300
p = 50  # SNPs

X = rand([0.0, 1.0, 2.0], n, p)
β_latent = randn(p) .* 0.1
L_true = X * β_latent

β_omics = randn(p) .* 0.05
O_true = X * β_omics + randn(n) .* 0.3

w_latent = 0.6
w_omics = 0.4
y_true = w_latent .* L_true + w_omics .* O_true + randn(n) .* 0.5

ids = ["ind_$i" for i in 1:n]

tmpdir = mktempdir()
println("Using temp directory: $tmpdir")

# Save data files
geno_df = DataFrame(ID = ids)
for j in 1:p
    geno_df[!, "snp$j"] = X[:, j]
end
geno_path = joinpath(tmpdir, "genotypes.csv")
CSV.write(geno_path, geno_df)

pheno_df = DataFrame(ID = ids, y = y_true)
pheno_path = joinpath(tmpdir, "phenotypes.csv")
CSV.write(pheno_path, pheno_df)

# Latent + Omics
latent_omics_path = joinpath(tmpdir, "latent_omics.csv")
open(latent_omics_path, "w") do io
    println(io, "ID,latent1,omics1")
    for i in 1:n
        # latent1: all missing (will be sampled)
        # omics1: all observed
        println(io, "$(ids[i]),NA,$(O_true[i])")
    end
end

# ============================================================
# Run NNMM.jl
# ============================================================
println("\n--- Running NNMM.jl ---")

using NNMM

layer1 = Layer(layer_name="Genotypes", data_path=geno_path, header=true)
layer2 = Layer(layer_name="MiddleLayer", data_path=latent_omics_path, header=true, missing_value="NA")
layer3 = Layer(layer_name="Phenotypes", data_path=pheno_path, header=true, missing_value="NA")

eq1 = Equation(
    from_layer_name="Genotypes", to_layer_name="MiddleLayer",
    equation="MiddleLayer = intercept + Genotypes",
    traits=["latent1", "omics1"],
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

# For comparison with a traditional model that uses observed omics (O),
# use EPV (uses observed middle-layer values), not EBV (uses genotype-predicted omics).
epv_nnmm_df = out_nnmm["EPV_Output_NonLinear"]
epv_nnmm = Dict(zip(epv_nnmm_df.ID, epv_nnmm_df.EPV))

# ============================================================
# Run JWAS: y = Xβ + Oω + e (genotypes + omics as covariate)
# ============================================================
println("\n--- Running JWAS: y = geno + omics1 (covariate) ---")

using JWAS

# Prepare phenotype data with omics as a column
jwas_pheno_df = DataFrame(ID = ids, y = y_true, omics1 = O_true)
jwas_pheno_path = joinpath(tmpdir, "jwas_pheno.csv")
CSV.write(jwas_pheno_path, jwas_pheno_df)

# Load genotypes
geno = JWAS.get_genotypes(geno_path, separator=',', header=true)

# Model: y = intercept + omics1 (covariate) + geno (marker effects)
mme_jwas = JWAS.build_model("y = intercept + omics1 + geno")
JWAS.set_covariate(mme_jwas, "omics1")

jwas_data = CSV.read(jwas_pheno_path, DataFrame)

output_jwas = joinpath(tmpdir, "jwas_out")
out_jwas = JWAS.runMCMC(mme_jwas, jwas_data,
    chain_length=5000, burnin=1000, output_folder=output_jwas, seed=12345, outputEBV=true)

ebv_jwas_df = out_jwas["EBV_y"]
ebv_jwas = Dict(zip(ebv_jwas_df.ID, ebv_jwas_df.EBV))

# ============================================================
# Compare Results
# ============================================================
println("\n--- Comparing Results ---")

common_ids = intersect(keys(epv_nnmm), keys(ebv_jwas))
pred_nnmm = [epv_nnmm[id] for id in common_ids]
pred_jwas = [ebv_jwas[id] for id in common_ids]
true_vals = [y_true[findfirst(==(id), ids)] for id in common_ids]

r_models = cor(pred_jwas, pred_nnmm)
acc_nnmm = cor(pred_nnmm, true_vals)
acc_jwas = cor(pred_jwas, true_vals)

println("\n=== Results ===")
println("r(JWAS, NNMM) = $(round(r_models, digits=4))")
println("Accuracy NNMM  = $(round(acc_nnmm, digits=4))")
println("Accuracy JWAS  = $(round(acc_jwas, digits=4))")

if r_models > 0.90
    println("\n✓ SUCCESS: JWAS Multi-Class ≈ NNMM.jl")
else
    println("\n⚠ Models differ significantly")
end

rm(tmpdir, recursive=true)
println("\nDone!")

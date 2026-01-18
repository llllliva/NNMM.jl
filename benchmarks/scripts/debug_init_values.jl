#!/usr/bin/env julia
#=
Debug script to print initial values used in NNMM.jl MCMC.
Compare these with PyNNMM initial values to identify differences.
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random
using DelimitedFiles

const SEED = 42
Random.seed!(SEED)

# Load simulated dataset
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

println("=" ^ 70)
println("NNMM.jl Initialization Values Debug")
println("=" ^ 70)

# Read genotypes directly
geno_data = readdlm(geno_path, ',')
geno_ids = geno_data[2:end, 1]
geno_matrix = Float64.(geno_data[2:end, 2:end])

println("\n1. GENOTYPE DATA (raw)")
println("   Shape: $(size(geno_matrix))")
println("   Mean (before centering): $(round(mean(geno_matrix), digits=6))")
println("   Std: $(round(std(geno_matrix), digits=6))")

# Center genotypes (like Julia does)
geno_means = mean(geno_matrix, dims=1)
geno_centered = geno_matrix .- geno_means
p = geno_means ./ 2.0
sum2pq = sum(2 .* p .* (1 .- p))

println("\n   After centering:")
println("   Mean: $(round(mean(geno_centered), digits=6))")
println("   Std: $(round(std(geno_centered), digits=6))")
println("   First 5x5 values:")
display(round.(geno_centered[1:5, 1:5], digits=4))

println("\n5. ALLELE FREQUENCY")
println("   Sum(2pq): $(round(sum2pq, digits=4))")

# Read omics data
omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
omics_matrix = Matrix(omics_df[:, 2:end])

println("\n2. OMICS DATA (original)")
println("   Shape: $(size(omics_matrix))")
omics_means = mean(omics_matrix, dims=1)
omics_stds = std(omics_matrix, dims=1)
println("   Mean per trait: $(round.(vec(omics_means), digits=6))")
println("   Std per trait: $(round.(vec(omics_stds), digits=6))")
println("   Overall mean: $(round(mean(omics_matrix), digits=6))")

# Intercepts = column means
intercepts1 = vec(omics_means)
println("\n3. INITIAL INTERCEPTS (Layer 1)")
println("   Values: $(round.(intercepts1, digits=6))")

# ycorr1 = omics - intercepts
ycorr1 = omics_matrix .- omics_means
println("\n4. INITIAL ycorr1 (omics - intercepts)")
println("   Mean per trait: $(round.(vec(mean(ycorr1, dims=1)), digits=10))")
println("   Std per trait: $(round.(vec(std(ycorr1, dims=1)), digits=6))")

# Variance initialization
omics_var = vec(var(omics_matrix, dims=1))
genetic_var_omics = omics_var .* 0.5
println("\n6. VARIANCE INITIALIZATION (Layer 1)")
println("   Omics variance: $(round.(omics_var, digits=6))")
println("   Genetic variance (h2=0.5): $(round.(genetic_var_omics, digits=6))")

# Pi and var_effect
estimate_pi = true
pi1_value = estimate_pi ? 0.001 : 0.95
one_minus_pi = max(1.0 - pi1_value, 0.001)
var_effect1 = [gv / (one_minus_pi * sum2pq) for gv in genetic_var_omics]
vare1 = omics_var .* 0.5

println("\n7. MARKER EFFECT VARIANCE (var_effect1)")
println("   Pi = $pi1_value")
println("   Formula: genetic_var / ((1-pi) * sum2pq)")
println("   Values: $(round.(var_effect1, digits=8))")
println("   Std of marker effects: $(round.(sqrt.(var_effect1), digits=8))")

println("\n8. RESIDUAL VARIANCE (vare1)")
println("   Values: $(round.(vare1, digits=6))")

# Scale parameters
prior_df = 4.0
scale1 = [v * (prior_df - 2.0) / prior_df for v in var_effect1]
scale_vare1 = [v * (prior_df - 2.0) / prior_df for v in vare1]

println("\n9. SCALE PARAMETERS (Layer 1)")
println("   scale1 (marker): $(round.(scale1, digits=8))")
println("   scale_vare1 (residual): $(round.(scale_vare1, digits=6))")

# Layer 2 - phenotype data
pheno_trait = pheno_df.trait1
yobs_obs = pheno_trait[.!ismissing.(pheno_trait)]
yobs_obs = Float64.(yobs_obs)
intercept2 = mean(yobs_obs)
ycorr2 = yobs_obs .- intercept2

println("\n10. PHENOTYPE DATA")
println("   Observed: $(length(yobs_obs))")
println("   Mean: $(round(mean(yobs_obs), digits=6))")
println("   Std: $(round(std(yobs_obs), digits=6))")
println("   Initial intercept2: $(round(intercept2, digits=6))")

pheno_var = var(yobs_obs)
genetic_var_pheno = pheno_var * 0.5
vare2 = pheno_var * 0.5
n_traits = 10
pi2 = estimate_pi ? 0.001 : 0.95
one_minus_pi2 = max(1.0 - pi2, 0.001)
var_effect2 = genetic_var_pheno / (one_minus_pi2 * n_traits)

println("\n11. VARIANCE INITIALIZATION (Layer 2)")
println("   Phenotype variance: $(round(pheno_var, digits=6))")
println("   Genetic variance: $(round(genetic_var_pheno, digits=6))")
println("   Residual variance: $(round(vare2, digits=6))")
println("   var_effect2 (omics effects): $(round(var_effect2, digits=6))")
println("   Std of omics effects: $(round(sqrt(var_effect2), digits=6))")

scale2 = var_effect2 * (prior_df - 2.0) / prior_df
scale_vare2 = vare2 * (prior_df - 2.0) / prior_df

println("\n12. SCALE PARAMETERS (Layer 2)")
println("   scale2 (omics effect): $(round(scale2, digits=6))")
println("   scale_vare2 (residual): $(round(scale_vare2, digits=6))")



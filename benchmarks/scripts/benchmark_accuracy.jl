#!/usr/bin/env julia
#=
NNMM Benchmark Script for PyNNMM Parity Testing

This script runs NNMM with the simulated_omics_data dataset and outputs
accuracy metrics that can be used to validate the PyNNMM implementation.

Usage:
    julia --project=. benchmarks/benchmark_accuracy.jl

Output:
    - Prints accuracy metrics to console
    - Can be extended to save results to files
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random
using Dates

println("=" ^ 70)
println("NNMM Benchmark for PyNNMM Parity Testing")
println("=" ^ 70)
println("Date: ", Dates.now())
println()

# Configuration - these parameters work reliably with seed=42
const SEED = 42
const CHAIN_LENGTH = 1000
const BURNIN = 200

Random.seed!(SEED)

# Load simulated dataset
println("Loading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

println("  Individuals: ", nrow(pheno_df))
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: 10")
println()

# Setup temp directory
benchmark_dir = mktempdir()

# Create data files for NNMM
omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
omics_path = joinpath(benchmark_dir, "omics.csv")
CSV.write(omics_path, omics_df; missingstring="NA")

pheno_out_df = pheno_df[:, [:ID, :trait1]]
pheno_out_path = joinpath(benchmark_dir, "phenotypes.csv")
CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")

# Define layers
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
    Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
]

# Define equations
equations = [
    Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = intercept + geno",
        omics_name=["omic$i" for i in 1:10],
        method="BayesC",
        estimatePi=true
    ),
    Equation(
        from_layer_name="omics",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + omics",
        phenotype_name=["trait1"],
        method="BayesC",
        activation_function="linear"
    )
]

# Run NNMM
println("Running MCMC (chain_length=$CHAIN_LENGTH, burnin=$BURNIN, seed=$SEED)...")
result = runNNMM(layers, equations;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=joinpath(benchmark_dir, "output")
)

# Get EBV results
ebv_df = result["EBV_NonLinear"]
ebv_df.ID = string.(ebv_df.ID)
pheno_df.ID = string.(pheno_df.ID)

# Get EPV results
epv_df = result["EPV_NonLinear"]
epv_df.ID = string.(epv_df.ID)

# Merge with true breeding values
merged_df = innerjoin(ebv_df, pheno_df[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect, :trait1]], on=:ID)
merged_epv = innerjoin(epv_df, pheno_df[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect, :trait1]], on=:ID)

# Calculate EBV accuracy metrics
ebv_accuracy_total = cor(merged_df.EBV, merged_df.genetic_total)
ebv_accuracy_direct = cor(merged_df.EBV, merged_df.genetic_direct)
ebv_accuracy_indirect = cor(merged_df.EBV, merged_df.genetic_indirect)

# Calculate EPV accuracy metrics  
# Note: EPV_NonLinear DataFrame has column named "EPV" (same as EBV DataFrame has "EBV")
epv_col = hasproperty(merged_epv, :EPV) ? merged_epv.EPV : merged_epv.EBV
epv_accuracy_total = cor(epv_col, merged_epv.genetic_total)
epv_accuracy_direct = cor(epv_col, merged_epv.genetic_direct)
epv_accuracy_indirect = cor(epv_col, merged_epv.genetic_indirect)
epv_accuracy_trait = cor(epv_col, merged_epv.trait1)

println()
println("=" ^ 70)
println("BENCHMARK RESULTS FOR PYNNMM COMPARISON")
println("=" ^ 70)
println()
println("Configuration:")
println("  Seed: $SEED")
println("  Chain length: $CHAIN_LENGTH")
println("  Burnin: $BURNIN")
println("  Method: BayesC (both layers)")
println("  Activation: linear")
println()
println("Dataset (simulated_omics_data):")
println("  Individuals: $(nrow(pheno_df))")
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: 10")
println("  Target heritability: 0.5 (20% direct, 80% indirect)")
println()
println("EBV ACCURACY METRICS (Estimated Breeding Value - from predicted omics):")
println("  ┌────────────────────────────────────────┐")
println("  │ cor(EBV, genetic_total):    $(round(ebv_accuracy_total, digits=4))  │")
println("  │ cor(EBV, genetic_direct):   $(round(ebv_accuracy_direct, digits=4))  │")
println("  │ cor(EBV, genetic_indirect): $(round(ebv_accuracy_indirect, digits=4))  │")
println("  └────────────────────────────────────────┘")
println()
println("EPV ACCURACY METRICS (Estimated Phenotypic Value - from observed omics):")
println("  ┌────────────────────────────────────────┐")
println("  │ cor(EPV, genetic_total):    $(round(epv_accuracy_total, digits=4))  │")
println("  │ cor(EPV, genetic_direct):   $(round(epv_accuracy_direct, digits=4))  │")
println("  │ cor(EPV, genetic_indirect): $(round(epv_accuracy_indirect, digits=4))  │")
println("  │ cor(EPV, trait1):           $(round(epv_accuracy_trait, digits=4))  │")
println("  └────────────────────────────────────────┘")
println()
println("EBV Statistics:")
println("  Mean: $(round(mean(merged_df.EBV), digits=4))")
println("  Std:  $(round(std(merged_df.EBV), digits=4))")
println()
println("EPV Statistics:")
println("  Mean: $(round(mean(epv_col), digits=4))")
println("  Std:  $(round(std(epv_col), digits=4))")
println()
println("=" ^ 70)
println()
println("Expected PyNNMM accuracy should be within ±0.05 of these values")
println("with the same configuration and random seed.")
println()

# Cleanup
rm(benchmark_dir, recursive=true)

#!/usr/bin/env julia
#=
Debug script to trace variance evolution in NNMM.jl.
Check if Layer 2 variance is growing.
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random
using DelimitedFiles

const SEED = 42
const CHAIN_LENGTH = 100  # Short chain for debugging
const BURNIN = 0

Random.seed!(SEED)

# Load simulated dataset
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

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

println("=" ^ 70)
println("Debug: Tracing Variance Evolution in NNMM.jl")
println("=" ^ 70)

# Run NNMM
output_folder = joinpath(benchmark_dir, "output")
result = runNNMM(layers, equations;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=output_folder
)

# Read EBV samples
ebv_file = joinpath(output_folder, "MCMC_samples_EBV_NonLinear.txt")
if isfile(ebv_file)
    println("\nReading EBV samples from $ebv_file")
    ebv_data, ids = readdlm(ebv_file, ',', header=true)
    println("EBV samples shape: $(size(ebv_data))")
    
    println("\nEBV statistics at different iterations:")
    println(rpad("Iter", 6), rpad("Mean", 12), rpad("Std", 12), rpad("Min", 12), rpad("Max", 12))
    for i in [1, 10, 50, 100]
        if i <= size(ebv_data, 1)
            row = ebv_data[i, :]
            println(rpad(string(i), 6), 
                    rpad(string(round(mean(row), digits=4)), 12),
                    rpad(string(round(std(row), digits=4)), 12),
                    rpad(string(round(minimum(row), digits=4)), 12),
                    rpad(string(round(maximum(row), digits=4)), 12))
        end
    end
else
    println("Warning: $ebv_file not found")
end

# Read Layer 2 variances
var_file = joinpath(output_folder, "MCMC_samples_layer2_effect_variance.txt")
if isfile(var_file)
    println("\nReading Layer 2 effect variance samples from $var_file")
    var_data = readdlm(var_file, ',')
    println("Shape: $(size(var_data))")
    
    println("\nLayer 2 effect variance at different iterations:")
    println(rpad("Iter", 6), rpad("Value", 12))
    for i in [1, 10, 50, 100]
        if i <= size(var_data, 1)
            println(rpad(string(i), 6), rpad(string(round(var_data[i, 1], digits=6)), 12))
        end
    end
else
    println("Warning: $var_file not found")
end

# Cleanup
rm(benchmark_dir, recursive=true)



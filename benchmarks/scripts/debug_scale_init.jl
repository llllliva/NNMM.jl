#!/usr/bin/env julia
#=
Debug: Check what scale values are initialized for Layer 2 omics
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

Random.seed!(42)

# Load data
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

# Setup
benchmark_dir = mktempdir()

omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
omics_path = joinpath(benchmark_dir, "omics.csv")
CSV.write(omics_path, omics_df; missingstring="NA")

pheno_out_df = pheno_df[:, [:ID, :trait1]]
pheno_out_path = joinpath(benchmark_dir, "phenotypes.csv")
CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")

# Define model
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
    Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
]

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
println("Debug: Checking Scale Initialization")
println("=" ^ 70)

# Run with very short chain just to see initialization
output_folder = joinpath(benchmark_dir, "output")
result = runNNMM(layers, equations;
    chain_length=2,
    burnin=0,
    printout_frequency=100,
    output_folder=output_folder
)

# Clean up
rm(benchmark_dir, recursive=true)

println("\nDone! Check the printed output above for variance/scale values.")



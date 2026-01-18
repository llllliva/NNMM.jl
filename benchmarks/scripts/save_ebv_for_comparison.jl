#!/usr/bin/env julia
#=
Save EBVs from NNMM.jl for cross-package comparison with PyNNMM

Usage:
    julia --project=. benchmarks/save_ebv_for_comparison.jl
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

# Configuration - matches PyNNMM benchmark
function get_arg(flag, default)
    for i in 1:length(ARGS)
        arg = ARGS[i]
        if arg == flag && i < length(ARGS)
            return ARGS[i + 1]
        elseif startswith(arg, flag * "=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end

function parse_bool(val)
    sval = lowercase(String(val))
    if sval in ["1", "true", "t", "yes", "y"]
        return true
    elseif sval in ["0", "false", "f", "no", "n"]
        return false
    end
    error("Invalid boolean value: $val")
end

const SEED = parse(Int, get_arg("--seed", "42"))
const CHAIN_LENGTH = parse(Int, get_arg("--chain-length", "1000"))
const BURNIN = parse(Int, get_arg("--burnin", "200"))
const ESTIMATE_PI = parse_bool(get_arg("--estimate-pi", "true"))
const ESTIMATE_VAR1 = parse_bool(get_arg("--estimate-var1", "true"))
const ESTIMATE_VAR2 = parse_bool(get_arg("--estimate-var2", "false"))
const DOUBLE_PRECISION = parse_bool(get_arg("--double-precision", "false"))
const OUTPUT_SUFFIX = get_arg("--suffix", "")
const PI_VALUE = 0.0

println("=" ^ 70)
println("NNMM.jl - Saving EBVs for Cross-Package Comparison")
println("=" ^ 70)
println("Config: seed=$SEED chain_length=$CHAIN_LENGTH burnin=$BURNIN")
println("        estimate_pi=$ESTIMATE_PI estimate_var1=$ESTIMATE_VAR1 estimate_var2=$ESTIMATE_VAR2")
println("        double_precision=$DOUBLE_PRECISION suffix=$(isempty(OUTPUT_SUFFIX) ? "(none)" : OUTPUT_SUFFIX)")
println()

Random.seed!(SEED)

# Load simulated dataset
println("Loading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

println("  Individuals: ", nrow(pheno_df))
println("  SNPs: 1000")
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
        Pi=PI_VALUE,
        estimatePi=ESTIMATE_PI,
        estimate_variance_G=ESTIMATE_VAR1
    ),
    Equation(
        from_layer_name="omics",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + omics",
        phenotype_name=["trait1"],
        method="BayesC",
        activation_function="linear",
        Pi=PI_VALUE,
        estimatePi=ESTIMATE_PI,
        estimate_variance_G=ESTIMATE_VAR2
    )
]

# Run NNMM
println("Running MCMC (chain_length=$CHAIN_LENGTH, burnin=$BURNIN, seed=$SEED)...")
result = runNNMM(layers, equations;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    double_precision=DOUBLE_PRECISION,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=joinpath(benchmark_dir, "output")
)

# Get EBV results
ebv_df = result["EBV_NonLinear"]
ebv_df.ID = string.(ebv_df.ID)

# Get EPV results
epv_df = result["EPV_NonLinear"]
epv_df.ID = string.(epv_df.ID)

# Save EBVs to shared location
suffix_part = isempty(OUTPUT_SUFFIX) ? "" : "_" * OUTPUT_SUFFIX
ebv_output_path = joinpath(@__DIR__, "ebv_julia$(suffix_part).csv")
CSV.write(ebv_output_path, ebv_df)
println()
println("✅ EBVs saved to: $ebv_output_path")
println("   Individuals: $(nrow(ebv_df))")
println("   Mean EBV: $(round(mean(ebv_df.EBV), digits=4))")
println("   Std EBV:  $(round(std(ebv_df.EBV), digits=4))")

# Save EPVs to shared location
epv_output_path = joinpath(@__DIR__, "epv_julia$(suffix_part).csv")
CSV.write(epv_output_path, epv_df)
println()
println("✅ EPVs saved to: $epv_output_path")
println("   Individuals: $(nrow(epv_df))")
epv_col = hasproperty(epv_df, :EPV) ? epv_df.EPV : epv_df.EBV
println("   Mean EPV: $(round(mean(epv_col), digits=4))")
println("   Std EPV:  $(round(std(epv_col), digits=4))")

# Cleanup temp directory
rm(benchmark_dir, recursive=true)

println()
println("Now run PyNNMM benchmark and then compare_ebv.py")

#!/usr/bin/env julia
#=
NNMM Benchmark Script with Missing Omics Data

This script tests NNMM's ability to handle missing intermediate layer values,
which is one of its key features. Missing omics are imputed using HMC sampling.

Usage:
    julia --project=. benchmarks/benchmark_missing_omics.jl

Output:
    - Prints accuracy metrics with various missing data percentages
    - Compares performance with full vs missing omics data
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random
using Dates
using StatsBase  # For sample()

println("=" ^ 70)
println("NNMM Benchmark with Missing Omics Data")
println("=" ^ 70)
println("Date: ", Dates.now())
println()

# Configuration (override via CLI args)
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

function parse_csv_floats(val)
    parts = split(String(val), ",")
    return [parse(Float64, strip(p)) for p in parts if !isempty(strip(p))]
end

const SEED = parse(Int, get_arg("--seed", "42"))
const CHAIN_LENGTH = parse(Int, get_arg("--chain-length", "1000"))
const BURNIN = parse(Int, get_arg("--burnin", "200"))
const MISSING_PERCENTAGES = parse_csv_floats(get_arg("--missing-pcts", "0.0,0.3,0.5"))
# Missingness mode:
# - individual (default): select a fraction of individuals and set ALL omics missing for them
# - cell: original behavior (per-column missing); note this makes almost all individuals incomplete for large pcts
const MISSING_MODE = lowercase(String(get_arg("--missing-mode", "individual")))

# Load simulated dataset
println("Loading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

n_individuals = nrow(pheno_df)
n_omics = 10

println("  Individuals: ", n_individuals)
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: ", n_omics)
println()

# Store results
results = DataFrame(
    missing_pct = Float64[],
    n_missing_cells = Int[],
    ebv_accuracy_total = Float64[],
    ebv_accuracy_direct = Float64[],
    ebv_accuracy_indirect = Float64[],
    epv_accuracy_total = Float64[],
    epv_accuracy_direct = Float64[],
    epv_accuracy_indirect = Float64[],
    epv_accuracy_trait = Float64[]
)

for missing_pct in MISSING_PERCENTAGES
    println("-" ^ 70)
    println("Running with $(Int(missing_pct * 100))% missing omics data...")
    println("-" ^ 70)
    
    Random.seed!(SEED)
    
    # Setup temp directory
    benchmark_dir = mktempdir()
    
    # Create omics data with missing values
    omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:n_omics])
    omics_df = copy(pheno_df[:, omics_cols])
    
	    # Introduce missing values randomly
	    n_missing_cells = 0
	    if missing_pct > 0
	        allowmissing!(omics_df, Not(:ID))

	        n_to_miss = round(Int, n_individuals * missing_pct)
	        if MISSING_MODE == "cell"
	            for col in [Symbol("omic$i") for i in 1:n_omics]
	                miss_idx = sample(1:n_individuals, n_to_miss, replace=false)
	                omics_df[miss_idx, col] .= missing
	                n_missing_cells += n_to_miss
	            end
	            println("  Missing mode: cell (per-column)")
	            println("  Introduced $n_missing_cells missing cells ($(n_omics) omics × $n_to_miss individuals per omic)")
	        else
	            miss_idx = sample(1:n_individuals, n_to_miss, replace=false)
	            for col in [Symbol("omic$i") for i in 1:n_omics]
	                omics_df[miss_idx, col] .= missing
	            end
	            n_missing_cells = n_to_miss * n_omics
	            println("  Missing mode: individual (whole-omics missing)")
	            println("  Introduced $n_missing_cells missing cells ($(n_omics) omics × $n_to_miss individuals)")
	        end
	    else
	        println("  No missing data (baseline)")
	    end
    
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
            omics_name=["omic$i" for i in 1:n_omics],
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
	    println("  Running MCMC (chain_length=$CHAIN_LENGTH, burnin=$BURNIN)...")
    result = runNNMM(layers, equations;
        chain_length=CHAIN_LENGTH,
        burnin=BURNIN,
        printout_frequency=CHAIN_LENGTH + 1,
        output_folder=joinpath(benchmark_dir, "output")
    )
    
    # Get EBV results
    ebv_df = result["EBV_NonLinear"]
    ebv_df.ID = string.(ebv_df.ID)
    
    # Get EPV results
    epv_df = result["EPV_NonLinear"]
    epv_df.ID = string.(epv_df.ID)
    
    pheno_df_copy = copy(pheno_df)
    pheno_df_copy.ID = string.(pheno_df_copy.ID)
    
    # Merge with true breeding values
    merged_df = innerjoin(ebv_df, pheno_df_copy[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect, :trait1]], on=:ID)
    merged_epv = innerjoin(epv_df, pheno_df_copy[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect, :trait1]], on=:ID)
    
    # Calculate EBV accuracy metrics
    ebv_accuracy_total = cor(merged_df.EBV, merged_df.genetic_total)
    ebv_accuracy_direct = cor(merged_df.EBV, merged_df.genetic_direct)
    ebv_accuracy_indirect = cor(merged_df.EBV, merged_df.genetic_indirect)
    
    # Calculate EPV accuracy metrics
    epv_col = hasproperty(merged_epv, :EPV) ? merged_epv.EPV : merged_epv.EBV
    epv_accuracy_total = cor(epv_col, merged_epv.genetic_total)
    epv_accuracy_direct = cor(epv_col, merged_epv.genetic_direct)
    epv_accuracy_indirect = cor(epv_col, merged_epv.genetic_indirect)
    epv_accuracy_trait = cor(epv_col, merged_epv.trait1)
    
    push!(results, (missing_pct, n_missing_cells, 
        ebv_accuracy_total, ebv_accuracy_direct, ebv_accuracy_indirect,
        epv_accuracy_total, epv_accuracy_direct, epv_accuracy_indirect, epv_accuracy_trait))
    
    println("  EBV Accuracy: cor(EBV, genetic_total) = $(round(ebv_accuracy_total, digits=4))")
    println("  EPV Accuracy: cor(EPV, genetic_total) = $(round(epv_accuracy_total, digits=4))")
    println()
    
    # Cleanup
    rm(benchmark_dir, recursive=true)
end

# Print summary
println()
println("=" ^ 70)
println("BENCHMARK RESULTS SUMMARY - MISSING OMICS DATA")
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
println("  Individuals: $n_individuals")
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: $n_omics")
println("  Target heritability: 0.5 (20% direct, 80% indirect)")
println()
println("EBV ACCURACY METRICS BY MISSING PERCENTAGE:")
println()
println("┌─────────────┬───────────────┬─────────────────┬────────────────┬──────────────────┐")
println("│ Missing %   │ Missing Cells │ cor(EBV,total)  │ cor(EBV,direct)│ cor(EBV,indirect)│")
println("├─────────────┼───────────────┼─────────────────┼────────────────┼──────────────────┤")
for row in eachrow(results)
    pct_str = lpad("$(Int(row.missing_pct * 100))%", 5)
    cells_str = lpad(string(row.n_missing_cells), 10)
    total_str = lpad(string(round(row.ebv_accuracy_total, digits=4)), 10)
    direct_str = lpad(string(round(row.ebv_accuracy_direct, digits=4)), 10)
    indirect_str = lpad(string(round(row.ebv_accuracy_indirect, digits=4)), 10)
    println("│ $pct_str       │ $cells_str    │ $total_str      │ $direct_str     │ $indirect_str       │")
end
println("└─────────────┴───────────────┴─────────────────┴────────────────┴──────────────────┘")
println()

println("EPV ACCURACY METRICS BY MISSING PERCENTAGE:")
println()
println("┌─────────────┬───────────────┬─────────────────┬────────────────┬──────────────────┬────────────────┐")
println("│ Missing %   │ Missing Cells │ cor(EPV,total)  │ cor(EPV,direct)│ cor(EPV,indirect)│ cor(EPV,trait) │")
println("├─────────────┼───────────────┼─────────────────┼────────────────┼──────────────────┼────────────────┤")
for row in eachrow(results)
    pct_str = lpad("$(Int(row.missing_pct * 100))%", 5)
    cells_str = lpad(string(row.n_missing_cells), 10)
    total_str = lpad(string(round(row.epv_accuracy_total, digits=4)), 10)
    direct_str = lpad(string(round(row.epv_accuracy_direct, digits=4)), 10)
    indirect_str = lpad(string(round(row.epv_accuracy_indirect, digits=4)), 10)
    trait_str = lpad(string(round(row.epv_accuracy_trait, digits=4)), 10)
    println("│ $pct_str       │ $cells_str    │ $total_str      │ $direct_str     │ $indirect_str       │ $trait_str     │")
end
println("└─────────────┴───────────────┴─────────────────┴────────────────┴──────────────────┴────────────────┘")
println()

# Calculate degradation
if nrow(results) > 1
    baseline_ebv = results[1, :ebv_accuracy_total]
    baseline_epv = results[1, :epv_accuracy_total]
    println("EBV Accuracy degradation from baseline (0% missing):")
    for row in eachrow(results)
        if row.missing_pct > 0
            degradation = (baseline_ebv - row.ebv_accuracy_total) / baseline_ebv * 100
            println("  $(Int(row.missing_pct * 100))% missing: $(round(degradation, digits=2))% reduction")
        end
    end
    println()
    println("EPV Accuracy degradation from baseline (0% missing):")
    for row in eachrow(results)
        if row.missing_pct > 0
            degradation = (baseline_epv - row.epv_accuracy_total) / baseline_epv * 100
            println("  $(Int(row.missing_pct * 100))% missing: $(round(degradation, digits=2))% reduction")
        end
    end
    println()
end

println("=" ^ 70)
println()
println("PyNNMM should show similar accuracy patterns with missing data.")
println("Key: NNMM uses HMC to impute missing omics values during MCMC.")
println()

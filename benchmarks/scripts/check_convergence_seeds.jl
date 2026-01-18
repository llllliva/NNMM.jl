#!/usr/bin/env julia
"""
Check MCMC Convergence Across Different Seeds
==============================================

Runs NNMM benchmark 5 times with different seeds to confirm convergence.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NNMM
using NNMM.Datasets
using CSV
using DataFrames
using Statistics
using Random
using Dates

println("=" ^ 70)
println("NNMM.jl Convergence Check - Multiple Seeds")
println("=" ^ 70)
println("Date: ", Dates.now())
println()

# Configuration
const CHAIN_LENGTH = 1000
const BURNIN = 200
const SEEDS = [42, 123, 456, 789, 2024]

# Load data paths
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

println("Configuration:")
println("  Chain length: $CHAIN_LENGTH")
println("  Burnin: $BURNIN")
println("  Seeds: $SEEDS")
println()
println("Dataset:")
println("  Individuals: $(nrow(pheno_df))")
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: 10")
println()

# Store results
results = DataFrame(
    seed = Int[],
    ebv_accuracy_total = Float64[],
    ebv_accuracy_direct = Float64[],
    ebv_accuracy_indirect = Float64[],
    epv_accuracy_total = Float64[],
    epv_accuracy_direct = Float64[],
    epv_accuracy_indirect = Float64[],
    time_seconds = Float64[]
)

println("Running benchmarks...")
println("-" ^ 70)

for (i, seed) in enumerate(SEEDS)
    println("\nRun $i/$(length(SEEDS)) - Seed: $seed")
    
    # Create temp directory
    benchmark_dir = mktempdir()
    
    # Create data files
    omics_cols = vcat([:ID], [Symbol("omic$j") for j in 1:10])
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
            omics_name=["omic$j" for j in 1:10],
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
    
    # Run MCMC
    Random.seed!(seed)
    start_time = time()
    result = runNNMM(layers, equations;
        chain_length=CHAIN_LENGTH,
        burnin=BURNIN,
        printout_frequency=CHAIN_LENGTH + 1,
        output_folder=joinpath(benchmark_dir, "output")
    )
    elapsed = time() - start_time
    
    # Get EBV results
    ebv_df = result["EBV_NonLinear"]
    ebv_df.ID = string.(ebv_df.ID)
    
    # Get EPV results
    epv_df = result["EPV_NonLinear"]
    epv_df.ID = string.(epv_df.ID)
    
    pheno_df_copy = copy(pheno_df)
    pheno_df_copy.ID = string.(pheno_df_copy.ID)
    
    # Merge and calculate accuracy
    merged_df = innerjoin(ebv_df, pheno_df_copy[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect]], on=:ID)
    merged_epv = innerjoin(epv_df, pheno_df_copy[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect]], on=:ID)
    
    ebv_accuracy_total = cor(merged_df.EBV, merged_df.genetic_total)
    ebv_accuracy_direct = cor(merged_df.EBV, merged_df.genetic_direct)
    ebv_accuracy_indirect = cor(merged_df.EBV, merged_df.genetic_indirect)
    
    epv_col = hasproperty(merged_epv, :EPV) ? merged_epv.EPV : merged_epv.EBV
    epv_accuracy_total = cor(epv_col, merged_epv.genetic_total)
    epv_accuracy_direct = cor(epv_col, merged_epv.genetic_direct)
    epv_accuracy_indirect = cor(epv_col, merged_epv.genetic_indirect)
    
    push!(results, (
        seed = seed,
        ebv_accuracy_total = ebv_accuracy_total,
        ebv_accuracy_direct = ebv_accuracy_direct,
        ebv_accuracy_indirect = ebv_accuracy_indirect,
        epv_accuracy_total = epv_accuracy_total,
        epv_accuracy_direct = epv_accuracy_direct,
        epv_accuracy_indirect = epv_accuracy_indirect,
        time_seconds = elapsed
    ))
    
    println("  EBV Accuracy (genetic_total): $(round(ebv_accuracy_total, digits=4))")
    println("  EPV Accuracy (genetic_total): $(round(epv_accuracy_total, digits=4))")
    println("  Time: $(round(elapsed, digits=1))s")
    
    # Cleanup
    rm(benchmark_dir, recursive=true)
end

# Summary
println("\n" * "=" ^ 70)
println("CONVERGENCE SUMMARY")
println("=" ^ 70)
println()

println("EBV ACCURACY:")
println("┌────────┬─────────────────┬─────────────────┬─────────────────┬──────────┐")
println("│  Seed  │ genetic_total   │ genetic_direct  │ genetic_indirect│ Time (s) │")
println("├────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤")

for row in eachrow(results)
    println("│ $(lpad(row.seed, 6)) │ $(lpad(round(row.ebv_accuracy_total, digits=4), 15)) │ $(lpad(round(row.ebv_accuracy_direct, digits=4), 15)) │ $(lpad(round(row.ebv_accuracy_indirect, digits=4), 15)) │ $(lpad(round(row.time_seconds, digits=1), 8)) │")
end

println("├────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤")
println("│  Mean  │ $(lpad(round(mean(results.ebv_accuracy_total), digits=4), 15)) │ $(lpad(round(mean(results.ebv_accuracy_direct), digits=4), 15)) │ $(lpad(round(mean(results.ebv_accuracy_indirect), digits=4), 15)) │ $(lpad(round(mean(results.time_seconds), digits=1), 8)) │")
println("│  Std   │ $(lpad(round(std(results.ebv_accuracy_total), digits=4), 15)) │ $(lpad(round(std(results.ebv_accuracy_direct), digits=4), 15)) │ $(lpad(round(std(results.ebv_accuracy_indirect), digits=4), 15)) │ $(lpad(round(std(results.time_seconds), digits=1), 8)) │")
println("└────────┴─────────────────┴─────────────────┴─────────────────┴──────────┘")
println()

println("EPV ACCURACY:")
println("┌────────┬─────────────────┬─────────────────┬─────────────────┐")
println("│  Seed  │ genetic_total   │ genetic_direct  │ genetic_indirect│")
println("├────────┼─────────────────┼─────────────────┼─────────────────┤")

for row in eachrow(results)
    println("│ $(lpad(row.seed, 6)) │ $(lpad(round(row.epv_accuracy_total, digits=4), 15)) │ $(lpad(round(row.epv_accuracy_direct, digits=4), 15)) │ $(lpad(round(row.epv_accuracy_indirect, digits=4), 15)) │")
end

println("├────────┼─────────────────┼─────────────────┼─────────────────┤")
println("│  Mean  │ $(lpad(round(mean(results.epv_accuracy_total), digits=4), 15)) │ $(lpad(round(mean(results.epv_accuracy_direct), digits=4), 15)) │ $(lpad(round(mean(results.epv_accuracy_indirect), digits=4), 15)) │")
println("│  Std   │ $(lpad(round(std(results.epv_accuracy_total), digits=4), 15)) │ $(lpad(round(std(results.epv_accuracy_direct), digits=4), 15)) │ $(lpad(round(std(results.epv_accuracy_indirect), digits=4), 15)) │")
println("└────────┴─────────────────┴─────────────────┴─────────────────┘")

println()
println("CONVERGENCE ASSESSMENT:")
ebv_std = std(results.ebv_accuracy_total)
epv_std = std(results.epv_accuracy_total)
if ebv_std < 0.02 && epv_std < 0.02
    println("  ✓ CONVERGED - EBV std=$(round(ebv_std, digits=4)), EPV std=$(round(epv_std, digits=4)) < 0.02")
elseif ebv_std < 0.05 && epv_std < 0.05
    println("  ⚠ MARGINAL - EBV std=$(round(ebv_std, digits=4)), EPV std=$(round(epv_std, digits=4)) between 0.02 and 0.05")
else
    println("  ✗ NOT CONVERGED - EBV std=$(round(ebv_std, digits=4)), EPV std=$(round(epv_std, digits=4)) > 0.05")
end

println()
println("=" ^ 70)

# Save results
results_file = joinpath(@__DIR__, "convergence_seeds_results.csv")
CSV.write(results_file, results)
println("Results saved to: $results_file")


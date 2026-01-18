#!/usr/bin/env julia
"""
NNMM.jl Performance Benchmark
=============================

Measures MCMC execution time for comparison with PyNNMM.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NNMM
using NNMM.Datasets
using CSV
using DataFrames
using Statistics
using Dates
using Printf

# Configuration
const SEED = 42

function run_benchmark(chain_length::Int, burnin::Int; n_runs::Int=3)
    """Run benchmark with specified chain length."""
    
    # Data files
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    times = Float64[]
    
    for i in 1:n_runs
        # Create output folder
        benchmark_dir = mktempdir()
        
        # Load phenotypes and create files
        pheno_df = CSV.read(pheno_path, DataFrame)
        
        # Create omics file
        omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
        omics_df = pheno_df[:, omics_cols]
        omics_path = joinpath(benchmark_dir, "omics.csv")
        CSV.write(omics_path, omics_df; missingstring="NA")
        
        # Create phenotypes file
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
        
        # Time the MCMC
        start_time = time()
        result = runNNMM(layers, equations;
            chain_length=chain_length,
            burnin=burnin,
            printout_frequency=chain_length + 1,
            output_folder=joinpath(benchmark_dir, "output")
        )
        elapsed = time() - start_time
        
        push!(times, elapsed)
        
        # Cleanup
        rm(benchmark_dir, recursive=true)
    end
    
    return Dict(
        "mean" => mean(times),
        "std" => std(times),
        "min" => minimum(times),
        "max" => maximum(times),
        "runs" => times
    )
end


function main()
    println("=" ^ 70)
    println("NNMM.jl Performance Benchmark")
    println("=" ^ 70)
    println("Date: $(now())")
    println("Julia: $(VERSION)")
    println()
    
    # Load data to get dimensions
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    n_individuals = nrow(pheno_df)
    n_snps = 1000  # Known from dataset
    n_snps_filtered = 927  # After MAF filtering
    
    println("Dataset:")
    println("  Individuals: $n_individuals")
    println("  SNPs: $n_snps ($n_snps_filtered after MAF filtering)")
    println("  Omics: 10")
    println()
    
    # Warm-up run
    println("Warming up JIT...")
    run_benchmark(10, 2; n_runs=1)
    println()
    
    # Benchmark at different chain lengths
    chain_configs = [
        (100, 20),
        (500, 100),
        (1000, 200),
    ]
    
    results = DataFrame(
        chain_length = Int[],
        burnin = Int[],
        mean_time = Float64[],
        std_time = Float64[],
        iter_per_sec = Float64[]
    )
    
    println("Running benchmarks...")
    println("-" ^ 70)
    
    for (chain_length, burnin) in chain_configs
        println("\nChain length: $chain_length, Burnin: $burnin")
        
        bench = run_benchmark(chain_length, burnin; n_runs=3)
        
        iter_per_sec = chain_length / bench["mean"]
        
        push!(results, (
            chain_length = chain_length,
            burnin = burnin,
            mean_time = bench["mean"],
            std_time = bench["std"],
            iter_per_sec = iter_per_sec
        ))
        
        println("  Time: $(round(bench["mean"], digits=2))s ± $(round(bench["std"], digits=2))s")
        println("  Iterations/sec: $(round(iter_per_sec, digits=1))")
    end
    
    # Print summary table
    println("\n" * "=" ^ 70)
    println("PERFORMANCE SUMMARY")
    println("=" ^ 70)
    println()
    println("Dataset: $n_individuals individuals × $n_snps_filtered SNPs (after MAF)")
    println()
    println("┌─────────────┬─────────┬─────────────┬──────────────┐")
    println("│ Chain       │ Burnin  │ Time (s)    │ Iter/sec     │")
    println("├─────────────┼─────────┼─────────────┼──────────────┤")
    
    for row in eachrow(results)
        time_str = @sprintf("%8.2f±%.2f", row.mean_time, row.std_time)
        println("│ $(lpad(row.chain_length, 11)) │ $(lpad(row.burnin, 7)) │ $(rpad(time_str, 11)) │ $(lpad(round(row.iter_per_sec, digits=1), 12)) │")
    end
    
    println("└─────────────┴─────────┴─────────────┴──────────────┘")
    println()
    
    # Save results
    results_file = joinpath(@__DIR__, "performance_results_julia.csv")
    CSV.write(results_file, results)
    
    println("Results saved to: benchmarks/performance_results_julia.csv")
    println()
    println("=" ^ 70)
end


# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

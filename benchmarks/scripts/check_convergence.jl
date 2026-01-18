#!/usr/bin/env julia
#=
Check MCMC Convergence by Running at Different Chain Lengths
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

const SEED = 42
const CHAIN_LENGTHS = [1000, 1500, 2000, 3000]

# Load simulated dataset
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

println("=" ^ 70)
println("MCMC Convergence Check")
println("=" ^ 70)
println("Testing chain lengths: ", CHAIN_LENGTHS)
println()

results = DataFrame(
    chain_length = Int[],
    burnin = Int[],
    accuracy_total = Float64[],
    accuracy_direct = Float64[],
    accuracy_indirect = Float64[]
)

for chain_length in CHAIN_LENGTHS
    Random.seed!(SEED)
    burnin = div(chain_length, 5)  # 20% burnin
    
    println("-" ^ 70)
    println("Running chain_length=$chain_length, burnin=$burnin...")
    println("-" ^ 70)
    
    benchmark_dir = mktempdir()
    
    # Create data files
    omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
    omics_df = pheno_df[:, omics_cols]
    omics_path = joinpath(benchmark_dir, "omics.csv")
    CSV.write(omics_path, omics_df; missingstring="NA")
    
    pheno_out_df = pheno_df[:, [:ID, :trait1]]
    pheno_out_path = joinpath(benchmark_dir, "phenotypes.csv")
    CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")
    
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
    
    result = runNNMM(layers, equations;
        chain_length=chain_length,
        burnin=burnin,
        printout_frequency=chain_length + 1,
        output_folder=joinpath(benchmark_dir, "output")
    )
    
    # Calculate accuracy
    ebv_df = result["EBV_NonLinear"]
    ebv_df.ID = string.(ebv_df.ID)
    pheno_df_copy = copy(pheno_df)
    pheno_df_copy.ID = string.(pheno_df_copy.ID)
    
    merged_df = innerjoin(ebv_df, pheno_df_copy[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect]], on=:ID)
    
    accuracy_total = cor(merged_df.EBV, merged_df.genetic_total)
    accuracy_direct = cor(merged_df.EBV, merged_df.genetic_direct)
    accuracy_indirect = cor(merged_df.EBV, merged_df.genetic_indirect)
    
    push!(results, (chain_length, burnin, accuracy_total, accuracy_direct, accuracy_indirect))
    
    println("  Accuracy: cor(EBV, genetic_total) = $(round(accuracy_total, digits=4))")
    println()
    
    rm(benchmark_dir, recursive=true)
end

# Print summary
println()
println("=" ^ 70)
println("CONVERGENCE CHECK RESULTS")
println("=" ^ 70)
println()
println("┌──────────────┬─────────┬─────────────────┬────────────────┬──────────────────┐")
println("│ Chain Length │ Burnin  │ cor(EBV,total)  │ cor(EBV,direct)│ cor(EBV,indirect)│")
println("├──────────────┼─────────┼─────────────────┼────────────────┼──────────────────┤")
for row in eachrow(results)
    cl_str = lpad(string(row.chain_length), 8)
    bu_str = lpad(string(row.burnin), 6)
    total_str = lpad(string(round(row.accuracy_total, digits=4)), 10)
    direct_str = lpad(string(round(row.accuracy_direct, digits=4)), 10)
    indirect_str = lpad(string(round(row.accuracy_indirect, digits=4)), 10)
    println("│ $cl_str     │ $bu_str  │ $total_str      │ $direct_str     │ $indirect_str       │")
end
println("└──────────────┴─────────┴─────────────────┴────────────────┴──────────────────┘")
println()

# Check convergence
if nrow(results) >= 2
    first_acc = results[1, :accuracy_total]
    last_acc = results[end, :accuracy_total]
    diff = abs(last_acc - first_acc)
    println("Accuracy change from $(results[1,:chain_length]) to $(results[end,:chain_length]): $(round(diff, digits=4))")
    if diff < 0.01
        println("✓ Chain appears to have converged (change < 0.01)")
    else
        println("⚠ Chain may not have converged (change >= 0.01)")
    end
end
println()


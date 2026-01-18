#!/usr/bin/env julia
#=
Debug Script: Test variance constraint for Layer 2 to fix scale drift
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

println("=" ^ 70)
println("DEBUGGING: Layer 2 Variance Constraint")
println("=" ^ 70)

const SEED = 42
const CHAIN_LENGTH = 1000
const BURNIN = 200

# Load simulated dataset
println("\nLoading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

# Genetic values for comparison
genetic_total = pheno_df.genetic_total
genetic_direct = pheno_df.genetic_direct
genetic_indirect = pheno_df.genetic_indirect

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

# Function to create fresh layers
function create_fresh_layers(geno_path, omics_path, pheno_out_path)
    return [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
    ]
end

println("\n" * "=" ^ 70)
println("TEST 1: Original (estimate_variance_G=true for Layer 2)")
println("=" ^ 70)

Random.seed!(SEED)
layers1 = create_fresh_layers(geno_path, omics_path, pheno_out_path)

equations_original = [
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
        activation_function="linear",
        estimate_variance_G=true  # DEFAULT - variance can drift
    )
]

result1 = runNNMM(layers1, equations_original;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=joinpath(benchmark_dir, "output1")
)

ebv1 = result1["EBV_NonLinear"].EBV
cor1_total = cor(ebv1, genetic_total)
cor1_direct = cor(ebv1, genetic_direct)
cor1_indirect = cor(ebv1, genetic_indirect)

println("\nOriginal Results:")
println("  EBV Std:                  $(round(std(ebv1), digits=2))")
println("  cor(EBV, genetic_total):  $(round(cor1_total, digits=4))")
println("  cor(EBV, genetic_direct): $(round(cor1_direct, digits=4))")
println("  cor(EBV, genetic_indirect): $(round(cor1_indirect, digits=4))")

println("\n" * "=" ^ 70)
println("TEST 2: Fixed Variance (estimate_variance_G=false for Layer 2)")
println("=" ^ 70)

Random.seed!(SEED)
layers2 = create_fresh_layers(geno_path, omics_path, pheno_out_path)

equations_fixed = [
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
        activation_function="linear",
        estimate_variance_G=false  # FIX - keep variance fixed
    )
]

result2 = runNNMM(layers2, equations_fixed;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=joinpath(benchmark_dir, "output2")
)

ebv2 = result2["EBV_NonLinear"].EBV
cor2_total = cor(ebv2, genetic_total)
cor2_direct = cor(ebv2, genetic_direct)
cor2_indirect = cor(ebv2, genetic_indirect)

println("\nFixed Variance Results:")
println("  EBV Std:                  $(round(std(ebv2), digits=2))")
println("  cor(EBV, genetic_total):  $(round(cor2_total, digits=4))")
println("  cor(EBV, genetic_direct): $(round(cor2_direct, digits=4))")
println("  cor(EBV, genetic_indirect): $(round(cor2_indirect, digits=4))")

println("\n" * "=" ^ 70)
println("TEST 3: Stronger Prior (df_G=10 for Layer 2)")
println("=" ^ 70)

Random.seed!(SEED)
layers3 = create_fresh_layers(geno_path, omics_path, pheno_out_path)

equations_stronger_prior = [
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
        activation_function="linear",
        df_G=10.0  # Stronger prior (default is 4.0)
    )
]

result3 = runNNMM(layers3, equations_stronger_prior;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=joinpath(benchmark_dir, "output3")
)

ebv3 = result3["EBV_NonLinear"].EBV
cor3_total = cor(ebv3, genetic_total)
cor3_direct = cor(ebv3, genetic_direct)
cor3_indirect = cor(ebv3, genetic_indirect)

println("\nStronger Prior Results:")
println("  EBV Std:                  $(round(std(ebv3), digits=2))")
println("  cor(EBV, genetic_total):  $(round(cor3_total, digits=4))")
println("  cor(EBV, genetic_direct): $(round(cor3_direct, digits=4))")
println("  cor(EBV, genetic_indirect): $(round(cor3_indirect, digits=4))")

println("\n" * "=" ^ 70)
println("COMPARISON SUMMARY")
println("=" ^ 70)

println("\n┌─────────────────────┬────────────┬────────────┬────────────┐")
println("│ Metric              │ Original   │ Fixed Var  │ df_G=10    │")
println("├─────────────────────┼────────────┼────────────┼────────────┤")
println("│ EBV Std             │ $(lpad(round(std(ebv1), digits=1), 10)) │ $(lpad(round(std(ebv2), digits=1), 10)) │ $(lpad(round(std(ebv3), digits=1), 10)) │")
println("│ cor(genetic_total)  │ $(lpad(round(cor1_total, digits=4), 10)) │ $(lpad(round(cor2_total, digits=4), 10)) │ $(lpad(round(cor3_total, digits=4), 10)) │")
println("│ cor(genetic_indirect)│ $(lpad(round(cor1_indirect, digits=4), 10)) │ $(lpad(round(cor2_indirect, digits=4), 10)) │ $(lpad(round(cor3_indirect, digits=4), 10)) │")
println("└─────────────────────┴────────────┴────────────┴────────────┘")

# Cross-correlation between fixes
cor_1_2 = cor(ebv1, ebv2)
cor_1_3 = cor(ebv1, ebv3)
cor_2_3 = cor(ebv2, ebv3)

println("\nCross-correlation between methods:")
println("  cor(Original, Fixed):     $(round(cor_1_2, digits=4))")
println("  cor(Original, df_G=10):   $(round(cor_1_3, digits=4))")
println("  cor(Fixed, df_G=10):      $(round(cor_2_3, digits=4))")

# Cleanup
rm(benchmark_dir, recursive=true)

println("\n" * "=" ^ 70)
println("RECOMMENDATION")
println("=" ^ 70)
if std(ebv2) < std(ebv1) / 10 && cor2_total > 0.8
    println("✅ estimate_variance_G=false FIXES the scale issue!")
    println("   Scale reduced from $(round(std(ebv1), digits=1)) to $(round(std(ebv2), digits=1))")
    println("   Accuracy maintained: $(round(cor2_total, digits=4))")
elseif std(ebv3) < std(ebv1) / 2 && cor3_total > 0.8
    println("✅ df_G=10 HELPS with the scale issue!")
    println("   Scale reduced from $(round(std(ebv1), digits=1)) to $(round(std(ebv3), digits=1))")
else
    println("⚠️ Neither fix fully resolves the scale issue. Further investigation needed.")
end


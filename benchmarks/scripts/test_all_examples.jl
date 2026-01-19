#!/usr/bin/env julia
#=
Test all documentation examples - Updated to use simulated_omics_data
All examples now use aligned data (3534 individuals)
All temp files and outputs are created in a temp directory to avoid polluting the workspace.
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

println("="^70)
println("NNMM DOCUMENTATION EXAMPLES - COMPREHENSIVE TEST")
println("="^70)
println("All examples use simulated_omics_data (3534 aligned individuals)")
println("Running 8 tests with chain_length=50, burnin=10")
println()

CHAIN_LENGTH = 50
BURNIN = 10

# Create a temp directory for all test outputs
const TEMP_DIR = mktempdir()
println("Using temp directory: $TEMP_DIR")
println()

test_results = []

#--- TEST 1: Basic NNMM (10 omics, linear) ---
print("TEST 1: Basic NNMM (10 omics, linear)... ")
try
    Random.seed!(42)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    omics_cols = vcat(:ID, [Symbol("omic$i") for i in 1:10])
    omics_df = pheno_df[:, omics_cols]
    omics_file = joinpath(TEMP_DIR, "t1_omics.csv")
    trait_file = joinpath(TEMP_DIR, "t1_trait.csv")
    CSV.write(omics_file, omics_df; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_file, missing_value="NA"),
        Layer(layer_name="pheno", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="geno", to_layer_name="omics", equation="omics = intercept + geno",
                 traits=["omic$i" for i in 1:10], method="BayesC"),
        Equation(from_layer_name="omics", to_layer_name="pheno", equation="pheno = intercept + omics",
                 traits=["trait1"], activation_function="linear")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t1_out"))
    println("✓ PASS")
    push!(test_results, ("1: Basic NNMM (10 omics)", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("1: Basic NNMM (10 omics)", false, sprint(showerror, e)))
end

#--- TEST 2: Latent Traits (3 latent, tanh) - Using simulated data ---
print("TEST 2: Latent Traits (3 latent, tanh)... ")
try
    Random.seed!(123)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    n = nrow(pheno_df)
    
    latent_file = joinpath(TEMP_DIR, "t2_latent.csv")
    trait_file = joinpath(TEMP_DIR, "t2_trait.csv")
    latent_df = DataFrame(ID=pheno_df.ID, latent1=fill(missing,n), latent2=fill(missing,n), latent3=fill(missing,n))
    CSV.write(latent_file, latent_df; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="genotypes", data_path=[geno_path]),
        Layer(layer_name="latent", data_path=latent_file, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="genotypes", to_layer_name="latent", equation="latent = intercept + genotypes",
                 traits=["latent1", "latent2", "latent3"], method="BayesC"),
        Equation(from_layer_name="latent", to_layer_name="phenotypes", equation="phenotypes = intercept + latent",
                 traits=["trait1"], activation_function="tanh")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t2_out"))
    println("✓ PASS")
    push!(test_results, ("2: Latent Traits (tanh)", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("2: Latent Traits (tanh)", false, sprint(showerror, e)))
end

#--- TEST 3: Observed Omics (3 omics, sigmoid) ---
print("TEST 3: Observed Omics (3 omics, sigmoid)... ")
try
    Random.seed!(123)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    omics_file = joinpath(TEMP_DIR, "t3_omics.csv")
    trait_file = joinpath(TEMP_DIR, "t3_trait.csv")
    CSV.write(omics_file, pheno_df[:, [:ID, :omic1, :omic2, :omic3]]; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_file, missing_value="NA"),
        Layer(layer_name="pheno", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="geno", to_layer_name="omics", equation="omics = intercept + geno",
                 traits=["omic1", "omic2", "omic3"], method="BayesC"),
        Equation(from_layer_name="omics", to_layer_name="pheno", equation="pheno = intercept + omics",
                 traits=["trait1"], activation_function="sigmoid")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t3_out"))
    println("✓ PASS")
    push!(test_results, ("3: Observed Omics (sigmoid)", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("3: Observed Omics (sigmoid)", false, sprint(showerror, e)))
end

#--- TEST 4: Partial Connected (3 geno groups) ---
print("TEST 4: Partial Connected (3 geno groups)... ")
try
    Random.seed!(123)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    # Split genotypes into 3 groups
    geno_full = CSV.read(geno_path, DataFrame)
    n_markers = ncol(geno_full) - 1
    markers_per_group = div(n_markers, 3)
    
    geno1_cols = vcat(:ID, names(geno_full)[2:markers_per_group+1])
    geno2_cols = vcat(:ID, names(geno_full)[markers_per_group+2:2*markers_per_group+1])
    geno3_cols = vcat(:ID, names(geno_full)[2*markers_per_group+2:end])
    
    geno1_file = joinpath(TEMP_DIR, "t4_geno1.csv")
    geno2_file = joinpath(TEMP_DIR, "t4_geno2.csv")
    geno3_file = joinpath(TEMP_DIR, "t4_geno3.csv")
    omics_file = joinpath(TEMP_DIR, "t4_omics.csv")
    trait_file = joinpath(TEMP_DIR, "t4_trait.csv")
    
    CSV.write(geno1_file, geno_full[:, geno1_cols])
    CSV.write(geno2_file, geno_full[:, geno2_cols])
    CSV.write(geno3_file, geno_full[:, geno3_cols])
    CSV.write(omics_file, pheno_df[:, [:ID, :omic1, :omic2, :omic3]]; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="geno", data_path=[geno1_file, geno2_file, geno3_file]),
        Layer(layer_name="omics", data_path=omics_file, missing_value="NA"),
        Layer(layer_name="pheno", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="geno", to_layer_name="omics", equation="omics = intercept + geno",
                 traits=["omic1", "omic2", "omic3"], method="BayesC"),
        Equation(from_layer_name="omics", to_layer_name="pheno", equation="pheno = intercept + omics",
                 traits=["trait1"], activation_function="sigmoid")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t4_out"))
    println("✓ PASS")
    push!(test_results, ("4: Partial Connected", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("4: Partial Connected", false, sprint(showerror, e)))
end

#--- TEST 5: tanh as workaround for user-defined function ---
print("TEST 5: tanh (workaround for custom func)... ")
try
    Random.seed!(123)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    n = nrow(pheno_df)
    
    latent_file = joinpath(TEMP_DIR, "t5_latent.csv")
    trait_file = joinpath(TEMP_DIR, "t5_trait.csv")
    latent_df = DataFrame(ID=pheno_df.ID, latent1=fill(missing,n), latent2=fill(missing,n))
    CSV.write(latent_file, latent_df; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="genotypes", data_path=[geno_path]),
        Layer(layer_name="latent", data_path=latent_file, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="genotypes", to_layer_name="latent", equation="latent = intercept + genotypes",
                 traits=["latent1", "latent2"], method="BayesC"),
        Equation(from_layer_name="latent", to_layer_name="phenotypes", equation="phenotypes = intercept + latent",
                 traits=["trait1"], activation_function="tanh")  # Use tanh as workaround
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t5_out"))
    println("✓ PASS")
    push!(test_results, ("5: tanh (custom workaround)", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("5: tanh (custom workaround)", false, sprint(showerror, e)))
end

#--- TEST 6: Traditional BayesC (linear) ---
print("TEST 6: Traditional BayesC (linear)... ")
try
    Random.seed!(42)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    n = nrow(pheno_df)
    
    latent_file = joinpath(TEMP_DIR, "t6_latent.csv")
    trait_file = joinpath(TEMP_DIR, "t6_trait.csv")
    latent_df = DataFrame(ID=pheno_df.ID, latent1=fill(missing,n), latent2=fill(missing,n))
    CSV.write(latent_file, latent_df; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="latent", data_path=latent_file, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="geno", to_layer_name="latent", equation="latent = intercept + geno",
                 traits=["latent1", "latent2"], method="BayesC"),
        Equation(from_layer_name="latent", to_layer_name="phenotypes", equation="phenotypes = intercept + latent",
                 traits=["trait1"], activation_function="linear")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t6_out"))
    println("✓ PASS")
    push!(test_results, ("6: Traditional BayesC", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("6: Traditional BayesC", false, sprint(showerror, e)))
end

#--- TEST 7: BayesA method ---
print("TEST 7: BayesA method... ")
try
    Random.seed!(42)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    omics_file = joinpath(TEMP_DIR, "t7_omics.csv")
    trait_file = joinpath(TEMP_DIR, "t7_trait.csv")
    CSV.write(omics_file, pheno_df[:, [:ID, :omic1, :omic2, :omic3]]; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_file, missing_value="NA"),
        Layer(layer_name="pheno", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="geno", to_layer_name="omics", equation="omics = intercept + geno",
                 traits=["omic1", "omic2", "omic3"], method="BayesA"),
        Equation(from_layer_name="omics", to_layer_name="pheno", equation="pheno = intercept + omics",
                 traits=["trait1"], activation_function="linear")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t7_out"))
    println("✓ PASS")
    push!(test_results, ("7: BayesA method", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("7: BayesA method", false, sprint(showerror, e)))
end

#--- TEST 8: RR-BLUP method ---
print("TEST 8: RR-BLUP method... ")
try
    Random.seed!(42)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    omics_file = joinpath(TEMP_DIR, "t8_omics.csv")
    trait_file = joinpath(TEMP_DIR, "t8_trait.csv")
    CSV.write(omics_file, pheno_df[:, [:ID, :omic1, :omic2, :omic3]]; missingstring="NA")
    CSV.write(trait_file, pheno_df[:, [:ID, :trait1]]; missingstring="NA")
    
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_file, missing_value="NA"),
        Layer(layer_name="pheno", data_path=trait_file, missing_value="NA")
    ]
    equations = [
        Equation(from_layer_name="geno", to_layer_name="omics", equation="omics = intercept + geno",
                 traits=["omic1", "omic2", "omic3"], method="RR-BLUP"),
        Equation(from_layer_name="omics", to_layer_name="pheno", equation="pheno = intercept + omics",
                 traits=["trait1"], activation_function="linear")
    ]
    out = runNNMM(layers, equations; chain_length=CHAIN_LENGTH, burnin=BURNIN,
                  printout_frequency=999, output_folder=joinpath(TEMP_DIR, "t8_out"))
    println("✓ PASS")
    push!(test_results, ("8: RR-BLUP method", true, ""))
catch e
    println("✗ FAIL")
    push!(test_results, ("8: RR-BLUP method", false, sprint(showerror, e)))
end

# Summary
println()
println("="^70)
println("SUMMARY")
println("="^70)
passed = count(x -> x[2], test_results)
println("PASSED: $passed/$(length(test_results))")
println()

if passed < length(test_results)
    println("FAILED TESTS:")
    for (name, status, err) in test_results
        if !status
            println("  ✗ $name")
            println("    Error: $(err[1:min(200, length(err))])")
        end
    end
else
    println("✓ All tests passed!")
end
println("="^70)

#!/usr/bin/env julia
#=
================================================================================
Validation: EPV vs EBV accuracy with/without skip connection
================================================================================
Purpose:
  When fitting all omics + direct genotype→phenotype skip, do EPV and EBV
  give the same accuracy for test individuals who have observed omics?

Setup:
  - Simulated data: 3534 individuals, 1000 SNPs, 10 omics
  - 80% train / 20% test split
  - Test individuals have OBSERVED omics (key: not missing)
  - Two models:
    (A) No skip:   phenotypes = intercept + omics
    (B) With skip: phenotypes = intercept + omics + genotypes

Expected results:
  - Model A (no skip): EPV should outperform EBV on test set because
    observed omics capture genetic + environmental info
  - Model B (with skip): EPV ≈ EBV because the skip connection absorbs
    the genetic signal, shrinking the omics weights (α₂) toward zero
================================================================================
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using NNMM
using NNMM.Datasets
using CSV, DataFrames, Random, Statistics, DelimitedFiles
using Printf

# ============================================================
# Configuration
# ============================================================
const SEED           = 42
const CHAIN_LENGTH   = 5000
const BURNIN         = 1000
const TEST_FRAC      = 0.20
const N_OMICS        = 10

println("="^70)
println("Validation: EPV vs EBV with skip connection")
println("="^70)
println("  CHAIN_LENGTH = $CHAIN_LENGTH")
println("  BURNIN       = $BURNIN")
println("  TEST_FRAC    = $TEST_FRAC")
println("  SEED         = $SEED")
println("="^70)

# ============================================================
# Load simulated data
# ============================================================
geno_path  = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt",     dataset_name="simulated_omics_data")

pheno_df = CSV.read(pheno_path, DataFrame)
pheno_df.ID = string.(pheno_df.ID)

n_individuals = nrow(pheno_df)
omics_syms = [Symbol("omic$i") for i in 1:N_OMICS]

println("\nData loaded:")
println("  Individuals: $n_individuals")
println("  SNPs:        1000")
println("  Omics:       $N_OMICS")
println("  Columns:     $(names(pheno_df))")

# ============================================================
# Train / test split
# ============================================================
rng_split = MersenneTwister(SEED)
perm      = shuffle(rng_split, collect(1:n_individuals))
n_test    = round(Int, n_individuals * TEST_FRAC)
test_idx  = sort(perm[1:n_test])
train_idx = sort(perm[n_test+1:end])
test_id_set  = Set(pheno_df.ID[test_idx])
train_id_set = Set(pheno_df.ID[train_idx])

println("\nSplit:")
println("  Train: $(length(train_idx))")
println("  Test:  $(length(test_idx))")

# ============================================================
# Prepare input files (shared by both models)
# ============================================================
inputs_dir = mktempdir()
println("\nTemp dir: $inputs_dir")

# Omics file: ALL omics observed for ALL individuals (train + test)
omics_df = pheno_df[:, vcat([:ID], omics_syms)]
omics_path = joinpath(inputs_dir, "omics_all_observed.csv")
CSV.write(omics_path, omics_df; missingstring="NA")
println("Omics: all observed for train + test")

# Phenotype file: mask test individuals
pheno_out_df = copy(pheno_df[:, [:ID, :trait1]])
allowmissing!(pheno_out_df, :trait1)
pheno_out_df[test_idx, :trait1] .= missing
pheno_out_path = joinpath(inputs_dir, "phenotypes_masked.csv")
CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")
println("Phenotypes: test individuals masked")

# ============================================================
# Helper: run NNMM
# ============================================================
function run_model(; use_skip::Bool, output_name::String)
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA"),
    ]

    eq1 = Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = intercept + geno",
        traits=["omic$i" for i in 1:N_OMICS],
        method="BayesC",
        estimatePi=true,
    )

    if use_skip
        class_priors = Dict(
            "omics" => (
                method      = "BayesC",
                Pi          = 0.0,
                estimatePi  = false,
                G           = false,
                G_is_marker_variance = false,
                df_G        = 4.0,
                estimate_variance_G = true,
                estimate_scale_G    = false,
                constraint_G        = true,
            ),
            "geno" => (
                method      = "BayesC",
                Pi          = 0.95,
                estimatePi  = true,
                G           = false,
                G_is_marker_variance = false,
                df_G        = 4.0,
                estimate_variance_G = true,
                estimate_scale_G    = false,
                constraint_G        = true,
            ),
        )
        eq2 = Equation(
            from_layer_name="omics",
            to_layer_name="phenotypes",
            equation="phenotypes = intercept + omics + geno",
            traits=["trait1"],
            activation_function="linear",
            class_priors=class_priors,
        )
    else
        eq2 = Equation(
            from_layer_name="omics",
            to_layer_name="phenotypes",
            equation="phenotypes = intercept + omics",
            traits=["trait1"],
            activation_function="linear",
        )
    end

    out_dir = joinpath(inputs_dir, output_name)
    result = runNNMM(
        layers, [eq1, eq2];
        chain_length = CHAIN_LENGTH,
        burnin       = BURNIN,
        printout_frequency = CHAIN_LENGTH + 1,
        seed         = SEED,
        output_folder = out_dir,
    )
    return result, out_dir
end

# ============================================================
# Helper: compute accuracies
# ============================================================
function compute_accuracies(result, label; out_dir=nothing)
    println("\n--- $label ---")

    # --- EBV (Estimated Breeding Value) ---
    ebv_df = result["EBV_NonLinear"]
    ebv_df.ID = string.(ebv_df.ID)

    # --- EPV (Estimated Phenotypic Value on output IDs) ---
    epv_df = result["EPV_Output_NonLinear"]
    epv_df.ID = string.(epv_df.ID)

    # Merge with truth
    truth = pheno_df[:, [:ID, :trait1, :genetic_total, :genetic_direct, :genetic_indirect]]

    merged_ebv = innerjoin(ebv_df, truth, on=:ID)
    merged_epv = innerjoin(epv_df, truth, on=:ID)

    # Separate train/test
    for (set_name, id_set) in [("TRAIN", train_id_set), ("TEST", test_id_set)]
        ebv_set = filter(r -> r.ID in id_set, merged_ebv)
        epv_set = filter(r -> r.ID in id_set, merged_epv)

        ebv_acc_total    = cor(ebv_set.EBV, ebv_set.genetic_total)
        ebv_acc_direct   = cor(ebv_set.EBV, ebv_set.genetic_direct)
        ebv_acc_indirect = cor(ebv_set.EBV, ebv_set.genetic_indirect)
        ebv_acc_trait    = cor(ebv_set.EBV, ebv_set.trait1)

        epv_acc_total    = cor(epv_set.EPV, epv_set.genetic_total)
        epv_acc_direct   = cor(epv_set.EPV, epv_set.genetic_direct)
        epv_acc_indirect = cor(epv_set.EPV, epv_set.genetic_indirect)
        epv_acc_trait    = cor(epv_set.EPV, epv_set.trait1)

        # Also compute r(EPV, EBV)
        both = innerjoin(ebv_set[:, [:ID, :EBV]], epv_set[:, [:ID, :EPV]], on=:ID)
        r_epv_ebv = cor(both.EBV, both.EPV)

        println("\n  $set_name (n=$(nrow(ebv_set))):")
        @printf("    %-30s  EBV      EPV      Δ(EPV-EBV)\n", "")
        @printf("    cor(., genetic_total)       %+.4f   %+.4f   %+.4f\n",
                ebv_acc_total, epv_acc_total, epv_acc_total - ebv_acc_total)
        @printf("    cor(., genetic_direct)      %+.4f   %+.4f   %+.4f\n",
                ebv_acc_direct, epv_acc_direct, epv_acc_direct - ebv_acc_direct)
        @printf("    cor(., genetic_indirect)    %+.4f   %+.4f   %+.4f\n",
                ebv_acc_indirect, epv_acc_indirect, epv_acc_indirect - ebv_acc_indirect)
        @printf("    cor(., trait1)              %+.4f   %+.4f   %+.4f\n",
                ebv_acc_trait, epv_acc_trait, epv_acc_trait - ebv_acc_trait)
        @printf("    cor(EBV, EPV)              %.4f\n", r_epv_ebv)
    end

    # --- Read NN weights to check if omics effects are small ---
    if out_dir !== nothing
        nn_file = joinpath(out_dir, "MCMC_samples_neural_networks_bias_and_weights.txt")
        if isfile(nn_file)
            nn_samples = readdlm(nn_file, ',', header=false)
            nn_mean = vec(mean(nn_samples, dims=1))
            nn_std  = vec(std(nn_samples, dims=1))
            println("\n  Neural Network Weights (α₂, omics→phenotype):")
            println("    Posterior mean: ", round.(nn_mean, digits=6))
            println("    Posterior std:  ", round.(nn_std, digits=6))
            println("    max|mean|:      ", round(maximum(abs.(nn_mean)), digits=6))
        end
    end

    # --- Read skip EBV components ---
    ebv_indirect_key = "EBV_Indirect_NonLinear"
    ebv_skip_key     = "EBV_Direct_Skip"
    if haskey(result, ebv_indirect_key) || haskey(result, ebv_skip_key)
        # Read from files
        indirect_file = joinpath(out_dir, "MCMC_samples_EBV_Indirect_NonLinear.txt")
        skip_file     = joinpath(out_dir, "MCMC_samples_EBV_Direct_Skip.txt")
        if isfile(indirect_file) && isfile(skip_file)
            indirect_samples = readdlm(indirect_file, ',', header=true)[1]
            skip_samples     = readdlm(skip_file, ',', header=true)[1]
            indirect_mean = vec(mean(indirect_samples, dims=1))
            skip_mean     = vec(mean(skip_samples, dims=1))
            var_indirect = var(indirect_mean)
            var_skip     = var(skip_mean)
            var_total    = var(indirect_mean .+ skip_mean)
            println("\n  Variance decomposition of EBV:")
            @printf("    Var(EBV_Indirect):  %.4f  (%.1f%%)\n", var_indirect, 100 * var_indirect / var_total)
            @printf("    Var(EBV_Direct):    %.4f  (%.1f%%)\n", var_skip, 100 * var_skip / var_total)
            @printf("    Var(EBV_Total):     %.4f\n", var_total)
        end
    end
end

# ============================================================
# Run Model A: No skip connection
# ============================================================
println("\n" * "="^70)
println("MODEL A: No skip connection")
println("  phenotypes = intercept + omics")
println("="^70)
result_noskip, dir_noskip = run_model(use_skip=false, output_name="noskip")
compute_accuracies(result_noskip, "Model A: No Skip"; out_dir=dir_noskip)

# ============================================================
# Run Model B: With skip connection
# ============================================================
println("\n" * "="^70)
println("MODEL B: With skip connection (direct geno→pheno)")
println("  phenotypes = intercept + omics + geno")
println("="^70)
result_skip, dir_skip = run_model(use_skip=true, output_name="skip")
compute_accuracies(result_skip, "Model B: With Skip"; out_dir=dir_skip)

# ============================================================
# Summary comparison
# ============================================================
println("\n" * "="^70)
println("SUMMARY: EPV vs EBV accuracy on TEST individuals")
println("  (test individuals have OBSERVED omics + genotypes)")
println("="^70)

function get_test_acc(result, id_set)
    ebv_df = result["EBV_NonLinear"]
    ebv_df.ID = string.(ebv_df.ID)
    epv_df = result["EPV_Output_NonLinear"]
    epv_df.ID = string.(epv_df.ID)
    truth = pheno_df[:, [:ID, :genetic_total, :trait1]]

    me = innerjoin(ebv_df, truth, on=:ID)
    me = filter(r -> r.ID in id_set, me)
    mp = innerjoin(epv_df, truth, on=:ID)
    mp = filter(r -> r.ID in id_set, mp)
    both = innerjoin(me[:, [:ID, :EBV]], mp[:, [:ID, :EPV]], on=:ID)

    return (
        ebv_total = cor(me.EBV, me.genetic_total),
        epv_total = cor(mp.EPV, mp.genetic_total),
        ebv_trait = cor(me.EBV, me.trait1),
        epv_trait = cor(mp.EPV, mp.trait1),
        r_epv_ebv = cor(both.EBV, both.EPV),
    )
end

acc_noskip = get_test_acc(result_noskip, test_id_set)
acc_skip   = get_test_acc(result_skip, test_id_set)

println()
@printf("%-35s  %-10s  %-10s\n", "", "No Skip", "With Skip")
println("-"^60)
@printf("%-35s  %+.4f     %+.4f\n", "EBV cor(., genetic_total)", acc_noskip.ebv_total, acc_skip.ebv_total)
@printf("%-35s  %+.4f     %+.4f\n", "EPV cor(., genetic_total)", acc_noskip.epv_total, acc_skip.epv_total)
@printf("%-35s  %+.4f     %+.4f\n", "Δ(EPV - EBV) genetic_total", acc_noskip.epv_total - acc_noskip.ebv_total, acc_skip.epv_total - acc_skip.ebv_total)
println("-"^60)
@printf("%-35s  %+.4f     %+.4f\n", "EBV cor(., trait1)", acc_noskip.ebv_trait, acc_skip.ebv_trait)
@printf("%-35s  %+.4f     %+.4f\n", "EPV cor(., trait1)", acc_noskip.epv_trait, acc_skip.epv_trait)
@printf("%-35s  %+.4f     %+.4f\n", "Δ(EPV - EBV) trait1", acc_noskip.epv_trait - acc_noskip.ebv_trait, acc_skip.epv_trait - acc_skip.ebv_trait)
println("-"^60)
@printf("%-35s  %.4f     %.4f\n", "cor(EBV, EPV)", acc_noskip.r_epv_ebv, acc_skip.r_epv_ebv)

println("\nInterpretation:")
delta_noskip = acc_noskip.epv_total - acc_noskip.ebv_total
delta_skip   = acc_skip.epv_total - acc_skip.ebv_total
if abs(delta_skip) < 0.01
    println("  ✓ With skip:    EPV ≈ EBV (Δ = $(round(delta_skip, digits=4))) — skip absorbs genetic signal")
else
    println("  ✗ With skip:    EPV ≠ EBV (Δ = $(round(delta_skip, digits=4))) — omics still contribute")
end
if delta_noskip > 0.01
    println("  ✓ Without skip: EPV > EBV (Δ = $(round(delta_noskip, digits=4))) — observed omics add information")
else
    println("  ✗ Without skip: EPV ≈ EBV (Δ = $(round(delta_noskip, digits=4))) — unexpected")
end

println("\n" * "="^70)
println("Validation complete!")
println("="^70)

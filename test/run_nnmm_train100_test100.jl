#!/usr/bin/env julia

using NNMM
using NNMM.Datasets
using CSV
using DataFrames
using Random
using Statistics
using Dates

const SEED = 42
const CHAIN_LENGTH = 1000
const BURNIN = 200
const TEST_FRAC = 0.2
const TRAIN_MISSING_PCT = 1.0
const TEST_MISSING_PCT = 1.0
const N_OMICS = 10

root_dir = joinpath(@__DIR__, "NNMM_100_100_fix")
inputs_dir = joinpath(root_dir, "inputs")
mkpath(inputs_dir)

run_stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
output_dir = joinpath(root_dir, "output_" * run_stamp)

geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)
pheno_df.ID = string.(pheno_df.ID)

n_individuals = nrow(pheno_df)
omics_syms = [Symbol("omic$i") for i in 1:N_OMICS]

# Train/test split (by individual)
rng_split = MersenneTwister(SEED)
perm = shuffle(rng_split, collect(1:n_individuals))
n_test = round(Int, n_individuals * TEST_FRAC)
test_idx = perm[1:n_test]
train_idx = perm[n_test+1:end]
test_id_set = Set(pheno_df.ID[test_idx])

println("Repro config:")
println("  seed=$SEED chain_length=$CHAIN_LENGTH burnin=$BURNIN test_frac=$TEST_FRAC")
println("  train_missing_pct=$TRAIN_MISSING_PCT test_missing_pct=$TEST_MISSING_PCT")
println("  n_train=$(length(train_idx)) n_test=$(length(test_idx))")

# Omics with missing
omics_df = copy(pheno_df[:, vcat([:ID], omics_syms)])
allowmissing!(omics_df, Not(:ID))

if TRAIN_MISSING_PCT == 1.0
    omics_df[train_idx, omics_syms] .= missing
else
    error("This repro script is fixed to TRAIN_MISSING_PCT=1.0; edit constants if needed.")
end
if TEST_MISSING_PCT == 1.0
    omics_df[test_idx, omics_syms] .= missing
else
    error("This repro script is fixed to TEST_MISSING_PCT=1.0; edit constants if needed.")
end

omics_path = joinpath(inputs_dir, "omics_train100_test100.csv")
CSV.write(omics_path, omics_df; missingstring="NA")

# Phenotypes: mask test
pheno_out_df = copy(pheno_df[:, [:ID, :trait1]])
allowmissing!(pheno_out_df, :trait1)
pheno_out_df[test_idx, :trait1] .= missing
pheno_out_path = joinpath(inputs_dir, "phenotypes_mask_test.csv")
CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")

layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
    Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA"),
]

equations = [
    Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = intercept + geno",
        traits=["omic$i" for i in 1:N_OMICS],
        method="BayesC",
        estimatePi=true,
    ),
    Equation(
        from_layer_name="omics",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + omics",
        traits=["trait1"],
        method="BayesC",
        activation_function="linear",
    ),
]

start_time = time()
result = runNNMM(
    layers,
    equations;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    seed=SEED,
    output_folder=output_dir,
)
elapsed = time() - start_time

ebv_df = result["EBV_NonLinear"]
ebv_df.ID = string.(ebv_df.ID)
ebv_test = ebv_df[in.(ebv_df.ID, Ref(test_id_set)), :]
truth_test = pheno_df[test_idx, [:ID, :genetic_total, :genetic_direct, :genetic_indirect]]
merged = innerjoin(ebv_test, truth_test, on=:ID)

ebv_test_total = cor(merged.EBV, merged.genetic_total)
ebv_test_direct = cor(merged.EBV, merged.genetic_direct)
ebv_test_indirect = cor(merged.EBV, merged.genetic_indirect)

epv_df = result["EPV_Output_NonLinear"]
epv_df.ID = string.(epv_df.ID)
epv_test = epv_df[in.(epv_df.ID, Ref(test_id_set)), :]
truth_test_epv = pheno_df[test_idx, [:ID, :genetic_total, :genetic_direct, :genetic_indirect, :trait1]]
merged_epv = innerjoin(epv_test, truth_test_epv, on=:ID)

epv_test_total = cor(merged_epv.EPV, merged_epv.genetic_total)
epv_test_direct = cor(merged_epv.EPV, merged_epv.genetic_direct)
epv_test_indirect = cor(merged_epv.EPV, merged_epv.genetic_indirect)
epv_test_trait = cor(merged_epv.EPV, merged_epv.trait1)

println()
println("Results:")
println("  EBV(test) cor(total)  = $(round(ebv_test_total, digits=4))")
println("  EBV(test) cor(direct) = $(round(ebv_test_direct, digits=4))")
println("  EBV(test) cor(indir)  = $(round(ebv_test_indirect, digits=4))")
println("  EPV(test) cor(total)  = $(round(epv_test_total, digits=4))")
println("  EPV(test) cor(direct) = $(round(epv_test_direct, digits=4))")
println("  EPV(test) cor(indir)  = $(round(epv_test_indirect, digits=4))")
println("  EPV(test) cor(trait1) = $(round(epv_test_trait, digits=4))")
println("  runtime_seconds       = $(round(elapsed, digits=2))")
println()
println("Outputs:")
println("  inputs_dir  = $inputs_dir")
println("  output_dir  = $output_dir")

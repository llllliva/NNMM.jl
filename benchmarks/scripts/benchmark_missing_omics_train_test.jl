#!/usr/bin/env julia
#=
NNMM Benchmark Script: Missing Omics in Train vs Test

Goal
----
Evaluate NNMM when omics (layer-2) are missing at different rates in:
  - training individuals (phenotypes observed)
  - testing individuals (phenotypes missing), with test omics either fully observed (0%) or fully missing (100%)

Notes
-----
- Test individuals always have genotypes (EBV is output for all genotyped IDs).
- By NNMM design, individuals with missing phenotypes but observed omics may still
  contribute to the 1->2 (geno->omics) equation; this benchmark keeps that behavior.
- EPV is evaluated on test individuals using `EPV_Output_NonLinear` (computed for all output IDs).

Usage
-----
    julia --project=. benchmarks/benchmark_missing_omics_train_test.jl

Common overrides:
    julia --project=. benchmarks/benchmark_missing_omics_train_test.jl \
      --seed=42 --chain-length=1000 --burnin=200 \
      --test-frac=0.2 --missing-mode=individual \
      --train-missing-grid=0:0.1:1 --test-missing-pcts=0,1
=#

using NNMM
using NNMM.Datasets
using CSV
using DataFrames
using Dates
using Random
using Statistics

println("=" ^ 70)
println("NNMM Benchmark: Missing Omics in Train vs Test")
println("=" ^ 70)
println("Date: ", Dates.now())
println()

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

function parse_grid(val; default_grid = collect(0.0:0.1:1.0))
    sval = strip(String(val))
    if isempty(sval)
        return default_grid
    end
    if occursin(":", sval)
        parts = split(sval, ":")
        if length(parts) == 3
            a = parse(Float64, strip(parts[1]))
            step = parse(Float64, strip(parts[2]))
            b = parse(Float64, strip(parts[3]))
            return collect(a:step:b)
        end
        error("Invalid range syntax for grid: \"$sval\" (expected start:step:stop)")
    end
    return parse_csv_floats(sval)
end

function normalize_pct_list(pcts)
    out = unique([clamp(p, 0.0, 1.0) for p in pcts])
    sort!(out)
    return out
end

const SEED = parse(Int, get_arg("--seed", "42"))
const CHAIN_LENGTH = parse(Int, get_arg("--chain-length", "1000"))
const BURNIN = parse(Int, get_arg("--burnin", "200"))
const TEST_FRAC = parse(Float64, get_arg("--test-frac", "0.2"))
const MISSING_MODE = lowercase(String(get_arg("--missing-mode", "individual"))) # individual | cell

const TRAIN_MISSING_PCTS = normalize_pct_list(parse_grid(get_arg("--train-missing-grid", "0:0.1:1")))
const TEST_MISSING_PCTS = normalize_pct_list(parse_grid(get_arg("--test-missing-pcts", "0,1"); default_grid=[0.0, 1.0]))

const KEEP_OUTPUT = lowercase(String(get_arg("--keep-output", "false"))) in ("1", "true", "yes", "y")

if !(0.0 < TEST_FRAC < 1.0)
    error("--test-frac must be in (0,1), got $TEST_FRAC")
end
if !(MISSING_MODE in ("individual", "cell"))
    error("--missing-mode must be \"individual\" or \"cell\", got \"$MISSING_MODE\"")
end

println("Configuration:")
println("  Seed: $SEED")
println("  Chain length: $CHAIN_LENGTH")
println("  Burnin: $BURNIN")
println("  Test fraction: $(round(TEST_FRAC, digits=3))")
println("  Missing mode (train): $MISSING_MODE")
println("  Train missing grid: ", join(Int.(round.(TRAIN_MISSING_PCTS .* 100)), "%, "), "%")
println("  Test missing pcts: ", join(Int.(round.(TEST_MISSING_PCTS .* 100)), "%, "), "%")
println("  Keep temp outputs: $KEEP_OUTPUT")
println()

# Load simulated dataset
println("Loading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

pheno_df.ID = string.(pheno_df.ID)

n_individuals = nrow(pheno_df)
n_omics = 10

println("  Individuals: $n_individuals")
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: $n_omics")
println()

# Train/test split (by individual)
rng_split = MersenneTwister(SEED)
perm = shuffle(rng_split, collect(1:n_individuals))
n_test = round(Int, n_individuals * TEST_FRAC)
test_idx = perm[1:n_test]
train_idx = perm[n_test+1:end]

train_ids = pheno_df.ID[train_idx]
test_ids = pheno_df.ID[test_idx]
test_id_set = Set(test_ids)

println("Split:")
println("  Train individuals: $(length(train_idx))")
println("  Test individuals:  $(length(test_idx))")
println()

# Precompute nested missingness selections for reproducibility
train_perm = shuffle(MersenneTwister(SEED + 1), train_idx)
test_perm = shuffle(MersenneTwister(SEED + 2), test_idx)

train_perm_by_col = Vector{Vector{Int}}(undef, n_omics)
test_perm_by_col = Vector{Vector{Int}}(undef, n_omics)
for j in 1:n_omics
    train_perm_by_col[j] = shuffle(MersenneTwister(SEED + 100 + j), train_idx)
    test_perm_by_col[j] = shuffle(MersenneTwister(SEED + 200 + j), test_idx)
end

function apply_missingness!(omics_df, omic_syms, pct, idx, perm, perm_by_col, mode)
    n = length(idx)
    n_to_miss = round(Int, n * pct)
    if n_to_miss <= 0
        return 0
    end
    if mode == "cell"
        for (j, col) in enumerate(omic_syms)
            miss_idx = perm_by_col[j][1:n_to_miss]
            omics_df[miss_idx, col] .= missing
        end
        return n_to_miss * length(omic_syms)
    end
    miss_idx = perm[1:n_to_miss]
    for col in omic_syms
        omics_df[miss_idx, col] .= missing
    end
    return n_to_miss * length(omic_syms)
end

# Results
results = DataFrame(
    seed = Int[],
    chain_length = Int[],
    burnin = Int[],
    test_frac = Float64[],
    missing_mode = String[],
    train_missing_pct = Float64[],
    test_missing_pct = Float64[],
    n_train = Int[],
    n_test = Int[],
    train_missing_cells = Int[],
    test_missing_cells = Int[],
    ebv_test_total = Float64[],
    ebv_test_direct = Float64[],
    ebv_test_indirect = Float64[],
    epv_test_total = Float64[],
    epv_test_direct = Float64[],
    epv_test_indirect = Float64[],
    epv_test_trait = Float64[],
    time_seconds = Float64[]
)

omics_syms = [Symbol("omic$i") for i in 1:n_omics]
omics_cols = vcat([:ID], omics_syms)

println("Running grid: $(length(TRAIN_MISSING_PCTS)) train levels × $(length(TEST_MISSING_PCTS)) test levels")
println("-" ^ 70)

for train_missing_pct in TRAIN_MISSING_PCTS
    for test_missing_pct in TEST_MISSING_PCTS
        println("Train missing $(Int(round(train_missing_pct * 100)))% | Test missing $(Int(round(test_missing_pct * 100)))%")

        benchmark_dir = mktempdir()
        try
            # Omics with missing
            omics_df = copy(pheno_df[:, omics_cols])
            allowmissing!(omics_df, Not(:ID))

            train_missing_cells = apply_missingness!(
                omics_df, omics_syms, train_missing_pct, train_idx, train_perm, train_perm_by_col, MISSING_MODE
            )
            test_missing_cells = apply_missingness!(
                omics_df, omics_syms, test_missing_pct, test_idx, test_perm, test_perm_by_col, "individual"
            )

            omics_path = joinpath(benchmark_dir, "omics.csv")
            CSV.write(omics_path, omics_df; missingstring="NA")

            # Phenotypes: mask test
            pheno_out_df = copy(pheno_df[:, [:ID, :trait1]])
            allowmissing!(pheno_out_df, :trait1)
            pheno_out_df[test_idx, :trait1] .= missing

            pheno_out_path = joinpath(benchmark_dir, "phenotypes.csv")
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
                    omics_name=["omic$i" for i in 1:n_omics],
                    method="BayesC",
                    estimatePi=true,
                ),
                Equation(
                    from_layer_name="omics",
                    to_layer_name="phenotypes",
                    equation="phenotypes = intercept + omics",
                    phenotype_name=["trait1"],
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
                output_folder=joinpath(benchmark_dir, "output"),
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

            push!(
                results,
                (
                    SEED,
                    CHAIN_LENGTH,
                    BURNIN,
                    TEST_FRAC,
                    MISSING_MODE,
                    train_missing_pct,
                    test_missing_pct,
                    length(train_idx),
                    length(test_idx),
                    train_missing_cells,
                    test_missing_cells,
                    ebv_test_total,
                    ebv_test_direct,
                    ebv_test_indirect,
                    epv_test_total,
                    epv_test_direct,
                    epv_test_indirect,
                    epv_test_trait,
                    elapsed,
                ),
            )

            println(
                "  EBV(test) cor(total) = $(round(ebv_test_total, digits=4)) | ",
                "EPV(test) cor(total) = $(round(epv_test_total, digits=4)) | ",
                "time = $(round(elapsed, digits=1))s",
            )
        finally
            if KEEP_OUTPUT
                println("  Kept temp dir: $benchmark_dir")
            else
                rm(benchmark_dir, recursive=true)
            end
        end
    end
end

println()
println("=" ^ 70)
println("SUMMARY (EBV on TEST; EPV on TEST)")
println("=" ^ 70)

println("┌───────────────┬──────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬──────────┐")
println("│ Train Missing │ Test Missing │ EBV(test,total) │ EBV(test,indir) │ EPV(test,total) │ EPV(test,trait) │ Time (s) │")
println("├───────────────┼──────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤")
for row in eachrow(results)
    println(
        "│ ",
        lpad("$(Int(round(row.train_missing_pct * 100)))%", 11),
        "     │ ",
        lpad("$(Int(round(row.test_missing_pct * 100)))%", 10),
        "     │ ",
        lpad(round(row.ebv_test_total, digits=4), 15),
        " │ ",
        lpad(round(row.ebv_test_indirect, digits=4), 15),
        " │ ",
        lpad(round(row.epv_test_total, digits=4), 15),
        " │ ",
        lpad(round(row.epv_test_trait, digits=4), 15),
        " │ ",
        lpad(round(row.time_seconds, digits=1), 8),
        " │",
    )
end
println("└───────────────┴──────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┴──────────┘")
println()

out_csv = joinpath(@__DIR__, "missing_omics_train_test_results.csv")
CSV.write(out_csv, results)
println("Saved results to: $out_csv")
println()

#!/usr/bin/env julia
#=
Experiment: Skip vs No-Skip, Latent Nodes, Missingness, Convergence

Uses real data from TempTestData and runs a factorial experiment to study:
  - skip vs no-skip
  - number of latent traits (fully missing)
  - missingness in observed omics
  - chain length (convergence proxy)
  - seed (stability)
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra, StatsBase
using NNMM
using JWAS

const DATA_DIR = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/nnmm_small_dataset/input_files/data1"
const OUT_DIR = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData"

println("="^80)
println("NNMM Skip Convergence Experiment")
println("="^80)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
geno_df = CSV.read(joinpath(DATA_DIR, "geno_rep1.csv"), DataFrame)
omics_df = CSV.read(joinpath(DATA_DIR, "omics_rep1_miss_0pct.csv"), DataFrame)
if "residual" in names(omics_df)
    select!(omics_df, Not(:residual))
end
omics_names = names(omics_df)[2:end]

phen_trn_df = CSV.read(joinpath(DATA_DIR, "phen_rep1_trn.csv"), DataFrame, types=Dict(:FIP => Float64))
phen_val_df = CSV.read(joinpath(DATA_DIR, "phen_rep1_val.csv"), DataFrame, types=Dict(:FIP => Float64))
phen_all_df = vcat(phen_trn_df, phen_val_df)

train_ids = Set(string.(CSV.read(joinpath(DATA_DIR, "ID_rep1_trn.csv"), DataFrame).ID))
val_ids = Set(string.(CSV.read(joinpath(DATA_DIR, "ID_rep1_val.csv"), DataFrame).ID))

all_ids = string.(geno_df.ID)
n = length(all_ids)

phen_dict = Dict(string(row.ID) => row.FIP for row in eachrow(phen_all_df))
omics_dict = Dict(string(row.ID) => collect(row)[2:end] for row in eachrow(omics_df))

println("Genotypes: $(nrow(geno_df)) individuals, $(ncol(geno_df)-1) SNPs")
println("Omics: $(nrow(omics_df)) individuals, $(length(omics_names)) features")
println("Training: $(length(train_ids)), Validation: $(length(val_ids))")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
function prepare_omics_file(all_ids, omics_dict, omics_names, n_latent, missing_pct; seed=12345)
    Random.seed!(seed)
    n = length(all_ids)
    result_df = DataFrame(ID = all_ids)
    trait_names = String[]

    # Add latent traits (all missing)
    for i in 1:n_latent
        latent_name = "latent$i"
        result_df[!, latent_name] = Vector{Union{Missing, Float64}}(fill(missing, n))
        push!(trait_names, latent_name)
    end

    # Add observed omics (with optional missing pattern)
    for oname in omics_names
        vals = Vector{Union{Missing, Float64}}(undef, n)
        for (i, id) in enumerate(all_ids)
            if haskey(omics_dict, id)
                idx = findfirst(==(oname), omics_names)
                vals[i] = omics_dict[id][idx]
            else
                vals[i] = missing
            end
        end
        if missing_pct > 0
            n_miss = round(Int, n * missing_pct / 100)
            valid_idx = findall(!ismissing, vals)
            if length(valid_idx) > n_miss
                miss_idx = sample(valid_idx, n_miss, replace=false)
                vals[miss_idx] .= missing
            end
        end
        result_df[!, oname] = vals
        push!(trait_names, oname)
    end

    return result_df, trait_names
end

function run_nnmm(geno_path, omics_path, pheno_path, trait_names, use_skip, output_dir; chain_length=500, burnin=100, seed=12345)
    layer1 = Layer(layer_name="Genotypes", data_path=geno_path, header=true)
    layer2 = Layer(layer_name="MiddleLayer", data_path=omics_path, header=true, missing_value="NA")
    layer3 = Layer(layer_name="Phenotypes", data_path=pheno_path, header=true, missing_value="NA")

    eq1 = Equation(
        from_layer_name="Genotypes", to_layer_name="MiddleLayer",
        equation="MiddleLayer = intercept + Genotypes",
        traits=trait_names,
        method="BayesC"
    )

    if use_skip
        eq2 = Equation(
            from_layer_name="MiddleLayer", to_layer_name="Phenotypes",
            equation="Phenotypes = intercept + MiddleLayer + Genotypes",
            traits=["y"],
            activation_function="linear",
            class_priors=Dict(
                "MiddleLayer" => (method="BayesC", Pi=0.0),
                "Genotypes"   => (method="BayesC", Pi=0.95)
            )
        )
    else
        eq2 = Equation(
            from_layer_name="MiddleLayer", to_layer_name="Phenotypes",
            equation="Phenotypes = intercept + MiddleLayer",
            traits=["y"],
            activation_function="linear"
        )
    end

    return runNNMM([layer1, layer2, layer3], [eq1, eq2],
        chain_length=chain_length, burnin=burnin, output_folder=output_dir, seed=seed)
end

function run_jwas(geno_path, omics_path, pheno_path, use_skip, output_dir; chain_length=500, burnin=100, seed=12345)
    geno = JWAS.get_genotypes(geno_path, separator=',', header=true)
    omics_class = JWAS.get_genotypes(omics_path, separator=',', header=true, quality_control=false)
    omics_var = var(omics_class.genotypes, dims=1)
    omics_class.sum2pq = max(float(sum(omics_var)), eps(Float64))

    Main.geno = geno
    Main.omics_class = omics_class

    if use_skip
        mme = JWAS.build_model("y = intercept + geno + omics_class")
    else
        mme = JWAS.build_model("y = intercept + omics_class")
    end

    jwas_data = CSV.read(pheno_path, DataFrame, missingstring="NA")
    return JWAS.runMCMC(mme, jwas_data,
        chain_length=chain_length, burnin=burnin, output_folder=output_dir, seed=seed, outputEBV=true)
end

function compute_metrics(nnmm_out, jwas_out, phen_dict, train_ids, val_ids)
    epv_df = nnmm_out["EPV_Output_NonLinear"]
    ebv_df = nnmm_out["EBV_NonLinear"]
    ebv_jwas_df = jwas_out["EBV_y"]

    epv = Dict(string(k) => v for (k, v) in zip(epv_df.ID, epv_df.EPV))
    ebv = Dict(string(k) => v for (k, v) in zip(ebv_df.ID, ebv_df.EBV))
    ebv_jwas = Dict(string(k) => v for (k, v) in zip(ebv_jwas_df.ID, ebv_jwas_df.EBV))

    common = [id for id in keys(phen_dict) if haskey(epv, id) && haskey(ebv, id) && haskey(ebv_jwas, id)]
    train_common = [id for id in common if id in train_ids]
    val_common = [id for id in common if id in val_ids]

    function cor_safe(a, b)
        length(a) > 2 ? cor(a, b) : NaN
    end

    epv_tr = Float64[epv[id] for id in train_common]
    ebv_tr = Float64[ebv[id] for id in train_common]
    y_tr = Float64[phen_dict[id] for id in train_common]
    ebv_j_tr = Float64[ebv_jwas[id] for id in train_common]

    epv_te = Float64[epv[id] for id in val_common]
    ebv_te = Float64[ebv[id] for id in val_common]
    y_te = Float64[phen_dict[id] for id in val_common]
    ebv_j_te = Float64[ebv_jwas[id] for id in val_common]

    return (
        train_epv = cor_safe(epv_tr, y_tr),
        test_epv = cor_safe(epv_te, y_te),
        train_ebv = cor_safe(ebv_tr, y_tr),
        test_ebv = cor_safe(ebv_te, y_te),
        r_epv_ebv_tr = cor_safe(epv_tr, ebv_tr),
        r_epv_ebv_te = cor_safe(epv_te, ebv_te),
        train_ebv_jwas = cor_safe(ebv_j_tr, y_tr),
        test_ebv_jwas = cor_safe(ebv_j_te, y_te),
        r_nnmm_jwas_tr = cor_safe(ebv_tr, ebv_j_tr),
        r_nnmm_jwas_te = cor_safe(ebv_te, ebv_j_te),
    )
end

# ------------------------------------------------------------------
# Experiment design
# ------------------------------------------------------------------
skips = [false, true]
n_latents = [0, 1, 3]
missing_pcts = [0, 50]
chains = [(500, 100), (2000, 500)]
seeds = [12345, 54321]

results = DataFrame(
    skip = Bool[],
    n_latent = Int[],
    missing_pct = Int[],
    chain_length = Int[],
    burnin = Int[],
    seed = Int[],
    train_epv = Float64[],
    test_epv = Float64[],
    train_ebv = Float64[],
    test_ebv = Float64[],
    r_epv_ebv_tr = Float64[],
    r_epv_ebv_te = Float64[],
    train_ebv_jwas = Float64[],
    test_ebv_jwas = Float64[],
    r_nnmm_jwas_tr = Float64[],
    r_nnmm_jwas_te = Float64[],
)

run_idx = 0
total_runs = length(skips) * length(n_latents) * length(missing_pcts) * length(chains) * length(seeds)

for use_skip in skips, n_latent in n_latents, missing_pct in missing_pcts, (chain_len, burn) in chains, seed in seeds
    global run_idx += 1
    println("\n[Run $run_idx / $total_runs] skip=$use_skip, n_latent=$n_latent, missing=$missing_pct, chain=$chain_len, seed=$seed")

    tmpdir = mktempdir()
    geno_path = joinpath(tmpdir, "genotypes.csv")
    CSV.write(geno_path, geno_df)

    omics_df_s, trait_names = prepare_omics_file(all_ids, omics_dict, omics_names, n_latent, missing_pct; seed=seed)
    omics_path = joinpath(tmpdir, "omics.csv")
    CSV.write(omics_path, omics_df_s, missingstring="NA")

    y_with_missing = Vector{Union{Missing, Float64}}(undef, n)
    for (i, id) in enumerate(all_ids)
        y_with_missing[i] = id in val_ids ? missing : get(phen_dict, id, missing)
    end
    pheno_df = DataFrame(ID = all_ids, y = y_with_missing)
    pheno_path = joinpath(tmpdir, "phenotypes.csv")
    CSV.write(pheno_path, pheno_df, missingstring="NA")

    nnmm_out = run_nnmm(geno_path, omics_path, pheno_path, trait_names, use_skip, joinpath(tmpdir, "nnmm");
        chain_length=chain_len, burnin=burn, seed=seed)
    jwas_out = run_jwas(geno_path, omics_path, pheno_path, use_skip, joinpath(tmpdir, "jwas");
        chain_length=chain_len, burnin=burn, seed=seed)

    m = compute_metrics(nnmm_out, jwas_out, phen_dict, train_ids, val_ids)

    push!(results, (
        use_skip, n_latent, missing_pct, chain_len, burn, seed,
        m.train_epv, m.test_epv, m.train_ebv, m.test_ebv, m.r_epv_ebv_tr, m.r_epv_ebv_te,
        m.train_ebv_jwas, m.test_ebv_jwas, m.r_nnmm_jwas_tr, m.r_nnmm_jwas_te
    ))
end

results_path = joinpath(OUT_DIR, "experiment_skip_convergence_results.csv")
CSV.write(results_path, results)
println("\nSaved results: $results_path")

# Summary across seeds (mean/std) for each condition
group_cols = [:skip, :n_latent, :missing_pct, :chain_length]
summary = combine(groupby(results, group_cols),
    :train_ebv => mean => :train_ebv_mean,
    :train_ebv => std  => :train_ebv_sd,
    :test_ebv  => mean => :test_ebv_mean,
    :test_ebv  => std  => :test_ebv_sd,
    :r_nnmm_jwas_tr => mean => :r_nnmm_jwas_tr_mean,
    :r_nnmm_jwas_tr => std  => :r_nnmm_jwas_tr_sd,
    :r_nnmm_jwas_te => mean => :r_nnmm_jwas_te_mean,
    :r_nnmm_jwas_te => std  => :r_nnmm_jwas_te_sd,
)

summary_path = joinpath(OUT_DIR, "experiment_skip_convergence_summary.csv")
CSV.write(summary_path, summary)
println("Saved summary: $summary_path")

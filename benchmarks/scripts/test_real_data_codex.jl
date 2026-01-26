#!/usr/bin/env julia
#=
Codex Test: NNMM.jl vs JWAS Multi-Class BayesC using REAL DATA

Goal
- Compare JWAS multi-class BayesC: y = intercept + geno + omics
  vs NNMM: X -> (latent1 missing + omics observed) -> y (linear activation)
- Focus on predictions when omics are 100% observed.

Notes
- `test_real_data.jl` in this repo evaluates only training phenotypes (it reads phen_rep1_trn.csv).
  This Codex variant loads BOTH `phen_rep1_trn.csv` and `phen_rep1_val.csv` so we can evaluate
  validation predictions even though validation phenotypes are set to missing during fitting.
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra, DelimitedFiles

println("="^60)
println("Codex Test: NNMM vs JWAS Multi-Class using REAL DATA")
println("="^60)

Random.seed!(12345)

# ============================================================
# Step 1: Load data
# ============================================================
println("\n--- Step 1: Loading data ---")

data_dir = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/nnmm_small_dataset/input_files/data1"

# Load genotypes
geno_df = CSV.read(joinpath(data_dir, "geno_rep1.csv"), DataFrame)
println("Genotypes: $(nrow(geno_df)) individuals, $(ncol(geno_df)-1) SNPs")

# Load omics (0% missing = all observed)
omics_df = CSV.read(joinpath(data_dir, "omics_rep1_miss_0pct.csv"), DataFrame)
if "residual" in names(omics_df)
    select!(omics_df, Not(:residual))
end
println("Omics: $(nrow(omics_df)) individuals, $(ncol(omics_df)-1) features")

# Load phenotypes (train + validation) for evaluation
phen_trn_df = CSV.read(joinpath(data_dir, "phen_rep1_trn.csv"), DataFrame;
    missingstring="NA",
    silencewarnings=true,
    types=Dict(:FIP => Union{Missing, Float64}),
)
phen_val_df = CSV.read(joinpath(data_dir, "phen_rep1_val.csv"), DataFrame;
    missingstring="NA",
    silencewarnings=true,
    types=Dict(:FIP => Union{Missing, Float64}),
)
phen_all_df = vcat(phen_trn_df, phen_val_df)
println("Phenotypes (for evaluation): $(nrow(phen_all_df)) individuals ($(nrow(phen_trn_df)) train + $(nrow(phen_val_df)) val)")

# Load train/val IDs
train_ids = CSV.read(joinpath(data_dir, "ID_rep1_trn.csv"), DataFrame).ID
val_ids = CSV.read(joinpath(data_dir, "ID_rep1_val.csv"), DataFrame).ID
println("Training IDs: $(length(train_ids)), Validation IDs: $(length(val_ids))")

# ============================================================
# Step 2: Prepare data for models
# ============================================================
println("\n--- Step 2: Preparing data ---")

tmpdir = mktempdir()
println("Temp dir: $tmpdir")

all_ids = geno_df.ID
n = length(all_ids)

train_ids_set = Set(train_ids)
val_ids_set = Set(val_ids)

train_idx = [id in train_ids_set for id in all_ids]
val_idx = [id in val_ids_set for id in all_ids]
println("Train count: $(sum(train_idx)), Val count: $(sum(val_idx))")

# Save genotypes (unchanged)
geno_path = joinpath(tmpdir, "genotypes.csv")
CSV.write(geno_path, geno_df)

# Phenotypes used for fitting: validation set is missing
phen_dict = Dict(row.ID => row.FIP for row in eachrow(phen_all_df))
y_with_missing = Vector{Union{Missing, Float64}}(undef, n)
for (i, id) in enumerate(all_ids)
    if id in val_ids_set
        y_with_missing[i] = missing
    else
        y_with_missing[i] = get(phen_dict, id, missing)
    end
end
pheno_df = DataFrame(ID = string.(all_ids), y = y_with_missing)
pheno_path = joinpath(tmpdir, "phenotypes.csv")
CSV.write(pheno_path, pheno_df, missingstring="NA")

# Omics file for NNMM: latent node is all missing; observed omics fully observed
omics_dict = Dict(row.ID => collect(row)[2:end] for row in eachrow(omics_df))
omics_names = names(omics_df)[2:end]

latent_omics_df = DataFrame(ID = string.(all_ids))
latent_omics_df[!, "latent1"] = Vector{Union{Missing, Float64}}(fill(missing, n))
for (j, oname) in enumerate(omics_names)
    vals = Vector{Union{Missing, Float64}}(undef, n)
    for (i, id) in enumerate(all_ids)
        vals[i] = haskey(omics_dict, id) ? omics_dict[id][j] : missing
    end
    latent_omics_df[!, oname] = vals
end

latent_omics_path = joinpath(tmpdir, "latent_omics.csv")
CSV.write(latent_omics_path, latent_omics_df, missingstring="NA")
println("Omics columns in NNMM middle layer: $(names(latent_omics_df)[2:end])")

# ============================================================
# Step 3: Run NNMM.jl
# ============================================================
println("\n--- Step 3: Running NNMM.jl ---")

using NNMM

layer1 = Layer(layer_name="Genotypes", data_path=geno_path, header=true)
layer2 = Layer(layer_name="MiddleLayer", data_path=latent_omics_path, header=true, missing_value="NA")
layer3 = Layer(layer_name="Phenotypes", data_path=pheno_path, header=true, missing_value="NA")

# Use a small subset by default for speed; set NNMM_USE_ALL_OMICS=1 to use all features.
use_all_omics = get(ENV, "NNMM_USE_ALL_OMICS", "0") == "1"
all_traits = use_all_omics ? vcat(["latent1"], omics_names) : ["latent1", omics_names[1]]
println("L2 traits used: $(all_traits)")

# Enable Layer 1 -> Layer 3 skip connection to match JWAS multi-class BayesC:
#   y = intercept + MiddleLayer + Genotypes
use_skip = get(ENV, "NNMM_USE_SKIP", "1") == "1"
println("Use skip connection (Genotypes -> Phenotypes): $(use_skip)")

mcmc_chain_length = parse(Int, get(ENV, "MCMC_CHAIN_LENGTH", "2000"))
mcmc_burnin = parse(Int, get(ENV, "MCMC_BURNIN", string(min(500, mcmc_chain_length ÷ 4))))
println("MCMC settings: chain_length=$(mcmc_chain_length), burnin=$(mcmc_burnin)")

eq1 = Equation(
    from_layer_name="Genotypes", to_layer_name="MiddleLayer",
    equation="MiddleLayer = intercept + Genotypes",
    traits=all_traits,
    method="BayesC",
)

# Separate priors per marker class (JWAS-style multi-class).
# Keys must match the layer names used in the 2->3 equation ("MiddleLayer" and "Genotypes").
class_priors_23 = Dict(
    "MiddleLayer" => (
        method="BayesC",
        Pi=0.0,
        estimatePi=true,
        G=false,
        G_is_marker_variance=false,
        df_G=4.0,
        estimate_variance_G=true,
        estimate_scale_G=false,
        constraint_G=true,
    ),
    "Genotypes" => (
        method="BayesC",
        Pi=0.0,
        estimatePi=true,
        G=false,
        G_is_marker_variance=false,
        df_G=4.0,
        estimate_variance_G=true,
        estimate_scale_G=false,
        constraint_G=true,
    ),
)
eq2 = Equation(
    from_layer_name="MiddleLayer", to_layer_name="Phenotypes",
    equation=use_skip ? "Phenotypes = intercept + MiddleLayer + Genotypes" : "Phenotypes = intercept + MiddleLayer",
    traits=["y"],
    class_priors=class_priors_23,
    activation_function="linear",
)

output_nnmm = joinpath(tmpdir, "nnmm_out")
out_nnmm = runNNMM([layer1, layer2, layer3], [eq1, eq2],
    chain_length=mcmc_chain_length, burnin=mcmc_burnin, output_folder=output_nnmm, seed=12345)

epv_out_df = out_nnmm["EPV_Output_NonLinear"]     # uses observed omics on output IDs
ebv_nn_df = out_nnmm["EBV_NonLinear"]             # uses genotype-predicted middle layer

epv_out = Dict(string.(epv_out_df.ID) .=> epv_out_df.EPV)
ebv_nn = Dict(string.(ebv_nn_df.ID) .=> ebv_nn_df.EBV)

# ============================================================
# Step 4: Run JWAS Multi-Class BayesC
# ============================================================
println("\n--- Step 4: Running JWAS Multi-Class BayesC ---")

using JWAS

# Save omics as "genotype" file for JWAS; match NNMM feature subset.
omics_geno_path = joinpath(tmpdir, "omics_as_geno.csv")
omics_geno_df = DataFrame(ID = all_ids)
for oname in all_traits
    if oname == "latent1"
        continue
    end
    omics_geno_df[!, oname] = latent_omics_df[!, oname]
end
CSV.write(omics_geno_path, omics_geno_df)

geno = JWAS.get_genotypes(geno_path, separator=',', header=true)
omics_class = JWAS.get_genotypes(omics_geno_path, separator=',', header=true, quality_control=false)
omics_class.sum2pq = max(float(sum(var(omics_class.genotypes, dims=1))), eps(Float64))

mme_jwas = JWAS.build_model("y = intercept + geno + omics_class")
jwas_data = CSV.read(pheno_path, DataFrame, missingstring="NA")

output_jwas = joinpath(tmpdir, "jwas_out")
out_jwas = JWAS.runMCMC(mme_jwas, jwas_data,
    chain_length=mcmc_chain_length, burnin=mcmc_burnin, output_folder=output_jwas, seed=12345, outputEBV=true)

ebv_jwas_df = out_jwas["EBV_y"]
ebv_jwas = Dict(string.(ebv_jwas_df.ID) .=> ebv_jwas_df.EBV)

# ============================================================
# Step 5: Compare (train vs validation)
# ============================================================
println("\n--- Step 5: Comparing Results ---")

phen_dict_eval = Dict(string(row.ID) => row.FIP for row in eachrow(phen_all_df))
train_ids_str = Set(string.(train_ids))
val_ids_str = Set(string.(val_ids))

function eval_block(label::String, ids::Vector{String})
    ids_valid = [id for id in ids if haskey(phen_dict_eval, id) && !ismissing(phen_dict_eval[id]) &&
        haskey(ebv_jwas, id) && !ismissing(ebv_jwas[id]) &&
        haskey(epv_out, id) && !ismissing(epv_out[id]) &&
        haskey(ebv_nn, id) && !ismissing(ebv_nn[id])]
    if isempty(ids_valid)
        println("\n=== $label ===")
        println("No valid IDs to evaluate.")
        return
    end

    y_true = Float64[phen_dict_eval[id] for id in ids_valid]
    y_jwas = Float64[ebv_jwas[id] for id in ids_valid]
    y_nn_epv = Float64[epv_out[id] for id in ids_valid]
    y_nn_ebv = Float64[ebv_nn[id] for id in ids_valid]

    println("\n=== $label ($(length(ids_valid))) ===")
    println("r(JWAS, NNMM EPV_Output) = $(round(cor(y_jwas, y_nn_epv), digits=4))")
    println("r(JWAS, NNMM EBV_NonLinear) = $(round(cor(y_jwas, y_nn_ebv), digits=4))")
    println("Acc JWAS (corr with true y) = $(round(cor(y_jwas, y_true), digits=4))")
    println("Acc NNMM EPV_Output = $(round(cor(y_nn_epv, y_true), digits=4))")
    println("Acc NNMM EBV_NonLinear = $(round(cor(y_nn_ebv, y_true), digits=4))")
end

all_ids_str = string.(all_ids)
train_block_ids = [id for id in all_ids_str if id in train_ids_str]
val_block_ids = [id for id in all_ids_str if id in val_ids_str]

eval_block("TRAINING (in-sample)", train_block_ids)
eval_block("VALIDATION (y missing during fit)", val_block_ids)

println("\nTemp files saved in: $tmpdir")
println("Done!")

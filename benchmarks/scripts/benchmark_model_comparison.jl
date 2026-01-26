#!/usr/bin/env julia
#=
Comprehensive Model Comparison Benchmark

Models compared:
1. G (genotypes only) using JWAS BayesC
2. G+M using JWAS multi-class BayesC  
3. NNMM with Skip (no omics) - genotypes -> phenotypes directly
4. NNMM with Skip (with omics)

Training scenarios:
- Training with different omics missing rates (0%, 30%, 50%, 100%)

Evaluation:
- Validation set evaluation with different omics availability scenarios
- EBV: Estimated Breeding Value (genotype-only prediction, uses predicted/sampled omics)
- EPV: Estimated Phenotypic Value (uses observed omics when available)

Note on evaluation with validation missing rates:
- The model is trained ONCE per training_miss_rate
- Then evaluated on validation set under different omics availability scenarios
- For NNMM: EPV uses observed omics at prediction time; EBV uses genotype-predicted omics

Data: TempTestData/nnmm_small_dataset
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra, DelimitedFiles
using NNMM
using JWAS
using Dates

# ============================================================
# Configuration
# ============================================================
const DATA_DIR = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/nnmm_small_dataset/input_files"

# Parse command-line arguments for quick testing
QUICK_RUN = get(ENV, "QUICK_RUN", "0") == "1"

const REPLICATES = QUICK_RUN ? (1:1) : (1:3)  # Use 1 rep for quick run, 3 for full
# Note: 100% missing rates cause CSV parsing issues, so we use 90% instead
const TRAIN_MISSING_RATES = QUICK_RUN ? [0, 90] : [0, 30, 50, 90]  # Training omics missing rates
const VAL_MISSING_RATES = QUICK_RUN ? [0, 90] : [0, 30, 50, 70, 90]  # Validation omics missing rates

const MCMC_CHAIN_LENGTH = QUICK_RUN ? 500 : 500
const MCMC_BURNIN = QUICK_RUN ? 100 : 100
const RANDOM_SEED = 12345

# Use only a subset of omics features to speed up NNMM
const MAX_OMICS_FEATURES = 5  # Reduce from 20 to 5 for faster runs

# ============================================================
# Helper Functions
# ============================================================

"""Load data for a given replicate"""
function load_data(rep::Int)
    rep_dir = joinpath(DATA_DIR, "data$rep")
    
    # Load genotypes
    geno_df = CSV.read(joinpath(rep_dir, "geno_rep$rep.csv"), DataFrame)
    
    # Load phenotypes (train + validation for evaluation)
    phen_trn_df = CSV.read(joinpath(rep_dir, "phen_rep$(rep)_trn.csv"), DataFrame;
        missingstring="NA", silencewarnings=true,
        types=Dict(:FIP => Union{Missing, Float64}))
    phen_val_df = CSV.read(joinpath(rep_dir, "phen_rep$(rep)_val.csv"), DataFrame;
        missingstring="NA", silencewarnings=true,
        types=Dict(:FIP => Union{Missing, Float64}))
    phen_all_df = vcat(phen_trn_df, phen_val_df)
    
    # Load train/val IDs
    train_ids = CSV.read(joinpath(rep_dir, "ID_rep$(rep)_trn.csv"), DataFrame).ID
    val_ids = CSV.read(joinpath(rep_dir, "ID_rep$(rep)_val.csv"), DataFrame).ID
    
    return (
        geno_df = geno_df,
        phen_all_df = phen_all_df,
        train_ids = train_ids,
        val_ids = val_ids,
        rep_dir = rep_dir,
        rep = rep
    )
end

"""Load omics data with specified missing rate"""
function load_omics(rep_dir::String, rep::Int, miss_rate::Int)
    omics_path = joinpath(rep_dir, "omics_rep$(rep)_miss_$(miss_rate)pct.csv")
    omics_df = CSV.read(omics_path, DataFrame; missingstring="NA", silencewarnings=true)
    if "residual" in names(omics_df)
        select!(omics_df, Not(:residual))
    end
    return omics_df
end

"""Prepare common data files in a temp directory"""
function prepare_data_files(data, tmpdir::String, train_miss_rate::Int, val_miss_rate::Int)
    all_ids = data.geno_df.ID
    n = length(all_ids)
    
    train_ids_set = Set(data.train_ids)
    val_ids_set = Set(data.val_ids)
    
    # Save genotypes
    geno_path = joinpath(tmpdir, "genotypes.csv")
    CSV.write(geno_path, data.geno_df)
    
    # Phenotypes with validation set as missing
    phen_dict = Dict(row.ID => row.FIP for row in eachrow(data.phen_all_df))
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
    
    # Load omics with different missing rates for training and validation
    train_omics_df = load_omics(data.rep_dir, data.rep, train_miss_rate)
    val_omics_df = load_omics(data.rep_dir, data.rep, val_miss_rate)
    
    # Limit to MAX_OMICS_FEATURES for faster processing
    all_omics_names = names(train_omics_df)[2:end]
    omics_names = all_omics_names[1:min(MAX_OMICS_FEATURES, length(all_omics_names))]
    
    # Only extract the features we're using
    train_omics_dict = Dict(row.ID => [row[Symbol(n)] for n in omics_names] for row in eachrow(train_omics_df))
    val_omics_dict = Dict(row.ID => [row[Symbol(n)] for n in omics_names] for row in eachrow(val_omics_df))
    
    # Create combined omics dict: training individuals use train_miss_rate, val use val_miss_rate
    combined_omics_dict = Dict{Any, Vector}()
    default_missing = Vector{Union{Missing, Float64}}(fill(missing, length(omics_names)))
    for id in all_ids
        if id in train_ids_set
            combined_omics_dict[id] = get(train_omics_dict, id, default_missing)
        else
            combined_omics_dict[id] = get(val_omics_dict, id, default_missing)
        end
    end
    
    return (
        geno_path = geno_path,
        pheno_path = pheno_path,
        all_ids = all_ids,
        phen_dict = phen_dict,
        train_ids_set = train_ids_set,
        val_ids_set = val_ids_set,
        omics_dict = combined_omics_dict,
        omics_names = omics_names,
        n = n
    )
end

"""Calculate accuracy (correlation with true phenotype)"""
function calc_accuracy(pred_dict::Dict, true_dict::Dict, ids::Vector)
    valid_ids = [id for id in ids if 
        haskey(pred_dict, string(id)) && !ismissing(pred_dict[string(id)]) &&
        haskey(true_dict, id) && !ismissing(true_dict[id])]
    
    if length(valid_ids) < 3
        return NaN
    end
    
    pred = Float64[pred_dict[string(id)] for id in valid_ids]
    true_vals = Float64[true_dict[id] for id in valid_ids]
    
    return cor(pred, true_vals)
end

# ============================================================
# Model 1: G (genotypes only) using JWAS BayesC
# ============================================================
function run_jwas_geno_only(prepared, tmpdir::String, seed::Int)
    println("  Running JWAS G (genotypes only)...")
    
    # Load genotypes - JWAS needs the variable in Main scope
    @eval Main geno_jwas = JWAS.get_genotypes($(prepared.geno_path), separator=',', header=true)
    mme = JWAS.build_model("y = intercept + geno_jwas")
    
    # Read phenotype data
    jwas_data = CSV.read(prepared.pheno_path, DataFrame, missingstring="NA")
    jwas_data[!, :ID] = string.(jwas_data.ID)
    
    output_dir = joinpath(tmpdir, "jwas_geno_only")
    out = JWAS.runMCMC(mme, jwas_data,
        chain_length=MCMC_CHAIN_LENGTH, burnin=MCMC_BURNIN,
        output_folder=output_dir, seed=seed, outputEBV=true)
    
    ebv_df = out["EBV_y"]
    ebv = Dict(string.(ebv_df.ID) .=> ebv_df.EBV)
    
    return ebv
end

# ============================================================
# Model 2: G+M using JWAS multi-class BayesC
# ============================================================
function run_jwas_multiclass(prepared, tmpdir::String, seed::Int)
    println("  Running JWAS G+M (multi-class BayesC)...")
    
    # Create omics file for JWAS - JWAS cannot handle missing values, so impute with column means
    omics_geno_path = joinpath(tmpdir, "omics_as_geno.csv")
    omics_geno_df = DataFrame(ID = prepared.all_ids)
    
    for (j, oname) in enumerate(prepared.omics_names)
        vals = Vector{Union{Missing, Float64}}(undef, prepared.n)
        for (i, id) in enumerate(prepared.all_ids)
            if haskey(prepared.omics_dict, id)
                vals[i] = prepared.omics_dict[id][j]
            else
                vals[i] = missing
            end
        end
        # Impute missing values with column mean for JWAS
        non_missing_vals = filter(!ismissing, vals)
        col_mean = isempty(non_missing_vals) ? 0.0 : mean(Float64.(non_missing_vals))
        imputed_vals = [ismissing(v) ? col_mean : Float64(v) for v in vals]
        omics_geno_df[!, oname] = imputed_vals
    end
    CSV.write(omics_geno_path, omics_geno_df)
    
    # JWAS needs variables in Main scope
    @eval Main geno_jwas_mc = JWAS.get_genotypes($(prepared.geno_path), separator=',', header=true)
    @eval Main omics_class_jwas = JWAS.get_genotypes($omics_geno_path, separator=',', header=true, quality_control=false)
    Main.omics_class_jwas.sum2pq = max(float(sum(var(Main.omics_class_jwas.genotypes, dims=1))), eps(Float64))
    
    mme = JWAS.build_model("y = intercept + geno_jwas_mc + omics_class_jwas")
    jwas_data = CSV.read(prepared.pheno_path, DataFrame, missingstring="NA")
    
    output_dir = joinpath(tmpdir, "jwas_multiclass")
    out = JWAS.runMCMC(mme, jwas_data,
        chain_length=MCMC_CHAIN_LENGTH, burnin=MCMC_BURNIN,
        output_folder=output_dir, seed=seed, outputEBV=true)
    
    ebv_df = out["EBV_y"]
    ebv = Dict(string.(ebv_df.ID) .=> ebv_df.EBV)
    
    return ebv
end

# ============================================================
# Model 3: NNMM with Skip (no omics)
# ============================================================
function run_nnmm_skip_no_omics(prepared, tmpdir::String, seed::Int)
    println("  Running NNMM Skip (no omics)...")
    
    # Create a minimal middle layer with only latent node (all missing)
    latent_df = DataFrame(ID = string.(prepared.all_ids))
    latent_df[!, "latent1"] = Vector{Union{Missing, Float64}}(fill(missing, prepared.n))
    latent_path = joinpath(tmpdir, "latent_only.csv")
    CSV.write(latent_path, latent_df, missingstring="NA")
    
    layer1 = Layer(layer_name="Genotypes", data_path=prepared.geno_path, header=true)
    layer2 = Layer(layer_name="MiddleLayer", data_path=latent_path, header=true, missing_value="NA")
    layer3 = Layer(layer_name="Phenotypes", data_path=prepared.pheno_path, header=true, missing_value="NA")
    
    eq1 = Equation(
        from_layer_name="Genotypes", to_layer_name="MiddleLayer",
        equation="MiddleLayer = intercept + Genotypes",
        traits=["latent1"],
        method="BayesC"
    )
    
    # Skip connection: genotypes directly affect phenotypes
    class_priors_23 = Dict(
        "MiddleLayer" => (method="BayesC", Pi=0.0, estimatePi=true, estimate_variance_G=true),
        "Genotypes" => (method="BayesC", Pi=0.0, estimatePi=true, estimate_variance_G=true),
    )
    eq2 = Equation(
        from_layer_name="MiddleLayer", to_layer_name="Phenotypes",
        equation="Phenotypes = intercept + MiddleLayer + Genotypes",
        traits=["y"],
        class_priors=class_priors_23,
        activation_function="linear"
    )
    
    output_dir = joinpath(tmpdir, "nnmm_skip_no_omics")
    out = runNNMM([layer1, layer2, layer3], [eq1, eq2],
        chain_length=MCMC_CHAIN_LENGTH, burnin=MCMC_BURNIN,
        output_folder=output_dir, seed=seed, double_precision=true)
    
    epv_df = out["EPV_Output_NonLinear"]
    ebv_df = out["EBV_NonLinear"]
    
    epv = Dict(string.(epv_df.ID) .=> epv_df.EPV)
    ebv = Dict(string.(ebv_df.ID) .=> ebv_df.EBV)
    
    return (ebv=ebv, epv=epv)
end

# ============================================================
# Model 4: NNMM with Skip (with omics)
# ============================================================
function run_nnmm_skip_with_omics(prepared, tmpdir::String, seed::Int)
    println("  Running NNMM Skip (with omics)...")
    
    # Create middle layer with latent node + observed omics
    latent_omics_df = DataFrame(ID = string.(prepared.all_ids))
    latent_omics_df[!, "latent1"] = Vector{Union{Missing, Float64}}(fill(missing, prepared.n))
    for (j, oname) in enumerate(prepared.omics_names)
        vals = Vector{Union{Missing, Float64}}(undef, prepared.n)
        for (i, id) in enumerate(prepared.all_ids)
            if haskey(prepared.omics_dict, id)
                vals[i] = prepared.omics_dict[id][j]
            else
                vals[i] = missing
            end
        end
        latent_omics_df[!, oname] = vals
    end
    latent_omics_path = joinpath(tmpdir, "latent_omics.csv")
    CSV.write(latent_omics_path, latent_omics_df, missingstring="NA")
    
    all_traits = ["latent1"; prepared.omics_names]
    
    layer1 = Layer(layer_name="Genotypes", data_path=prepared.geno_path, header=true)
    layer2 = Layer(layer_name="MiddleLayer", data_path=latent_omics_path, header=true, missing_value="NA")
    layer3 = Layer(layer_name="Phenotypes", data_path=prepared.pheno_path, header=true, missing_value="NA")
    
    eq1 = Equation(
        from_layer_name="Genotypes", to_layer_name="MiddleLayer",
        equation="MiddleLayer = intercept + Genotypes",
        traits=all_traits,
        method="BayesC"
    )
    
    class_priors_23 = Dict(
        "MiddleLayer" => (method="BayesC", Pi=0.0, estimatePi=true, estimate_variance_G=true),
        "Genotypes" => (method="BayesC", Pi=0.0, estimatePi=true, estimate_variance_G=true),
    )
    eq2 = Equation(
        from_layer_name="MiddleLayer", to_layer_name="Phenotypes",
        equation="Phenotypes = intercept + MiddleLayer + Genotypes",
        traits=["y"],
        class_priors=class_priors_23,
        activation_function="linear"
    )
    
    output_dir = joinpath(tmpdir, "nnmm_skip_with_omics")
    out = runNNMM([layer1, layer2, layer3], [eq1, eq2],
        chain_length=MCMC_CHAIN_LENGTH, burnin=MCMC_BURNIN,
        output_folder=output_dir, seed=seed, double_precision=true)
    
    epv_df = out["EPV_Output_NonLinear"]
    ebv_df = out["EBV_NonLinear"]
    
    epv = Dict(string.(epv_df.ID) .=> epv_df.EPV)
    ebv = Dict(string.(ebv_df.ID) .=> ebv_df.EBV)
    
    return (ebv=ebv, epv=epv)
end

# ============================================================
# Evaluation function
# ============================================================
function evaluate_models(prepared, nnmm_results, jwas_geno_ebv, jwas_multi_ebv)
    val_ids = [id for id in prepared.all_ids if id in prepared.val_ids_set]
    
    results = Dict{String, Float64}()
    
    # JWAS G (genotypes only) - EBV only
    results["JWAS_G_EBV"] = calc_accuracy(jwas_geno_ebv, prepared.phen_dict, val_ids)
    
    # JWAS G+M (multi-class) - EBV only
    results["JWAS_GM_EBV"] = calc_accuracy(jwas_multi_ebv, prepared.phen_dict, val_ids)
    
    # NNMM Skip (no omics) - EBV and EPV
    results["NNMM_Skip_NoOmics_EBV"] = calc_accuracy(nnmm_results.no_omics.ebv, prepared.phen_dict, val_ids)
    results["NNMM_Skip_NoOmics_EPV"] = calc_accuracy(nnmm_results.no_omics.epv, prepared.phen_dict, val_ids)
    
    # NNMM Skip (with omics) - EBV and EPV
    results["NNMM_Skip_WithOmics_EBV"] = calc_accuracy(nnmm_results.with_omics.ebv, prepared.phen_dict, val_ids)
    results["NNMM_Skip_WithOmics_EPV"] = calc_accuracy(nnmm_results.with_omics.epv, prepared.phen_dict, val_ids)
    
    return results
end

# ============================================================
# Main Benchmark Loop
# ============================================================
function run_benchmark()
    println("="^80)
    println("Model Comparison Benchmark")
    println("="^80)
    println("MCMC Settings: chain_length=$MCMC_CHAIN_LENGTH, burnin=$MCMC_BURNIN")
    println("Replicates: $(collect(REPLICATES))")
    println("Training missing rates: $TRAIN_MISSING_RATES")
    println("Validation missing rates: $VAL_MISSING_RATES")
    QUICK_RUN && println("*** QUICK RUN MODE (set QUICK_RUN=0 for full benchmark) ***")
    println("="^80)
    
    # Calculate total iterations for progress tracking
    total_iters = length(REPLICATES) * length(TRAIN_MISSING_RATES) * length(VAL_MISSING_RATES)
    current_iter = 0
    
    # Results storage
    all_results = DataFrame(
        Replicate = Int[],
        TrainMissRate = Int[],
        ValMissRate = Int[],
        Model = String[],
        Metric = String[],
        Accuracy = Float64[]
    )
    
    for rep in REPLICATES
        println("\n" * "="^60)
        println("Processing Replicate $rep")
        println("="^60)
        
        data = load_data(rep)
        println("Loaded: $(nrow(data.geno_df)) individuals, $(ncol(data.geno_df)-1) SNPs")
        println("Train: $(length(data.train_ids)), Val: $(length(data.val_ids))")
        
        for train_miss_rate in TRAIN_MISSING_RATES
            for val_miss_rate in VAL_MISSING_RATES
                current_iter += 1
                println("\n--- [$current_iter/$total_iters] Train Miss: $(train_miss_rate)%, Val Miss: $(val_miss_rate)% ---")
                
                tmpdir = mktempdir()
                seed = RANDOM_SEED + rep * 1000 + train_miss_rate * 10 + val_miss_rate
                
                try
                    # Prepare data with different missing rates for train and val
                    prepared = prepare_data_files(data, tmpdir, train_miss_rate, val_miss_rate)
                    println("  Omics features: $(length(prepared.omics_names))")
                    flush(stdout)
                    
                    # Run all models with timing
                    t1 = time()
                    jwas_geno_ebv = run_jwas_geno_only(prepared, tmpdir, seed)
                    println("    JWAS G done ($(round(time()-t1, digits=1))s)")
                    flush(stdout)
                    
                    t2 = time()
                    jwas_multi_ebv = run_jwas_multiclass(prepared, tmpdir, seed)
                    println("    JWAS G+M done ($(round(time()-t2, digits=1))s)")
                    flush(stdout)
                    
                    t3 = time()
                    nnmm_no_omics = run_nnmm_skip_no_omics(prepared, tmpdir, seed)
                    println("    NNMM Skip (no omics) done ($(round(time()-t3, digits=1))s)")
                    flush(stdout)
                    
                    t4 = time()
                    nnmm_with_omics = run_nnmm_skip_with_omics(prepared, tmpdir, seed)
                    println("    NNMM Skip (with omics) done ($(round(time()-t4, digits=1))s)")
                    flush(stdout)
                    
                    nnmm_results = (no_omics=nnmm_no_omics, with_omics=nnmm_with_omics)
                    
                    # Evaluate
                    results = evaluate_models(prepared, nnmm_results, jwas_geno_ebv, jwas_multi_ebv)
                    
                    # Store results
                    for (key, acc) in results
                        parts = split(key, "_")
                        model = join(parts[1:end-1], "_")
                        metric = parts[end]
                        
                        push!(all_results, (
                            Replicate = rep,
                            TrainMissRate = train_miss_rate,
                            ValMissRate = val_miss_rate,
                            Model = model,
                            Metric = metric,
                            Accuracy = acc
                        ))
                    end
                    
                    # Print intermediate accuracy
                    println("  >> Val Accuracy: JWAS_G=$(round(results["JWAS_G_EBV"], digits=3)), " *
                            "NNMM_Skip(omics)_EPV=$(round(results["NNMM_Skip_WithOmics_EPV"], digits=3))")
                    flush(stdout)
                    
                    # Save intermediate results
                    CSV.write("/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/benchmark_results_full.csv", all_results)
                catch e
                    println("  ERROR: $e")
                    @warn "Failed for rep=$rep, train_miss=$train_miss_rate, val_miss=$val_miss_rate" exception=(e, catch_backtrace())
                finally
                    # Cleanup
                    rm(tmpdir, recursive=true, force=true)
                end
            end
        end
    end
    
    return all_results
end

# ============================================================
# Summary and Output
# ============================================================
function summarize_results(results::DataFrame)
    println("\n" * "="^80)
    println("SUMMARY RESULTS (averaged across replicates)")
    println("="^80)
    
    # Group and average
    summary = combine(
        groupby(results, [:TrainMissRate, :ValMissRate, :Model, :Metric]),
        :Accuracy => mean => :MeanAcc,
        :Accuracy => std => :StdAcc
    )
    
    # Define model display order
    model_order = [
        "JWAS_G",        # G only BayesC
        "JWAS_GM",       # G+M multi-class BayesC
        "NNMM_Skip_NoOmics",   # NNMM skip (no omics)
        "NNMM_Skip_WithOmics"  # NNMM skip (with omics)
    ]
    
    val_rates = sort(unique(summary.ValMissRate))
    
    # Pivot for nicer display
    for train_miss in sort(unique(summary.TrainMissRate))
        println("\n" * "="^100)
        println("TRAINING OMICS MISSING RATE: $(train_miss)%")
        println("="^100)
        
        sub = filter(row -> row.TrainMissRate == train_miss, summary)
        
        # Print header
        header = rpad("Model", 25) * rpad("Metric", 10)
        for vr in val_rates
            header *= rpad("Val$(vr)%", 10)
        end
        println(header)
        println("-"^(35 + 10 * length(val_rates)))
        
        for model in model_order
            for metric in ["EBV", "EPV"]
                row_data = [rpad(model, 25), rpad(metric, 10)]
                has_data = false
                
                for vr in val_rates
                    val = filter(r -> r.Model == model && r.Metric == metric && r.ValMissRate == vr, sub)
                    if nrow(val) > 0 && !isnan(val.MeanAcc[1])
                        push!(row_data, rpad(string(round(val.MeanAcc[1], digits=3)), 10))
                        has_data = true
                    else
                        push!(row_data, rpad("N/A", 10))
                    end
                end
                
                if has_data
                    println(join(row_data, ""))
                end
            end
        end
    end
    
    return summary
end

"""Create a formatted table similar to the user's example"""
function create_comparison_table(summary::DataFrame)
    println("\n" * "="^120)
    println("FORMATTED COMPARISON TABLE")
    println("="^120)
    
    val_rates = sort(unique(summary.ValMissRate))
    train_rates = sort(unique(summary.TrainMissRate))
    
    # Header row
    header = rpad("Train Miss", 12) * rpad("Model", 25) * rpad("Metric", 8)
    for vr in val_rates
        header *= rpad("$(vr)%", 8)
    end
    println(header)
    println("-"^(45 + 8 * length(val_rates)))
    
    model_names = [
        ("JWAS_G", "G (JWAS BayesC)"),
        ("JWAS_GM", "G+M (JWAS Multi-class)"),
        ("NNMM_Skip_NoOmics", "NNMM Skip (no omics)"),
        ("NNMM_Skip_WithOmics", "NNMM Skip (with omics)")
    ]
    
    for train_miss in train_rates
        first_in_group = true
        for (model_key, model_name) in model_names
            for metric in ["EBV", "EPV"]
                sub = filter(r -> r.TrainMissRate == train_miss && r.Model == model_key && r.Metric == metric, summary)
                if nrow(sub) == 0
                    continue
                end
                
                # Check if any values exist (handle vector properly)
                has_values = any(x -> !isnan(x), sub.MeanAcc)
                if !has_values
                    continue
                end
                
                row = first_in_group ? rpad("$(train_miss)%", 12) : rpad("", 12)
                first_in_group = false
                row *= rpad(model_name, 25) * rpad(metric, 8)
                
                for vr in val_rates
                    val_row = filter(r -> r.ValMissRate == vr, sub)
                    if nrow(val_row) > 0 && !isnan(val_row.MeanAcc[1])
                        row *= rpad(string(round(val_row.MeanAcc[1], digits=3)), 8)
                    else
                        row *= rpad("-", 8)
                    end
                end
                println(row)
            end
        end
        println("-"^(45 + 8 * length(val_rates)))
    end
end

# ============================================================
# Run the benchmark
# ============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting benchmark at $(Dates.now())...")
    
    results = run_benchmark()
    
    # Save raw results
    output_file = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/benchmark_results_full.csv"
    CSV.write(output_file, results)
    println("\nRaw results saved to: $output_file")
    
    # Display summary
    summary = summarize_results(results)
    
    # Create formatted comparison table
    create_comparison_table(summary)
    
    # Save summary
    summary_file = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/benchmark_summary.csv"
    CSV.write(summary_file, summary)
    println("\nSummary saved to: $summary_file")
    
    println("\nBenchmark completed at $(Dates.now())!")
end

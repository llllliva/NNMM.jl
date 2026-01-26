#!/usr/bin/env julia
#=
================================================================================
Skip Connection Test Suite for NNMM.jl
================================================================================
Tests the skip connection feature (Layer 1 → Layer 3 shortcut) across multiple
middle layer scenarios. Focuses on NNMM behavior.
================================================================================
=#

using Pkg
Pkg.activate("/Users/haocheng/Github/AFOCUS/NNMM.jl")

using Random, Statistics, DataFrames, CSV, LinearAlgebra, StatsBase
using NNMM
using JWAS

println("="^70)
println("Skip Connection Test Suite for NNMM.jl")
println("="^70)

# ============================================================
# Configuration
# ============================================================
const CHAIN_LENGTH = 500
const BURNIN = 100
const SEED = 12345

# Scenario definitions
# Each scenario defines the MIDDLE LAYER composition (latent traits + observed omics)
# The SAME middle layer data is used for BOTH skip=false and skip=true tests
scenarios = [
    (name="S1_all_observed", n_latent=0, missing_pct=0),      # 0 latent + 20 observed (all complete)
    (name="S2_1_latent", n_latent=1, missing_pct=0),          # 1 latent + 20 observed (all complete)
    (name="S3_3_latent", n_latent=3, missing_pct=0),          # 3 latent + 20 observed (all complete)
    (name="S4_all_latent", n_latent=21, missing_pct=0),       # 21 latent + 0 observed (all missing)
    (name="S5_partial_missing", n_latent=0, missing_pct=50),  # 0 latent + 20 observed (50% missing)
]

# ============================================================
# WHAT SKIP=NO vs SKIP=YES MEANS:
# ============================================================
# 
# SKIP=NO (skip=false):
#   - Equation 2→3: "Phenotypes = intercept + MiddleLayer"
#   - NO direct genotype→phenotype connection
#   - Latent traits ARE INCLUDED in Layer 2 (sampled from genotypes)
#   - Phenotypes depend ONLY on omics (which includes latent traits)
#   - Single marker class: MiddleLayer (omics effects only)
#
# SKIP=YES (skip=true):
#   - Equation 2→3: "Phenotypes = intercept + MiddleLayer + Genotypes"
#   - Direct genotype→phenotype connection INCLUDED
#   - Latent traits are REMOVED from Layer 2 (handled via skip connection)
#   - Phenotypes depend on BOTH observed omics AND direct genetic effects
#   - Two marker classes: MiddleLayer (observed omics) + Genotypes (skip, handles latent)
#
# IMPORTANT: The middle layer composition DIFFERS between skip=false and skip=true:
#   - skip=false: Layer 2 includes latent traits (all missing) + observed omics
#   - skip=true:  Layer 2 includes ONLY observed omics (latent handled via skip)
#
# ============================================================

# Results storage
results = DataFrame(
    Scenario = String[],
    Skip = Bool[],
    N_Latent = Int[],
    N_Traits = Int[],
    Train_EPV_Acc = Float64[],
    Test_EPV_Acc = Float64[],
    Train_EBV_Acc = Float64[],
    Test_EBV_Acc = Float64[],
    r_EPV_EBV_Train = Float64[],
    r_EPV_EBV_Test = Float64[],
    Train_EBV_JWAS = Float64[],
    Test_EBV_JWAS = Float64[],
    r_NNMM_JWAS_Train = Float64[],
    r_NNMM_JWAS_Test = Float64[],
)

# ============================================================
# Load Data
# ============================================================
println("\n--- Loading data ---")

data_dir = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/nnmm_small_dataset/input_files/data1"

# Load genotypes
geno_df = CSV.read(joinpath(data_dir, "geno_rep1.csv"), DataFrame)
println("Genotypes: $(nrow(geno_df)) individuals, $(ncol(geno_df)-1) SNPs")

# Load omics (0% missing = all observed)
omics_df = CSV.read(joinpath(data_dir, "omics_rep1_miss_0pct.csv"), DataFrame)
if "residual" in names(omics_df)
    select!(omics_df, Not(:residual))
end
omics_names = names(omics_df)[2:end]
println("Omics: $(nrow(omics_df)) individuals, $(length(omics_names)) features")

# Load phenotypes
phen_trn_df = CSV.read(joinpath(data_dir, "phen_rep1_trn.csv"), DataFrame, types=Dict(:FIP => Float64))
phen_val_df = CSV.read(joinpath(data_dir, "phen_rep1_val.csv"), DataFrame, types=Dict(:FIP => Float64))
phen_all_df = vcat(phen_trn_df, phen_val_df)
println("Phenotypes: $(nrow(phen_all_df)) individuals")

# Load train/val IDs
train_ids = Set(string.(CSV.read(joinpath(data_dir, "ID_rep1_trn.csv"), DataFrame).ID))
val_ids = Set(string.(CSV.read(joinpath(data_dir, "ID_rep1_val.csv"), DataFrame).ID))
println("Training: $(length(train_ids)), Validation: $(length(val_ids))")

# Get all IDs in genotype order
all_ids = string.(geno_df.ID)
n = length(all_ids)

# Create phenotype dictionary
phen_dict = Dict(string(row.ID) => row.FIP for row in eachrow(phen_all_df))

# Create omics dictionary
omics_dict = Dict(string(row.ID) => collect(row)[2:end] for row in eachrow(omics_df))

# ============================================================
# Helper Functions
# ============================================================

"""
    prepare_omics_file(all_ids, omics_dict, omics_names, n_latent, missing_pct; seed=12345)

Create omics file with specified number of latent traits and missing percentage.
"""
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
    
    # For S4 (all latent), we don't add observed omics
    if n_latent < 21
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
            
            # Apply random missing pattern if specified
            if missing_pct > 0
                n_miss = round(Int, n * missing_pct / 100)
                # Only make non-missing values missing
                valid_idx = findall(!ismissing, vals)
                if length(valid_idx) > n_miss
                    miss_idx = sample(valid_idx, n_miss, replace=false)
                    vals[miss_idx] .= missing
                end
            end
            
            result_df[!, oname] = vals
            push!(trait_names, oname)
        end
    end
    
    return result_df, trait_names
end

"""
    run_nnmm_test(geno_path, omics_path, pheno_path, trait_names, use_skip, output_dir; seed=12345)

Run NNMM model with or without skip connection.

Args:
    use_skip: If true, equation 2→3 includes "+ Genotypes" (direct genotype→phenotype connection)
              If false, equation 2→3 is "Phenotypes = intercept + MiddleLayer" only
              
Note: The middle layer data (omics_path) is the SAME regardless of use_skip.
      use_skip only affects whether the direct genotype→phenotype connection is included.
"""
function run_nnmm_test(geno_path, omics_path, pheno_path, trait_names, use_skip, output_dir; seed=12345)
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
    
    out = runNNMM([layer1, layer2, layer3], [eq1, eq2],
        chain_length=CHAIN_LENGTH, burnin=BURNIN, output_folder=output_dir, seed=seed)
    
    return out
end

"""
    compute_nnmm_metrics(nnmm_out, all_ids, phen_dict, train_ids, val_ids)

Compute accuracy and correlation metrics for NNMM output.
"""
function compute_nnmm_metrics(nnmm_out, all_ids, phen_dict, train_ids, val_ids)
    # Extract NNMM predictions
    epv_nnmm_df = nnmm_out["EPV_Output_NonLinear"]
    epv_nnmm = Dict(string(k) => v for (k, v) in zip(epv_nnmm_df.ID, epv_nnmm_df.EPV))
    
    ebv_nnmm_df = nnmm_out["EBV_NonLinear"]
    ebv_nnmm = Dict(string(k) => v for (k, v) in zip(ebv_nnmm_df.ID, ebv_nnmm_df.EBV))
    
    # Get common IDs with valid phenotypes
    common_ids = [id for id in all_ids if 
        haskey(epv_nnmm, id) && haskey(ebv_nnmm, id) &&
        haskey(phen_dict, id) && !ismissing(phen_dict[id]) &&
        !ismissing(epv_nnmm[id]) && !ismissing(ebv_nnmm[id])]
    
    train_common = [id for id in common_ids if id in train_ids]
    val_common = [id for id in common_ids if id in val_ids]
    
    metrics = Dict{String, Float64}()
    
    # Training set metrics
    if length(train_common) > 2
        epv_train = Float64[epv_nnmm[id] for id in train_common]
        ebv_train = Float64[ebv_nnmm[id] for id in train_common]
        y_train = Float64[phen_dict[id] for id in train_common]
        
        metrics["train_epv_acc"] = cor(epv_train, y_train)
        metrics["train_ebv_acc"] = cor(ebv_train, y_train)
        metrics["r_epv_ebv_train"] = cor(epv_train, ebv_train)
    else
        metrics["train_epv_acc"] = NaN
        metrics["train_ebv_acc"] = NaN
        metrics["r_epv_ebv_train"] = NaN
    end
    
    # Validation set metrics
    if length(val_common) > 2
        epv_val = Float64[epv_nnmm[id] for id in val_common]
        ebv_val = Float64[ebv_nnmm[id] for id in val_common]
        y_val = Float64[phen_dict[id] for id in val_common]
        
        metrics["test_epv_acc"] = cor(epv_val, y_val)
        metrics["test_ebv_acc"] = cor(ebv_val, y_val)
        metrics["r_epv_ebv_test"] = cor(epv_val, ebv_val)
    else
        metrics["test_epv_acc"] = NaN
        metrics["test_ebv_acc"] = NaN
        metrics["r_epv_ebv_test"] = NaN
    end
    
    return metrics
end

"""
    run_jwas_multiclass(geno_path, omics_path, pheno_path, use_skip, output_dir; seed=12345)

Run JWAS multi-class BayesC for comparison with NNMM skip connection.
When use_skip=true: two classes (geno + omics_class)
When use_skip=false: one class (omics_class only)
"""
function run_jwas_multiclass(geno_path, omics_path, pheno_path, use_skip, output_dir; seed=12345)
    # Load genotypes
    geno = JWAS.get_genotypes(geno_path, separator=',', header=true)
    
    # Load omics as second class (treat as genotypes for JWAS)
    omics_class = JWAS.get_genotypes(omics_path, separator=',', header=true, quality_control=false)
    # Ensure sum2pq is positive
    omics_var = var(omics_class.genotypes, dims=1)
    omics_class.sum2pq = max(float(sum(omics_var)), eps(Float64))
    
    # Build model - JWAS needs variables in global scope
    # We'll use eval to set them in Main
    Main.geno = geno
    Main.omics_class = omics_class
    
    if use_skip
        mme = JWAS.build_model("y = intercept + geno + omics_class")
    else
        mme = JWAS.build_model("y = intercept + omics_class")
    end
    
    jwas_data = CSV.read(pheno_path, DataFrame, missingstring="NA")
    
    out = JWAS.runMCMC(mme, jwas_data,
        chain_length=CHAIN_LENGTH, burnin=BURNIN, output_folder=output_dir, seed=seed, outputEBV=true)
    
    return out
end

"""
    compute_jwas_metrics(jwas_out, all_ids, phen_dict, train_ids, val_ids)

Compute accuracy metrics for JWAS output.
"""
function compute_jwas_metrics(jwas_out, all_ids, phen_dict, train_ids, val_ids)
    # Extract JWAS EBV
    ebv_jwas_df = jwas_out["EBV_y"]
    ebv_jwas = Dict(string(k) => v for (k, v) in zip(ebv_jwas_df.ID, ebv_jwas_df.EBV))
    
    # Get common IDs with valid phenotypes
    common_ids = [id for id in all_ids if 
        haskey(ebv_jwas, id) &&
        haskey(phen_dict, id) && !ismissing(phen_dict[id]) &&
        !ismissing(ebv_jwas[id])]
    
    train_common = [id for id in common_ids if id in train_ids]
    val_common = [id for id in common_ids if id in val_ids]
    
    metrics = Dict{String, Float64}()
    
    # Training set metrics
    if length(train_common) > 2
        ebv_train = Float64[ebv_jwas[id] for id in train_common]
        y_train = Float64[phen_dict[id] for id in train_common]
        metrics["train_ebv_jwas"] = cor(ebv_train, y_train)
    else
        metrics["train_ebv_jwas"] = NaN
    end
    
    # Validation set metrics
    if length(val_common) > 2
        ebv_val = Float64[ebv_jwas[id] for id in val_common]
        y_val = Float64[phen_dict[id] for id in val_common]
        metrics["test_ebv_jwas"] = cor(ebv_val, y_val)
    else
        metrics["test_ebv_jwas"] = NaN
    end
    
    return metrics, ebv_jwas
end

"""
    compute_nnmm_jwas_correlation(ebv_nnmm, ebv_jwas, all_ids, train_ids, val_ids)

Compute correlation between NNMM and JWAS EBV predictions.
"""
function compute_nnmm_jwas_correlation(ebv_nnmm, ebv_jwas, all_ids, train_ids, val_ids)
    # Get common IDs
    common_ids = [id for id in all_ids if 
        haskey(ebv_nnmm, id) && haskey(ebv_jwas, id) &&
        !ismissing(ebv_nnmm[id]) && !ismissing(ebv_jwas[id])]
    
    train_common = [id for id in common_ids if id in train_ids]
    val_common = [id for id in common_ids if id in val_ids]
    
    metrics = Dict{String, Float64}()
    
    # Training set correlation
    if length(train_common) > 2
        nnmm_train = Float64[ebv_nnmm[id] for id in train_common]
        jwas_train = Float64[ebv_jwas[id] for id in train_common]
        metrics["r_nnmm_jwas_train"] = cor(nnmm_train, jwas_train)
    else
        metrics["r_nnmm_jwas_train"] = NaN
    end
    
    # Validation set correlation
    if length(val_common) > 2
        nnmm_val = Float64[ebv_nnmm[id] for id in val_common]
        jwas_val = Float64[ebv_jwas[id] for id in val_common]
        metrics["r_nnmm_jwas_test"] = cor(nnmm_val, jwas_val)
    else
        metrics["r_nnmm_jwas_test"] = NaN
    end
    
    return metrics
end

"""
    print_scenario_results(scenario_name, use_skip, metrics, n_traits)

Print results for a single scenario.
"""
function print_scenario_results(scenario_name, use_skip, metrics, n_traits)
    skip_str = use_skip ? "WITH SKIP" : "NO SKIP"
    println("\n" * "="^60)
    println("$scenario_name ($n_traits traits) - $skip_str")
    println("="^60)
    println("Training Set:")
    println("  EPV Accuracy:     $(round(metrics["train_epv_acc"], digits=4))")
    println("  EBV Accuracy:     $(round(metrics["train_ebv_acc"], digits=4))")
    println("  EBV JWAS:         $(round(get(metrics, "train_ebv_jwas", NaN), digits=4))")
    println("  r(EPV, EBV):      $(round(metrics["r_epv_ebv_train"], digits=4))")
    println("  r(NNMM, JWAS):    $(round(get(metrics, "r_nnmm_jwas_train", NaN), digits=4))")
    println("Validation Set:")
    println("  EPV Accuracy:     $(round(metrics["test_epv_acc"], digits=4))")
    println("  EBV Accuracy:     $(round(metrics["test_ebv_acc"], digits=4))")
    println("  EBV JWAS:         $(round(get(metrics, "test_ebv_jwas", NaN), digits=4))")
    println("  r(EPV, EBV):      $(round(metrics["r_epv_ebv_test"], digits=4))")
    println("  r(NNMM, JWAS):    $(round(get(metrics, "r_nnmm_jwas_test", NaN), digits=4))")
end

# ============================================================
# Main Test Loop
# ============================================================
println("\n" * "="^70)
println("Starting Test Suite")
println("="^70)

println("\nTEST STRUCTURE:")
println("  For EACH scenario, we test BOTH:")
println("    1. skip=false: Phenotypes = intercept + MiddleLayer")
println("       → Layer 2 INCLUDES latent traits (sampled from genotypes)")
println("    2. skip=true:  Phenotypes = intercept + MiddleLayer + Genotypes")
println("       → Layer 2 EXCLUDES latent traits (handled via direct Genotypes connection)")
println("\n  The MIDDLE LAYER composition DIFFERS:")
println("    - skip=false: Layer 2 = latent traits + observed omics")
println("    - skip=true:  Layer 2 = observed omics only (latent via skip connection)")
println("\nSCENARIOS:")
for (i, s) in enumerate(scenarios)
    n_observed = 20 - s.n_latent  # Assuming 20 total omics features
    if s.n_latent == 21
        n_observed = 0
    end
    println("  $(i). $(s.name):")
    println("     - Latent traits (all missing): $(s.n_latent)")
    println("     - Observed omics: $n_observed")
    if s.missing_pct > 0
        println("     - Missing pattern: $(s.missing_pct)% of observed omics randomly missing")
    else
        println("     - Missing pattern: None (all observed omics complete)")
    end
end
println("="^70)

for scenario in scenarios
    println("\n" * "#"^70)
    println("# SCENARIO: $(scenario.name)")
    println("# n_latent=$(scenario.n_latent), missing_pct=$(scenario.missing_pct)")
    println("#"^70)
    
    # Create temp directory for this scenario
    tmpdir = mktempdir()
    println("Temp dir: $tmpdir")
    
    # Prepare files
    geno_path = joinpath(tmpdir, "genotypes.csv")
    CSV.write(geno_path, geno_df)
    
    # Prepare phenotype file (validation set as missing)
    y_with_missing = Vector{Union{Missing, Float64}}(undef, n)
    for (i, id) in enumerate(all_ids)
        if id in val_ids
            y_with_missing[i] = missing
        else
            y_with_missing[i] = get(phen_dict, id, missing)
        end
    end
    pheno_df = DataFrame(ID = all_ids, y = y_with_missing)
    pheno_path = joinpath(tmpdir, "phenotypes.csv")
    CSV.write(pheno_path, pheno_df, missingstring="NA")
    
    # Test with and without skip connection
    for use_skip in [false, true]
        skip_str = use_skip ? "skip" : "noskip"
        println("\n" * "="^70)
        println("Running $(scenario.name) with skip=$use_skip")
        println("="^70)
        
        # CRITICAL: When skip=true, latent traits are REMOVED from Layer 2
        #           They are handled via direct genotype→phenotype connection instead
        if use_skip
            # skip=true: NO latent traits in Layer 2 (they go via skip connection)
            n_latent_for_layer2 = 0
            println("MODEL STRUCTURE (skip=YES):")
            println("  Equation 2→3: Phenotypes = intercept + MiddleLayer + Genotypes")
            println("  → Direct genotype→phenotype connection: INCLUDED (handles latent traits)")
            println("  → Omics effects: INCLUDED")
            println("  → Two separate marker classes: MiddleLayer (omics) + Genotypes (skip)")
            println("\nMIDDLE LAYER COMPOSITION (skip=YES):")
            println("  → Latent traits: REMOVED from Layer 2 (handled via skip connection)")
        else
            # skip=false: latent traits ARE in Layer 2
            n_latent_for_layer2 = scenario.n_latent
            println("MODEL STRUCTURE (skip=NO):")
            println("  Equation 2→3: Phenotypes = intercept + MiddleLayer")
            println("  → Direct genotype→phenotype connection: NOT INCLUDED")
            println("  → Omics effects: INCLUDED")
            println("  → Single marker class: MiddleLayer (omics only)")
            println("\nMIDDLE LAYER COMPOSITION (skip=NO):")
            println("  → Latent traits: INCLUDED in Layer 2 (sampled from genotypes)")
        end
        
        # Prepare omics file based on scenario AND skip setting
        omics_scenario_df, trait_names = prepare_omics_file(
            all_ids, omics_dict, omics_names, 
            n_latent_for_layer2, scenario.missing_pct, seed=SEED
        )
        omics_path = joinpath(tmpdir, "omics_$(skip_str).csv")
        CSV.write(omics_path, omics_scenario_df, missingstring="NA")
        n_traits = length(trait_names)
        
        println("  Total traits in Layer 2: $n_traits")
        println("  Latent traits (all missing): $n_latent_for_layer2")
        println("  Observed omics: $(n_traits - n_latent_for_layer2)")
        if scenario.missing_pct > 0
            println("  Missing pattern: $(scenario.missing_pct)% of observed omics randomly missing")
        else
            println("  Missing pattern: None (all observed omics complete)")
        end
        println("="^70)
        
        try
            # Run NNMM
            nnmm_out_dir = joinpath(tmpdir, "nnmm_$(skip_str)")
            nnmm_out = run_nnmm_test(geno_path, omics_path, pheno_path, trait_names, use_skip, nnmm_out_dir, seed=SEED)
            
            # Compute NNMM metrics
            metrics = compute_nnmm_metrics(nnmm_out, all_ids, phen_dict, train_ids, val_ids)
            
            # Extract NNMM EBV for correlation
            ebv_nnmm_df = nnmm_out["EBV_NonLinear"]
            ebv_nnmm = Dict(string(k) => v for (k, v) in zip(ebv_nnmm_df.ID, ebv_nnmm_df.EBV))
            
            # Run JWAS multi-class BayesC for comparison
            println("  Running JWAS multi-class BayesC (use_skip=$use_skip)...")
            jwas_out_dir = joinpath(tmpdir, "jwas_$(skip_str)")
            jwas_out = run_jwas_multiclass(geno_path, omics_path, pheno_path, use_skip, jwas_out_dir, seed=SEED)
            
            # Compute JWAS metrics
            jwas_metrics, ebv_jwas = compute_jwas_metrics(jwas_out, all_ids, phen_dict, train_ids, val_ids)
            metrics = merge(metrics, jwas_metrics)
            
            # Compute NNMM-JWAS correlation
            corr_metrics = compute_nnmm_jwas_correlation(ebv_nnmm, ebv_jwas, all_ids, train_ids, val_ids)
            metrics = merge(metrics, corr_metrics)
            
            # Print results
            print_scenario_results(scenario.name, use_skip, metrics, n_traits)
            
            # Store results
            push!(results, (
                scenario.name, use_skip, scenario.n_latent, n_traits,
                metrics["train_epv_acc"], metrics["test_epv_acc"],
                metrics["train_ebv_acc"], metrics["test_ebv_acc"],
                metrics["r_epv_ebv_train"], metrics["r_epv_ebv_test"],
                get(metrics, "train_ebv_jwas", NaN), get(metrics, "test_ebv_jwas", NaN),
                get(metrics, "r_nnmm_jwas_train", NaN), get(metrics, "r_nnmm_jwas_test", NaN)
            ))
            
        catch e
            println("ERROR in $(scenario.name) with $skip_str: $e")
            @error "Stack trace" exception=(e, catch_backtrace())
            # Store NaN results
            push!(results, (scenario.name, use_skip, scenario.n_latent, n_traits, 
                NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN))
        end
    end
end

# ============================================================
# Final Summary Table
# ============================================================
println("\n" * "="^120)
println("FINAL SUMMARY")
println("="^120)

println("\n| Scenario          | Skip | Latent | Traits | Train EPV | Test EPV  | Train EBV | Test EBV  | Train JWAS | Test JWAS | r(NNMM,JWAS) Train | r(NNMM,JWAS) Test |")
println("|-------------------|------|--------|--------|-----------|-----------|-----------|-----------|------------|-----------|---------------------|-------------------|")

for row in eachrow(results)
    skip_str = row.Skip ? "Yes" : "No"
    println("| $(rpad(row.Scenario, 17)) | $(rpad(skip_str, 4)) | " *
            "$(lpad(row.N_Latent, 6)) | $(lpad(row.N_Traits, 6)) | " *
            "$(lpad(round(row.Train_EPV_Acc, digits=3), 9)) | " *
            "$(lpad(round(row.Test_EPV_Acc, digits=3), 9)) | " *
            "$(lpad(round(row.Train_EBV_Acc, digits=3), 9)) | " *
            "$(lpad(round(row.Test_EBV_Acc, digits=3), 9)) | " *
            "$(lpad(round(row.Train_EBV_JWAS, digits=3), 10)) | " *
            "$(lpad(round(row.Test_EBV_JWAS, digits=3), 9)) | " *
            "$(lpad(round(row.r_NNMM_JWAS_Train, digits=3), 19)) | " *
            "$(lpad(round(row.r_NNMM_JWAS_Test, digits=3), 17)) |")
end

# Save results to CSV
results_path = "/Users/haocheng/Github/AFOCUS/NNMM.jl/TempTestData/skip_connection_results.csv"
CSV.write(results_path, results)
println("\nResults saved to: $results_path")

# ============================================================
# Key Observations
# ============================================================
println("\n" * "="^70)
println("KEY OBSERVATIONS")
println("="^70)

# 1. Skip connection impact
println("\n1. Skip Connection Impact on Test Accuracy:")
for scenario_name in unique(results.Scenario)
    skip_row = filter(row -> row.Scenario == scenario_name && row.Skip, results)
    noskip_row = filter(row -> row.Scenario == scenario_name && !row.Skip, results)
    if nrow(skip_row) > 0 && nrow(noskip_row) > 0
        skip_acc = skip_row[1, :Test_EBV_Acc]
        noskip_acc = noskip_row[1, :Test_EBV_Acc]
        diff = skip_acc - noskip_acc
        direction = diff > 0 ? "+" : ""
        println("   $scenario_name: Skip=$(round(skip_acc, digits=3)), NoSkip=$(round(noskip_acc, digits=3)), Δ=$(direction)$(round(diff, digits=3))")
    end
end

# 2. EPV vs EBV behavior
println("\n2. EPV vs EBV Behavior:")
println("   (EPV should ≈ EBV on test set, but EPV > EBV on train set)")
for row in eachrow(results)
    if !isnan(row.r_EPV_EBV_Train) && !isnan(row.r_EPV_EBV_Test)
        skip_str = row.Skip ? "Skip" : "NoSkip"
        train_diff = row.Train_EPV_Acc - row.Train_EBV_Acc
        test_diff = row.Test_EPV_Acc - row.Test_EBV_Acc
        status_train = train_diff > 0 ? "✓" : "~"
        status_test = abs(test_diff) < 0.05 ? "✓" : "~"
        println("   $(row.Scenario) ($skip_str): Train EPV-EBV=$(round(train_diff, digits=3))$status_train, Test EPV-EBV=$(round(test_diff, digits=3))$status_test")
    end
end

# 3. Latent vs Observed impact
println("\n3. Impact of Latent Traits on Accuracy:")
println("   (More latent traits should reduce accuracy due to less observed information)")
noskip_results = filter(row -> !row.Skip, results)
if nrow(noskip_results) > 0
    for row in eachrow(noskip_results)
        println("   $(row.Scenario): $(row.N_Latent) latent / $(row.N_Traits) total → Test EBV=$(round(row.Test_EBV_Acc, digits=3))")
    end
end

# 4. NNMM-JWAS Equivalence (with skip connection)
println("\n4. NNMM-JWAS Equivalence (Skip Connection):")
println("   (NNMM with skip should be equivalent to JWAS multi-class BayesC)")
for row in eachrow(results)
    if row.Skip && !isnan(row.r_NNMM_JWAS_Train) && !isnan(row.r_NNMM_JWAS_Test)
        train_corr = row.r_NNMM_JWAS_Train
        test_corr = row.r_NNMM_JWAS_Test
        status_train = train_corr > 0.9 ? "✓✓" : train_corr > 0.7 ? "✓" : "~"
        status_test = test_corr > 0.9 ? "✓✓" : test_corr > 0.7 ? "✓" : "~"
        println("   $(row.Scenario): r(Train)=$(round(train_corr, digits=3))$status_train, r(Test)=$(round(test_corr, digits=3))$status_test")
    end
end

println("\n" * "="^70)
println("Test Suite Complete!")
println("="^70)

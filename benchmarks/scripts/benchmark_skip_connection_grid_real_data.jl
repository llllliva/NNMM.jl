#!/usr/bin/env julia
#=
================================================================================
Benchmark: Skip Connection 3×3 Grid (Real Data, TempTestData)
================================================================================

Runs NNMM on TempTestData (nnmm_small_dataset) across a 3×3 grid:
  - training omics missing rate ∈ {0%, 50%, 100%}
  - testing  omics missing rate ∈ {0%, 50%, 100%}

For each grid cell, fits two NNMM models:
  A) Omics-only:           y = intercept + MiddleLayer
  B) Omics + Genotypes:    y = intercept + MiddleLayer + Genotypes   (skip connection)
     with JWAS-like multi-class priors (separate priors per class).

Also runs JWAS multi-class BayesC once for the complete-omics case (0%, 0%) and
reports correlation with NNMM (skip) EPV_Output.

Outputs:
  - CSV:      <OUTPUT_DIR>/results.csv
  - Markdown: <OUTPUT_DIR>/summary.md
  - Logs:     <OUTPUT_DIR>/run.log   (if you redirect stdout/stderr externally)

Notes
  - This script expects TempTestData to exist locally (it is gitignored).
  - Run from repo root with: julia --project=. benchmarks/scripts/benchmark_skip_connection_grid_real_data.jl
=#

using Random, Statistics, DataFrames, CSV
using Dates

using NNMM
using JWAS

function _env_int(name::AbstractString, default::Int)
    v = get(ENV, name, "")
    return isempty(v) ? default : parse(Int, v)
end

function _env_str(name::AbstractString, default::String)
    v = get(ENV, name, "")
    return isempty(v) ? default : v
end

function _cor_or_nan(x::Vector{Float64}, y::Vector{Float64})
    if length(x) < 3
        return NaN
    end
    # If the predictor is constant (e.g., EBV_Direct_Skip in an omics-only model),
    # correlation is undefined; treat predictive ability as 0 in that case.
    if iszero(var(x)) || iszero(var(y))
        return 0.0
    end
    c = cor(x, y)
    return isfinite(c) ? c : NaN
end

function _eval_corr(pred_by_id::Dict{String, Float64}, y_true::Dict{String, Float64}, ids::Vector{String})
    ids_valid = [id for id in ids if haskey(y_true, id) && haskey(pred_by_id, id) && isfinite(pred_by_id[id])]
    y = Float64[y_true[id] for id in ids_valid]
    p = Float64[pred_by_id[id] for id in ids_valid]
    return (n=length(ids_valid), corr=_cor_or_nan(p, y))
end

function _df_to_dict(df::DataFrame, id_col::Symbol, value_col::Symbol)
    out = Dict{String, Float64}()
    for row in eachrow(df)
        id = string(row[id_col])
        v = row[value_col]
        if !(ismissing(v) || isnothing(v))
            out[id] = Float64(v)
        end
    end
    return out
end

function _mask_val_omics!(omics_df::DataFrame, val_ids_sorted::Vector{String}, test_missing_pct::Int)
    if test_missing_pct <= 0
        return omics_df
    end
    n_val = length(val_ids_sorted)
    k = round(Int, n_val * (test_missing_pct / 100))
    k = clamp(k, 0, n_val)
    if k == 0
        return omics_df
    end

    # Deterministic: take the first k validation IDs after sorting.
    to_mask = Set(val_ids_sorted[1:k])
    id_to_row = Dict{String, Int}(string(omics_df.ID[i]) => i for i in 1:nrow(omics_df))

    cols = names(omics_df)[2:end]
    for id in to_mask
        i = get(id_to_row, id, 0)
        if i == 0
            continue
        end
        for c in cols
            omics_df[i, c] = missing
        end
    end
    return omics_df
end

function _build_middlelayer_df(all_ids::Vector{String}, omics_df::DataFrame, omics_names::Vector{String}; include_latent::Bool)
    omics_df = copy(omics_df)
    omics_df.ID = string.(omics_df.ID)
    id_to_row = Dict{String, Int}(omics_df.ID[i] => i for i in 1:nrow(omics_df))

    n = length(all_ids)
    middle_df = DataFrame(ID = all_ids)
    if include_latent
        middle_df[!, "latent1"] = Vector{Union{Missing, Float64}}(fill(missing, n))
    end

    for oname in omics_names
        col = Symbol(oname)
        vals = Vector{Union{Missing, Float64}}(undef, n)
        for (i, id) in enumerate(all_ids)
            j = get(id_to_row, id, 0)
            vals[i] = (j == 0) ? missing : omics_df[j, col]
        end
        middle_df[!, oname] = vals
    end

    return middle_df
end

function main()
    chain_length = _env_int("MCMC_CHAIN_LENGTH", 200)
    burnin = _env_int("MCMC_BURNIN", 50)
    seed = _env_int("SEED", 12345)

    project_root = abspath(joinpath(@__DIR__, "..", ".."))
    data_dir = joinpath(project_root, "TempTestData", "nnmm_small_dataset", "input_files", "data1")
    if !isdir(data_dir)
        error("TempTestData not found. Expected directory: $data_dir")
    end

    out_dir = _env_str("OUTPUT_DIR", joinpath(project_root, "dev_workspace", "skip_grid_real_data_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    mkpath(out_dir)

    println("Skip grid real-data benchmark")
    println("Data dir: $data_dir")
    println("Output:   $out_dir")
    println("MCMC:     chain_length=$chain_length, burnin=$burnin, seed=$seed")

    Random.seed!(seed)

    # ----------------------------
    # Load base data
    # ----------------------------
    geno_df = CSV.read(joinpath(data_dir, "geno_rep1.csv"), DataFrame)
    all_ids = string.(geno_df.ID)

    phen_trn_df = CSV.read(
        joinpath(data_dir, "phen_rep1_trn.csv"),
        DataFrame;
        missingstring="NA",
        silencewarnings=true,
        types=Dict(:FIP => Union{Missing, Float64}),
    )
    phen_val_df = CSV.read(
        joinpath(data_dir, "phen_rep1_val.csv"),
        DataFrame;
        missingstring="NA",
        silencewarnings=true,
        types=Dict(:FIP => Union{Missing, Float64}),
    )

    train_ids = Set(string.(CSV.read(joinpath(data_dir, "ID_rep1_trn.csv"), DataFrame).ID))
    val_ids = Set(string.(CSV.read(joinpath(data_dir, "ID_rep1_val.csv"), DataFrame).ID))
    val_ids_sorted = sort!(collect(val_ids))

    # Truth for evaluation (train values from phen_rep1_trn, val values from phen_rep1_val).
    y_true = Dict{String, Float64}()
    for row in eachrow(phen_trn_df)
        if !ismissing(row.FIP)
            y_true[string(row.ID)] = Float64(row.FIP)
        end
    end
    for row in eachrow(phen_val_df)
        if !ismissing(row.FIP)
            y_true[string(row.ID)] = Float64(row.FIP)
        end
    end

    # Omics feature names (20 observed features; drop residual).
    omics0 = CSV.read(joinpath(data_dir, "omics_rep1_miss_0pct.csv"), DataFrame; missingstring="NA", silencewarnings=true)
    if "residual" in names(omics0)
        select!(omics0, Not(:residual))
    end
    omics_names = string.(names(omics0)[2:end])
    all_traits_latent = vcat(["latent1"], omics_names)
    all_traits_nolatent = omics_names

    # ----------------------------
    # Write stable files (geno + pheno) once
    # ----------------------------
    tmpdir = joinpath(out_dir, "tmp_inputs")
    mkpath(tmpdir)

    geno_path = joinpath(tmpdir, "genotypes.csv")
    CSV.write(geno_path, geno_df)

    pheno_fit_df = DataFrame(ID = string.(phen_trn_df.ID), y = phen_trn_df.FIP) # val IDs already missing in this file
    pheno_path = joinpath(tmpdir, "phenotypes.csv")
    CSV.write(pheno_path, pheno_fit_df; missingstring="NA")

    # ----------------------------
    # Model specs
    # ----------------------------
    # Multi-class BayesC priors (JWAS-like) for 2->3 when using skip.
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

    make_eq1(traits) = Equation(
        from_layer_name="Genotypes",
        to_layer_name="MiddleLayer",
        equation="MiddleLayer = intercept + Genotypes",
        traits=traits,
        method="BayesC",
        Pi=0.0,
        estimatePi=true,
    )

    function make_eq2(use_skip::Bool)
        return Equation(
            from_layer_name="MiddleLayer",
            to_layer_name="Phenotypes",
            equation=use_skip ? "Phenotypes = intercept + MiddleLayer + Genotypes" : "Phenotypes = intercept + MiddleLayer",
            traits=["y"],
            method="BayesC",
            Pi=0.0,
            estimatePi=true,
            class_priors=use_skip ? class_priors_23 : false,
            activation_function="linear",
        )
    end

    # ----------------------------
    # JWAS baseline (complete case only)
    # ----------------------------
    jwas_ebv = Dict{String, Float64}()
    begin
        # Reuse existing JWAS outputs if present (so we can append new NNMM model variants
        # without re-running JWAS each time).
        existing_jwas_path = nothing
        for d in readdir(out_dir; join=false)
            if startswith(d, "jwas_complete")
                cand = joinpath(out_dir, d, "EBV_y.txt")
                if isfile(cand)
                    existing_jwas_path = cand
                    break
                end
            end
        end
        if existing_jwas_path !== nothing
            ebv_jwas_df = CSV.read(existing_jwas_path, DataFrame; missingstring="NA", silencewarnings=true)
            jwas_ebv = _df_to_dict(ebv_jwas_df, :ID, :EBV)
        else
            omics_class_path = joinpath(tmpdir, "omics_as_geno_complete.csv")
            omics_complete = copy(omics0)
            omics_complete.ID = string.(omics_complete.ID)
            CSV.write(omics_class_path, omics_complete; missingstring="NA")

            # JWAS uses `build_model("... + geno + omics_class")` which expects `geno` and
            # `omics_class` to be visible in `Main` (i.e., global scope).
            global geno = JWAS.get_genotypes(geno_path, separator=',', header=true)
            global omics_class = JWAS.get_genotypes(omics_class_path, separator=',', header=true, quality_control=false)
            omics_class.sum2pq = max(float(sum(var(omics_class.genotypes, dims=1))), eps(Float64))

            mme_jwas = JWAS.build_model("y = intercept + geno + omics_class")
            jwas_data = CSV.read(pheno_path, DataFrame; missingstring="NA")

            jwas_outdir = joinpath(out_dir, "jwas_complete")
            _ = mkpath(jwas_outdir)
            out_jwas = JWAS.runMCMC(
                mme_jwas,
                jwas_data;
                chain_length=chain_length,
                burnin=burnin,
                output_folder=jwas_outdir,
                seed=seed,
                outputEBV=true,
            )
            ebv_jwas_df = out_jwas["EBV_y"]
            jwas_ebv = _df_to_dict(ebv_jwas_df, :ID, :EBV)
        end
    end

    # ----------------------------
    # Grid
    # ----------------------------
    train_levels = [0, 50, 100]
    test_levels = [0, 50, 100]
    train_omics_file = Dict(
        0 => "omics_rep1_miss_0pct.csv",
        50 => "omics_rep1_miss_50pct.csv",
        100 => "omics_rep1_miss_100pct.csv",
    )

    results_csv = joinpath(out_dir, "results.csv")
    results = if isfile(results_csv)
        CSV.read(results_csv, DataFrame; missingstring="NA", silencewarnings=true, stringtype=String)
    else
        DataFrame(
            train_missing_pct=Int[],
            test_missing_pct=Int[],
            model=String[], # "omics", "skip", or "skip_nolatent"
            n_train=Int[],
            n_val=Int[],
            train_acc_epv=Float64[],
            val_acc_epv=Float64[],
            train_acc_ebv=Float64[],          # EBV_total (EBV_NonLinear)
            val_acc_ebv=Float64[],
            train_acc_ebv_indirect=Float64[], # EBV_Indirect_NonLinear
            val_acc_ebv_indirect=Float64[],
            train_acc_ebv_direct=Float64[],   # EBV_Direct_Skip
            val_acc_ebv_direct=Float64[],
            r_jwas_epv_train=Float64[],
            r_jwas_epv_val=Float64[],
            wall_time_sec=Float64[],
        )
    end
    if :model in propertynames(results)
        results.model = String.(results.model)
    end

    # Backward/forward compatibility if the on-disk CSV predates a column addition.
    needed_cols = Dict(
        :train_acc_ebv_indirect => Float64,
        :val_acc_ebv_indirect => Float64,
        :train_acc_ebv_direct => Float64,
        :val_acc_ebv_direct => Float64,
    )
    for (col, T) in needed_cols
        if !(col in propertynames(results))
            results[!, col] = fill(T(NaN), nrow(results))
        end
    end

    train_ids_vec = [id for id in all_ids if id in train_ids]
    val_ids_vec = [id for id in all_ids if id in val_ids]

    function _has_result(tr_miss::Int, te_miss::Int, model_name::String)
        if nrow(results) == 0
            return false
        end
        mask = (results.train_missing_pct .== tr_miss) .& (results.test_missing_pct .== te_miss) .& (results.model .== model_name)
        return any(mask)
    end

    for tr_miss in train_levels
        for te_miss in test_levels
            println("\n=== Scenario: train_omics_missing=$(tr_miss)%, test_omics_missing=$(te_miss)% ===")

            omics_base = CSV.read(
                joinpath(data_dir, train_omics_file[tr_miss]),
                DataFrame;
                missingstring="NA",
                silencewarnings=true,
            )
            if "residual" in names(omics_base)
                select!(omics_base, Not(:residual))
            end
            allowmissing!(omics_base)

            _mask_val_omics!(omics_base, val_ids_sorted, te_miss)

            middle_latent_df = _build_middlelayer_df(all_ids, omics_base, omics_names; include_latent=true)
            omics_path_latent = joinpath(tmpdir, "latent_omics_tr$(tr_miss)_te$(te_miss).csv")
            CSV.write(omics_path_latent, middle_latent_df; missingstring="NA")

            middle_nolatent_df = _build_middlelayer_df(all_ids, omics_base, omics_names; include_latent=false)
            omics_path_nolatent = joinpath(tmpdir, "omics_nolatent_tr$(tr_miss)_te$(te_miss).csv")
            CSV.write(omics_path_nolatent, middle_nolatent_df; missingstring="NA")

            for model_name in ("omics", "skip", "skip_nolatent")
                if _has_result(tr_miss, te_miss, model_name)
                    println("Skipping existing NNMM model: $model_name")
                    continue
                end
                use_skip = model_name != "omics"
                include_latent = model_name != "skip_nolatent"
                traits_12 = include_latent ? all_traits_latent : all_traits_nolatent
                omics_path = include_latent ? omics_path_latent : omics_path_nolatent

                println("Running NNMM model: $model_name")

                # Run NNMM
                t_start = time()
                nnmm_res = runNNMM(
                    [
                        Layer(layer_name="Genotypes", data_path=geno_path, header=true),
                        Layer(layer_name="MiddleLayer", data_path=omics_path, header=true, missing_value="NA"),
                        Layer(layer_name="Phenotypes", data_path=pheno_path, header=true, missing_value="NA"),
                    ],
                    [
                        make_eq1(traits_12),
                        make_eq2(use_skip),
                    ];
                    chain_length=chain_length,
                    burnin=burnin,
                    output_samples_frequency=max(1, div(chain_length, 20)),
                    printout_model_info=false,
                    printout_frequency=10^9,
                    output_folder=joinpath(out_dir, "nnmm_" * model_name * "_tr$(tr_miss)_te$(te_miss)_" * Dates.format(now(), "HHMMSS")),
                    seed=seed,
                )
                wall_time = time() - t_start

                epv_df = nnmm_res["EPV_Output_NonLinear"]
                ebv_df = nnmm_res["EBV_NonLinear"]
                ebv_indirect_df = nnmm_res["EBV_Indirect_NonLinear"]
                ebv_direct_df = nnmm_res["EBV_Direct_Skip"]
                epv_by_id = _df_to_dict(epv_df, :ID, :EPV)
                ebv_by_id = _df_to_dict(ebv_df, :ID, :EBV)
                ebv_indirect_by_id = _df_to_dict(ebv_indirect_df, :ID, :EBV)
                ebv_direct_by_id = _df_to_dict(ebv_direct_df, :ID, :EBV)

                train_epv = _eval_corr(epv_by_id, y_true, train_ids_vec)
                val_epv = _eval_corr(epv_by_id, y_true, val_ids_vec)
                train_ebv = _eval_corr(ebv_by_id, y_true, train_ids_vec)
                val_ebv = _eval_corr(ebv_by_id, y_true, val_ids_vec)
                train_ebv_indirect = _eval_corr(ebv_indirect_by_id, y_true, train_ids_vec)
                val_ebv_indirect = _eval_corr(ebv_indirect_by_id, y_true, val_ids_vec)
                train_ebv_direct = _eval_corr(ebv_direct_by_id, y_true, train_ids_vec)
                val_ebv_direct = _eval_corr(ebv_direct_by_id, y_true, val_ids_vec)

                # JWAS comparison (only meaningful for complete case with skip model).
                r_jwas_train = NaN
                r_jwas_val = NaN
                if tr_miss == 0 && te_miss == 0 && use_skip
                    jwas_train = _eval_corr(jwas_ebv, y_true, train_ids_vec)
                    jwas_val = _eval_corr(jwas_ebv, y_true, val_ids_vec)

                    # Correlation JWAS EBV vs NNMM EPV_Output (not vs true y)
                    common_train = [id for id in train_ids_vec if haskey(jwas_ebv, id) && haskey(epv_by_id, id)]
                    common_val = [id for id in val_ids_vec if haskey(jwas_ebv, id) && haskey(epv_by_id, id)]
                    r_jwas_train = _cor_or_nan(Float64[jwas_ebv[id] for id in common_train], Float64[epv_by_id[id] for id in common_train])
                    r_jwas_val = _cor_or_nan(Float64[jwas_ebv[id] for id in common_val], Float64[epv_by_id[id] for id in common_val])

                    println("Complete case sanity: Acc JWAS train=$(round(jwas_train.corr, digits=4)) val=$(round(jwas_val.corr, digits=4))")
                    println("Complete case sanity: r(JWAS EBV, NNMM EPV_Output) train=$(round(r_jwas_train, digits=4)) val=$(round(r_jwas_val, digits=4))")
                end

                push!(
                    results,
                    (
                        train_missing_pct=tr_miss,
                        test_missing_pct=te_miss,
                        model=model_name,
                        n_train=train_epv.n,
                        n_val=val_epv.n,
                        train_acc_epv=train_epv.corr,
                        val_acc_epv=val_epv.corr,
                        train_acc_ebv=train_ebv.corr,
                        val_acc_ebv=val_ebv.corr,
                        train_acc_ebv_indirect=train_ebv_indirect.corr,
                        val_acc_ebv_indirect=val_ebv_indirect.corr,
                        train_acc_ebv_direct=train_ebv_direct.corr,
                        val_acc_ebv_direct=val_ebv_direct.corr,
                        r_jwas_epv_train=r_jwas_train,
                        r_jwas_epv_val=r_jwas_val,
                        wall_time_sec=wall_time,
                    ),
                )

                println("Done $model_name: val EPV acc=$(round(val_epv.corr, digits=4)), val EBV acc=$(round(val_ebv.corr, digits=4)) (time=$(round(wall_time, digits=1))s)")
            end
        end
    end

    # ----------------------------
    # Write outputs
    # ----------------------------
    CSV.write(results_csv, results)

    # Build a compact markdown summary with 3×3 tables (val set).
    function grid_table(metric_col::Symbol; title::String)
        # matrix rows=train, cols=test, cell="omics / skip(latent1) / skip(no-latent)"
        header = "| Train \\\\ Test | 0% | 50% | 100% |\n|---:|---:|---:|---:|\n"
        body = ""
        for tr in train_levels
            row = "| $(tr)% |"
            for te in test_levels
                om = results[(results.train_missing_pct .== tr) .& (results.test_missing_pct .== te) .& (results.model .== "omics"), metric_col]
                sk = results[(results.train_missing_pct .== tr) .& (results.test_missing_pct .== te) .& (results.model .== "skip"), metric_col]
                sn = results[(results.train_missing_pct .== tr) .& (results.test_missing_pct .== te) .& (results.model .== "skip_nolatent"), metric_col]
                omv = isempty(om) ? NaN : om[1]
                skv = isempty(sk) ? NaN : sk[1]
                snv = isempty(sn) ? NaN : sn[1]
                cell = "$(round(omv, digits=4)) / $(round(skv, digits=4)) / $(round(snv, digits=4))"
                row *= " $cell |"
            end
            body *= row * "\n"
        end
        return "### $title (omics / skip(latent1) / skip(no-latent))\n\n" * header * body * "\n"
    end

    md = """
# Skip Connection 3×3 Grid (Real Data)

- Data: `TempTestData/nnmm_small_dataset/input_files/data1`
- Omics features: 20 observed; the default MiddleLayer includes an extra `latent1` node (all missing), and `skip_nolatent` removes it.
- MCMC: chain_length=$chain_length, burnin=$burnin, seed=$seed
- Models:
  - **omics**: `y = intercept + MiddleLayer`
  - **skip**:  `y = intercept + MiddleLayer + Genotypes` (multi-class BayesC via `class_priors`, MiddleLayer includes `latent1`)
  - **skip_nolatent**: same as **skip**, but MiddleLayer excludes `latent1` (only the 20 observed omics nodes)

The tables below report **validation accuracy** (correlation with true y on the validation IDs),
as `omics / skip(latent1) / skip(no-latent)` in each cell.

EBV definitions:
- `EBV_NonLinear` = `EBV_Indirect_NonLinear` + `EBV_Direct_Skip` (total EBV; direct term is nonzero only under skip models).

"""

    md *= grid_table(:val_acc_epv; title="Validation Acc: EPV_Output_NonLinear")
    md *= grid_table(:val_acc_ebv; title="Validation Acc: EBV_NonLinear (Total)")
    md *= grid_table(:val_acc_ebv_indirect; title="Validation Acc: EBV_Indirect_NonLinear")
    md *= grid_table(:val_acc_ebv_direct; title="Validation Acc: EBV_Direct_Skip")

    # Add JWAS sanity (complete case). Prefer the no-latent skip variant for reporting.
    complete_skip = results[(results.train_missing_pct .== 0) .& (results.test_missing_pct .== 0) .& (results.model .== "skip_nolatent"), :]
    if nrow(complete_skip) == 1
        rtr = complete_skip.r_jwas_epv_train[1]
        rval = complete_skip.r_jwas_epv_val[1]
        md *= """
## JWAS sanity (complete omics, skip_nolatent model)

- r(JWAS EBV, NNMM EPV_Output) train: $(round(rtr, digits=4))
- r(JWAS EBV, NNMM EPV_Output) val:   $(round(rval, digits=4))

"""
    end

    md_path = joinpath(out_dir, "summary.md")
    open(md_path, "w") do io
        write(io, md)
    end

    println("\nWrote:")
    println("  $results_csv")
    println("  $md_path")
end

main()

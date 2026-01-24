using Test
using NNMM
using DataFrames
using CSV
using Random
using Statistics

@testset "Skip Connection (Layer 1 → Layer 3)" begin
    Random.seed!(12345)

    n = 80
    p = 20

    ids = ["id_$i" for i in 1:n]
    X = rand([0.0, 1.0, 2.0], n, p)
    β = randn(p)
    y = X * β .+ randn(n) .* 0.1

    # Omics are unrelated to y (so any strong EPV signal should come from the skip term).
    omics_full = randn(n)

    mktempdir() do tmpdir
        geno_path = joinpath(tmpdir, "geno.csv")
        geno_df = DataFrame(ID=ids)
        for j in 1:p
            geno_df[!, "snp$j"] = X[:, j]
        end
        CSV.write(geno_path, geno_df)

        pheno_path = joinpath(tmpdir, "pheno.csv")
        CSV.write(pheno_path, DataFrame(ID=ids, trait1=y); missingstring="NA")

        make_layers(omics_path) = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_path, missing_value="NA"),
        ]

        make_eq1() = Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            traits=["omic1"],
            method="BayesC",
        )

        @testset "Skip improves EPV when omics are uninformative" begin
            omics_path = joinpath(tmpdir, "omics_full.csv")
            CSV.write(omics_path, DataFrame(ID=ids, omic1=omics_full); missingstring="NA")

            layers = make_layers(omics_path)

            # Use an intentionally-invalid global method to ensure class_priors are actually used.
            class_priors_omics_only = Dict(
                "omics" => (
                    method="BayesC",
                    Pi=0.9,
                    estimatePi=false,
                    G=false,
                    G_is_marker_variance=false,
                    df_G=4.0,
                    estimate_variance_G=true,
                    estimate_scale_G=false,
                    constraint_G=true,
                ),
            )

            class_priors_with_skip = Dict(
                "omics" => class_priors_omics_only["omics"],
                "geno" => (
                    method="RR-BLUP",
                    Pi=0.0,
                    estimatePi=false,
                    G=false,
                    G_is_marker_variance=false,
                    df_G=4.0,
                    estimate_variance_G=true,
                    estimate_scale_G=false,
                    constraint_G=true,
                ),
            )

            eq2_noskip = Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                traits=["trait1"],
                method="GBLUP", # should be ignored for "omics" due to class_priors
                activation_function="linear",
                class_priors=class_priors_omics_only,
            )

            eq2_skip = Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics + geno",
                traits=["trait1"],
                method="GBLUP", # should be ignored for "omics" due to class_priors
                activation_function="linear",
                class_priors=class_priors_with_skip,
            )

            out_noskip = joinpath(tmpdir, "out_noskip")
            res_noskip = runNNMM(
                layers,
                [make_eq1(), eq2_noskip];
                chain_length=20,
                burnin=0,
                output_samples_frequency=1,
                printout_model_info=false,
                printout_frequency=10^9,
                output_folder=out_noskip,
                seed=12345,
            )

            out_skip = joinpath(tmpdir, "out_skip")
            res_skip = runNNMM(
                make_layers(omics_path),
                [make_eq1(), eq2_skip];
                chain_length=20,
                burnin=0,
                output_samples_frequency=1,
                printout_model_info=false,
                printout_frequency=10^9,
                output_folder=out_skip,
                seed=12345,
            )

            @test haskey(res_noskip, "EPV_NonLinear")
            @test haskey(res_skip, "EPV_NonLinear")

            epv_noskip_df = res_noskip["EPV_NonLinear"]
            epv_skip_df = res_skip["EPV_NonLinear"]
            y_by_id = Dict(ids[i] => y[i] for i in 1:n)

            y_noskip = [y_by_id[string(id)] for id in epv_noskip_df.ID]
            y_skip = [y_by_id[string(id)] for id in epv_skip_df.ID]

            cor_noskip = cor(epv_noskip_df.EPV, y_noskip)
            cor_skip = cor(epv_skip_df.EPV, y_skip)

            @test isfinite(cor_skip)
            @test cor_skip > 0.5
            cor_noskip_val = isfinite(cor_noskip) ? cor_noskip : 0.0
            @test cor_skip > cor_noskip_val + 0.2
        end

        @testset "Skip runs with partial missing omics" begin
            omics_missing = Vector{Union{Missing, Float64}}(omics_full)
            omics_missing[1:10] .= missing
            omics_missing_path = joinpath(tmpdir, "omics_missing.csv")
            CSV.write(omics_missing_path, DataFrame(ID=ids, omic1=omics_missing); missingstring="NA")

            layers_missing = make_layers(omics_missing_path)

            class_priors_with_skip = Dict(
                "omics" => (
                    method="BayesC",
                    Pi=0.9,
                    estimatePi=false,
                    G=false,
                    G_is_marker_variance=false,
                    df_G=4.0,
                    estimate_variance_G=true,
                    estimate_scale_G=false,
                    constraint_G=true,
                ),
                "geno" => (
                    method="RR-BLUP",
                    Pi=0.0,
                    estimatePi=false,
                    G=false,
                    G_is_marker_variance=false,
                    df_G=4.0,
                    estimate_variance_G=true,
                    estimate_scale_G=false,
                    constraint_G=true,
                ),
            )

            eq2_skip = Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics + geno",
                traits=["trait1"],
                method="GBLUP", # should be ignored for "omics" due to class_priors
                activation_function="linear",
                class_priors=class_priors_with_skip,
            )

            out_missing = joinpath(tmpdir, "out_missing")
            res_missing = runNNMM(
                layers_missing,
                [make_eq1(), eq2_skip];
                chain_length=5,
                burnin=0,
                output_samples_frequency=1,
                printout_model_info=false,
                printout_frequency=10^9,
                output_folder=out_missing,
                seed=12345,
            )

            @test haskey(res_missing, "EPV_NonLinear")
            epv = res_missing["EPV_NonLinear"].EPV
            @test all(isfinite, epv)
        end
    end
end

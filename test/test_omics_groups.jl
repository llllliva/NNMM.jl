using DataFrames
using CSV
using Random

@testset "Grouped omics marker classes" begin
    Random.seed!(2026)

    mktempdir() do tmpdir
        ids = string.(1:24)

        geno = DataFrame(ID=ids)
        for j in 1:8
            geno[!, "m$j"] = rand(0:2, length(ids))
        end
        geno_path = joinpath(tmpdir, "geno.csv")
        CSV.write(geno_path, geno)

        omics = DataFrame(ID=ids)
        omics[!, "o1a"] = randn(length(ids))
        omics[!, "o1b"] = randn(length(ids))
        omics[!, "o2a"] = randn(length(ids))
        omics[!, "o2b"] = randn(length(ids))
        omics_path = joinpath(tmpdir, "omics.csv")
        CSV.write(omics_path, omics; missingstring="NA")

        pheno = DataFrame(
            ID=ids,
            y=0.8 .* omics.o1a .- 0.4 .* omics.o2b .+ 0.1 .* randn(length(ids)),
        )
        pheno_path = joinpath(tmpdir, "pheno.csv")
        CSV.write(pheno_path, pheno; missingstring="NA")

        groups = Dict(
            "omics1" => ["o1a", "o1b"],
            "omics2" => ["o2a", "o2b"],
        )
        class_priors = Dict(
            "omics1" => (method="BayesC", Pi=0.75, estimatePi=false),
            "omics2" => (method="BayesC", Pi=0.25, estimatePi=false),
            "geno" => (method="BayesC", Pi=0.0, estimatePi=false),
        )

        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_path, missing_value="NA"),
        ]
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + geno",
                traits=vcat(groups["omics1"], groups["omics2"]),
                method="BayesC",
                omics_groups=groups,
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics1 + omics2 + geno",
                traits=["y"],
                class_priors=class_priors,
                omics_groups=groups,
                activation_function="linear",
            ),
        ]

        result = runNNMM(
            layers,
            equations;
            chain_length=4,
            burnin=0,
            output_samples_frequency=1,
            output_prediction_frequency=1,
            printout_model_info=false,
            printout_frequency=10^9,
            output_folder=joinpath(tmpdir, "out"),
            seed=2026,
        )

        @test haskey(result, "EBV_NonLinear")
        @test haskey(result, "EPV_Output_NonLinear")
        @test nrow(result["EBV_NonLinear"]) == length(ids)
    end
end

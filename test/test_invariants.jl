using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Internal Invariants" begin
    # Use simulated omics dataset
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    pheno_df = CSV.read(pheno_path, DataFrame)

    # Reuse a small omics subset for faster tests
    omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3]]

    mktempdir() do tmpdir
        o_path = joinpath(tmpdir, "omics.csv")
        y_path = joinpath(tmpdir, "phenotypes.csv")
        outdir = joinpath(tmpdir, "out")

        CSV.write(o_path, omics_df; missingstring="NA")
        CSV.write(y_path, pheno_df[:, [:ID, :trait1]]; missingstring="NA")

        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA"),
        ]

        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + geno",
                traits=["omic1", "omic2", "omic3"],
                method="BayesC",
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

        @testset "Residual recomputation agrees (debug mode)" begin
            p = Pipe()
            withenv("NNMM_DEBUG_INVARIANTS" => "1", "NNMM_DEBUG_INVARIANTS_ITERS" => "2") do
                redirect_stdout(p) do
                    runNNMM(
                        layers,
                        equations;
                        chain_length=2,
                        burnin=0,
                        printout_model_info=false,
                        printout_frequency=10^9,
                        output_samples_frequency=1,
                        output_folder=outdir,
                    )
                end
            end

            close(p.in)
            out = read(p, String)

            ycorr1_vals = [parse(Float64, m.captures[1]) for m in eachmatch(r"ycorr1 maxabs\(check-current\)=([0-9.eE+-]+)", out)]
            ycorr2_vals = [parse(Float64, m.captures[1]) for m in eachmatch(r"ycorr2 maxabs\(check-current\)=([0-9.eE+-]+)", out)]

            @test !isempty(ycorr1_vals)
            @test !isempty(ycorr2_vals)
            @test maximum(ycorr1_vals) < 1e-3
            @test maximum(ycorr2_vals) < 1e-3
        end

        @testset "Prediction equation accepts marker terms" begin
            # Marker data is always included for now, so allowing "genotypes" here should not error.
            layers_pred = [
                Layer(layer_name="geno", data_path=[geno_path]),
                Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
                Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA"),
            ]
            equations_pred = [
                Equation(
                    from_layer_name="geno",
                    to_layer_name="omics",
                    equation="omics = intercept + geno",
                    traits=["omic1", "omic2", "omic3"],
                    method="BayesC",
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
            result = runNNMM(
                layers_pred,
                equations_pred;
                chain_length=1,
                burnin=0,
                prediction_equation="genotypes",
                printout_model_info=false,
                printout_frequency=10^9,
                output_samples_frequency=1,
                output_folder=joinpath(tmpdir, "out_pred"),
            )
            @test result !== nothing
            @test haskey(result, "EBV_NonLinear")
        end
    end
end

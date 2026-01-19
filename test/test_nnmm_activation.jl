using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Activation Functions" begin
    # Use simulated omics dataset
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "activation")
    mkpath(data_dir)
    
    # Load phenotype data and create test files
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    # Create omics file (use 3 omics for faster tests)
    omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3]]
    o_path = joinpath(data_dir, "omics.csv")
    CSV.write(o_path, omics_df; missingstring="NA")
    
    # Create phenotype file
    pheno_out_df = pheno_df[:, [:ID, :trait1]]
    y_path = joinpath(data_dir, "phenotypes.csv")
    CSV.write(y_path, pheno_out_df; missingstring="NA")
    
    # Test each activation function
    for activation in ["linear", "sigmoid"]
        @testset "Activation: $activation" begin
            layers = [
                Layer(layer_name="geno", data_path=[geno_path]),
                Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
                Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
            ]
            
            equations = [
                Equation(
                    from_layer_name="geno",
                    to_layer_name="omics",
                    equation="omics = intercept + geno",
                    traits=["omic1", "omic2", "omic3"],
                    method="BayesC"
                ),
                Equation(
                    from_layer_name="omics",
                    to_layer_name="phenotypes",
                    equation="phenotypes = intercept + omics",
                    traits=["trait1"],
                    method="BayesC",
                    activation_function=activation
                )
            ]
            
            # Verify equation has correct activation function
            @test equations[2].activation_function == activation
            
            # Run NNMM
            result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
            
            # Verify results
            @test result !== nothing
            @test haskey(result, "EBV_NonLinear")
            
            ebv_df = result["EBV_NonLinear"]
            @test nrow(ebv_df) > 3000
            @test all(!isnan, ebv_df.EBV)
        end
    end
    
    # Cleanup
    rm(data_dir, recursive=true, force=true)
end

using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV

@testset "Additional Bayesian Methods" begin
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    data_dir = joinpath(@__DIR__, "fixtures", "more_methods")
    mkpath(data_dir)
    
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    omics_df = pheno_df[:, [:ID, :omic1, :omic2]]
    o_path = joinpath(data_dir, "omics.csv")
    CSV.write(o_path, omics_df; missingstring="NA")
    
    pheno_out_df = pheno_df[:, [:ID, :trait1]]
    y_path = joinpath(data_dir, "phenotypes.csv")
    CSV.write(y_path, pheno_out_df; missingstring="NA")
    
    @testset "BayesB Method" begin
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
                traits=["omic1", "omic2"],
                method="BayesB",
                estimatePi=true
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                traits=["trait1"],
                method="BayesB",
                estimatePi=true
            )
        ]
        
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        ebv_df = result["EBV_NonLinear"]
        @test nrow(ebv_df) > 3000
        @test all(!isnan, ebv_df.EBV)
    end
    
    @testset "BayesL (Bayesian LASSO)" begin
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
                traits=["omic1", "omic2"],
                method="BayesL"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                traits=["trait1"],
                method="BayesL"
            )
        ]
        
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        ebv_df = result["EBV_NonLinear"]
        @test nrow(ebv_df) > 3000
        @test all(!isnan, ebv_df.EBV)
    end
    
    @testset "RR-BLUP Method" begin
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
                traits=["omic1", "omic2"],
                method="RR-BLUP"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                traits=["trait1"],
                method="RR-BLUP"
            )
        ]
        
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        ebv_df = result["EBV_NonLinear"]
        @test nrow(ebv_df) > 3000
        @test all(!isnan, ebv_df.EBV)
    end
    
    # Cleanup
    rm(data_dir, recursive=true, force=true)
end

@testset "Multi-Omics Analysis" begin
    # Test with multiple omics layers (single-trait phenotype)
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    data_dir = joinpath(@__DIR__, "fixtures", "multi_omics")
    mkpath(data_dir)
    
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    # Use 5 omics
    omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3, :omic4, :omic5]]
    o_path = joinpath(data_dir, "omics.csv")
    CSV.write(o_path, omics_df; missingstring="NA")
    
    # Single trait
    pheno_out_df = pheno_df[:, [:ID, :trait1]]
    y_path = joinpath(data_dir, "phenotypes.csv")
    CSV.write(y_path, pheno_out_df; missingstring="NA")
    
    @testset "Five-Omics Model (constraint=true)" begin
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
                traits=["omic1", "omic2", "omic3", "omic4", "omic5"],
                method="BayesC"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                traits=["trait1"],
                method="BayesC"
            )
        ]
        
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
    end
    
    # Cleanup
    rm(data_dir, recursive=true, force=true)
end


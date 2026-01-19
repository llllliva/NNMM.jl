using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random
using DelimitedFiles
using LinearAlgebra

@testset "NNMM Basic Module" begin
    # Use simulated omics dataset
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    # Setup test data directory for generated files
    test_data_dir = joinpath(@__DIR__, "fixtures", "basic")
    mkpath(test_data_dir)
    
    @testset "Layer construction" begin
        layer = Layer(
            layer_name="test",
            data_path="test.csv",
            separator=',',
            header=true
        )
        @test layer.layer_name == "test"
        @test layer.data_path == "test.csv"
        @test layer.MAF == 0.01
        @test layer.center == true
    end
    
    @testset "Equation construction" begin
        eq = Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            traits=["o1"]
        )
        @test eq.from_layer_name == "geno"
        @test eq.to_layer_name == "omics"
        @test eq.method == "BayesC"
        @test eq.Pi == 0.0
        @test eq.estimatePi == true
    end
    
    @testset "NNMM workflow with simulated data" begin
        # Read phenotypes to extract omics columns
        pheno_df = CSV.read(pheno_path, DataFrame)
        
        # Create omics file from simulated data (omic1-omic10)
        omics_cols = [:ID, :omic1, :omic2, :omic3]
        omics_df = pheno_df[:, omics_cols]
        omics_path = joinpath(test_data_dir, "test_omics.csv")
        CSV.write(omics_path, omics_df; missingstring="NA")
        
        # Create phenotype file
        pheno_out_df = pheno_df[:, [:ID, :trait1]]
        pheno_out_path = joinpath(test_data_dir, "test_phenotypes.csv")
        CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")
        
        # Define layers
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
        ]
        
        # Define equations
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
                method="BayesC"
            )
        ]
        
        # Run NNMM with short chain for testing
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        # Test that results are returned
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        
        # Test that EBV DataFrame has expected structure
        ebv_df = result["EBV_NonLinear"]
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        @test nrow(ebv_df) > 3000
        
        # Cleanup test files
        rm(omics_path, force=true)
        rm(pheno_out_path, force=true)
    end
    
    @testset "nnmm_get_genotypes function" begin
        geno = NNMM.nnmm_get_genotypes(geno_path)
        
        @test typeof(geno) == NNMM.Genotypes
        @test length(geno.obsID) > 3000
        @test geno.nMarkers > 900
        @test geno.nObs > 3000
        @test size(geno.genotypes, 1) == geno.nObs
        @test size(geno.genotypes, 2) == geno.nMarkers
    end
    
    # Cleanup
    rm(test_data_dir, recursive=true, force=true)
end

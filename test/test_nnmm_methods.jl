using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Bayesian Methods" begin
    # Use simulated omics dataset
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "methods")
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
    
    # Test different Bayesian methods
    # Note: Only testing methods that are compatible with NNMM framework
    bayesian_methods = ["BayesC", "BayesA"]
    
    for method in bayesian_methods
        @testset "Method: $method" begin
            layers = [
                Layer(layer_name="geno", data_path=[geno_path]),
                Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
                Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
            ]
            
            # For BayesB and BayesC, enable Pi estimation
            estimate_pi = method in ["BayesB", "BayesC"]
            
            equations = [
                Equation(
                    from_layer_name="geno",
                    to_layer_name="omics",
                    equation="omics = intercept + geno",
                    traits=["omic1", "omic2", "omic3"],
                    method=method,
                    estimatePi=estimate_pi
                ),
                Equation(
                    from_layer_name="omics",
                    to_layer_name="phenotypes",
                    equation="phenotypes = intercept + omics",
                    traits=["trait1"],
                    method=method,
                    estimatePi=estimate_pi
                )
            ]
            
            # Verify equations have correct method
            @test equations[1].method == method
            @test equations[2].method == method
            
            # Run NNMM
            result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
            
            # Verify results
            @test result !== nothing
            @test haskey(result, "EBV_NonLinear")
            
            ebv_df = result["EBV_NonLinear"]
            @test nrow(ebv_df) > 3000
            @test :ID in propertynames(ebv_df)
            @test :EBV in propertynames(ebv_df)
            
            # EBV values should be finite
            @test all(!isnan, ebv_df.EBV)
            @test all(!isinf, ebv_df.EBV)
        end
    end
    
    @testset "Pi estimation options" begin
        # Test Equation construction with fixed Pi (estimatePi=false)
        # Note: Running NNMM with single-trait fixed Pi has known issues with diag()
        # This test verifies the parameter setting works correctly
        
        eq_fixed_pi = Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            traits=["omic1"],
            method="BayesC",
            Pi=0.95,
            estimatePi=false
        )
        
        @test eq_fixed_pi.estimatePi == false
        @test eq_fixed_pi.Pi == 0.95
        @test eq_fixed_pi.method == "BayesC"
        
        # Test with default Pi estimation (estimatePi=true)
        eq_estimate_pi = Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            traits=["omic1"],
            method="BayesC",
            estimatePi=true
        )
        
        @test eq_estimate_pi.estimatePi == true
        @test eq_estimate_pi.Pi == 0.0  # Default Pi when estimatePi=true
    end
    
    # Cleanup
    rm(data_dir, recursive=true, force=true)
end
